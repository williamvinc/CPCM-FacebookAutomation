#!/usr/bin/env python3
"""
classification_posts.py

Pipeline to:
1) Download the Google Sheet as CSV (SHEET_ID + gid from env)
2) For rows where `Posts` is filled AND any of the target columns is blank
   (Post Category, Product Name, Product Category, Price, Type),
   call a Groq LLM to classify/extract values.
3) Write results to a local CSV and optionally back to the Google Sheet.

Logging:
- Logs every LLM call (row index + short hash + preview of Posts)
- Logs every rate-limit sleep to respect GROQ_MAX_RPM
"""

import os
import re
import json
import time
import logging
from typing import Dict, Any, List, Optional

import requests
import pandas as pd
from dotenv import load_dotenv

# Optional write-back to Google Sheet
try:
    import gspread
    from gspread_dataframe import set_with_dataframe

    HAS_GSPREAD = True
except Exception:
    HAS_GSPREAD = False

# ----------------------------
# Environment & basic logging
# ----------------------------
load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)

SHEET_ID = os.getenv("SHEET_ID", "").strip()
GID = os.getenv("gid", "").strip()
CSV_URL = (
    f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv&gid={GID}"
)

WRITE_BACK = os.getenv("WRITE_BACK", "false").lower() == "true"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# Use a model with generous free quota; you can override via env
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Rate limit: requests per minute
GROQ_MAX_RPM = int(os.getenv("GROQ_MAX_RPM", "25"))
_GROQ_INTERVAL = 60.0 / GROQ_MAX_RPM if GROQ_MAX_RPM > 0 else 0.0

TARGET_COLS = [
    "Post Category",
    "Product Name",
    "Product Category",
    "Price",
    "Type",
]

CATEGORY_CHOICES = ["Trading", "Buying", "Selling", "Sharing"]
PRODUCT_CATEGORY_CHOICES = [
    "Saldo",
    "Ticket",
    "Merchandise",
    "Card",
    "Coin",
    "Complain",
    "Sharing",
    "Membership Card",
]


# ----------------------------
# Utilities
# ----------------------------
def is_blank(v) -> bool:
    """True if value is NaN or empty string after strip."""
    return pd.isna(v) or (isinstance(v, str) and v.strip() == "")


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure target columns exist; if missing, create empty columns."""
    for c in TARGET_COLS:
        if c not in df.columns:
            df[c] = ""
    if "Posts" not in df.columns:
        raise ValueError("Column 'Posts' is required in the Sheet.")
    return df


def is_row_needs_llm(row: pd.Series) -> bool:
    """
    We only classify if:
      - 'Posts' is filled (non-blank)
      - at least one of TARGET_COLS is blank
    Special rule for Price: '0' means already filled (skip LLM).
    """
    if "Posts" not in row or is_blank(row["Posts"]):
        return False

    for c in TARGET_COLS:
        if c not in row:
            return True
        val = row[c]
        if c == "Price":
            # consider 0 (int or "0") as filled; blank means needs LLM
            if is_blank(val):
                return True
        else:
            if is_blank(val):
                return True
    return False


def extract_price_regex(posts_text: str) -> Optional[int]:
    """
    Try to extract an IDR price from text:
      - patterns like 'Rp 50.000', '50,000', '50k', '50 rb', 'IDR 75k'
    Returns integer or None if not found.
    """
    if not posts_text:
        return None

    text = posts_text.lower()
    patterns = [
        r"(?:rp\.?\s*|idr\s*)?(\d{1,3}(?:[.,]\d{3})+)",  # 50.000 / 50,000
        r"(?:rp\.?\s*|idr\s*)?(\d+)\s*rb\b",  # 50rb
        r"(?:rp\.?\s*|idr\s*)?(\d+)\s*[kK]\b",  # 50k
        r"(?:rp\.?\s*|idr\s*)?(\d{4,})",  # plain number >= 4 digits
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            raw = m.group(1).replace(",", "").replace(".", "")
            try:
                return int(raw)
            except Exception:
                continue

    m2 = re.search(r"(\d+)\s*[kK]\b", text)
    if m2:
        try:
            return int(m2.group(1)) * 1000
        except Exception:
            pass

    m3 = re.search(r"(\d+)\s*rb\b", text)
    if m3:
        try:
            return int(m3.group(1)) * 1000
        except Exception:
            pass

    return None


def brand_context() -> str:
    """
    Short brand context to assist the LLM. (Static, keep concise.)
    """
    return (
        "Cow Play Cow Moo (CPCM) is a large arcade brand originating from Singapore, "
        "also present in Indonesia. Common terms: balance/membership card, physical tickets, "
        "coins, licensed merchandise (e.g., Disney, Sanrio), trading/buy/sell posts (WTT/WTS/WTB), "
        "and sharing/information posts."
    )


def build_system_prompt() -> str:
    return (
        "You are a precise information extraction assistant. "
        "Given a Facebook group post text related to the Cow Play Cow Moo arcade ecosystem, "
        "classify it and extract the requested structured fields. "
        "Return ONLY a compact valid JSON object (no markdown)."
        "Return a SINGLE valid JSON object with EXACT keys only. "
        "Do NOT include any markdown, code fences, comments, or extra text."
    )


def _normalize_outputs(
    posts_text: str, result: Dict[str, Any], price_hint: Optional[int]
) -> Dict[str, Any]:
    """Validate/normalize LLM outputs with simple fallbacks."""
    # Category
    post_cat = str(result.get("Post Category", "")).strip()
    if post_cat not in CATEGORY_CHOICES:
        low = posts_text.lower()
        if any(k in low for k in ["wts", "jual", "sell"]):
            post_cat = "Selling"
        elif any(k in low for k in ["wtb", "beli", "buy"]):
            post_cat = "Buying"
        elif any(k in low for k in ["wtt", "tukar", "trade", "swap"]):
            post_cat = "Trading"
        else:
            post_cat = "Sharing"

    # Product Category
    prod_cat = str(result.get("Product Category", "")).strip()
    if prod_cat not in PRODUCT_CATEGORY_CHOICES:
        low = posts_text.lower()
        if any(k in low for k in ["ticket", "tiket"]):
            prod_cat = "Ticket"
        elif any(k in low for k in ["saldo", "topup", "top up", "balance"]):
            prod_cat = "Saldo"
        elif any(k in low for k in ["membership", "member card", "kartu anggota"]):
            prod_cat = "Membership Card"
        elif any(k in low for k in ["coin", "koin"]):
            prod_cat = "Coin"
        elif any(k in low for k in ["card", "trading card"]):
            prod_cat = "Card"
        elif any(k in low for k in ["complain", "komplain"]):
            prod_cat = "Complain"
        elif any(
            k in low
            for k in [
                "jual",
                "wts",
                "wtt",
                "wtb",
                "merch",
                "medal",
                "boneka",
                "figur",
                "merchandise",
            ]
        ):
            prod_cat = "Merchandise"
        else:
            prod_cat = "Sharing"

    # Product Name
    product_name = str(result.get("Product Name", "")).strip()

    # Type mirrors Category
    _type = str(result.get("Type", "")).strip() or post_cat

    # Price
    price_val = result.get("Price", 0)
    try:
        price_val = int(price_val)
    except Exception:
        price_val = 0
    if price_val == 0 and price_hint is not None:
        price_val = price_hint

    # --- NEW RULES ---
    low_posts = posts_text.lower()

    # If post mentions foil/metal/laser, enforce Product Category = Card
    if any(k in low_posts for k in ["foil", "metal", "laser"]):
        prod_cat = "Card"

    # If Category/Type is Sharing, blank out Product Name & Product Category
    # (use empty string as "null" to keep CSV/Sheet clean)
    if post_cat == "Sharing" or _type == "Sharing":
        product_name = ""
        prod_cat = ""

    return {
        "Post Category": post_cat,
        "Product Category": prod_cat,
        "Product Name": product_name,
        "Type": _type,
        "Price": price_val,
    }


def build_user_prompt(posts_text: str) -> str:
    schema = {
        "Post Category": "(one of: Trading, Buying, Selling, Sharing)",
        "Type": "identical to Post Category",
        "Product Name": "short item name inferred from the text",
        "Product Category": f"(one of: {', '.join(PRODUCT_CATEGORY_CHOICES)})",
        "Price": "integer in IDR; if absent → 0",
    }
    guide = (
        f"- Category choices: {', '.join(CATEGORY_CHOICES)}.\n"
        f"- Product Category choices: {', '.join(PRODUCT_CATEGORY_CHOICES)}.\n"
        "- 'Type' MUST equal 'Post Category'.\n"
        "- If user requests to buy → Buying; offering to sell → Selling; propose trade → Trading; "
        "general info/tips/story → Sharing.\n"
        "- Price examples: '50k' → 50000, '50rb' → 50000, 'Rp 50.000' → 50000.\n"
        "- IMPORTANT: If the post text contains the words 'foil', 'metal', or 'laser' "
        "then 'Product Category' MUST be 'Card'.\n"
        "- Keep values concise. No quotes in values.\n"
    )
    schema_str = json.dumps(schema, indent=2)
    return (
        f"{brand_context()}\n\n"
        f"POST TEXT:\n{posts_text}\n\n"
        "OUTPUT JSON KEYS (exactly these keys):\n"
        f"{schema_str}\n\n"
        f"GUIDELINES:\n{guide}\n"
        "Return ONLY a JSON object with those keys."
    )


# ----------------------------
# Groq client (simple sleep-based rate limit)
# ----------------------------
def groq_chat_complete(
    prompt: str, temperature: float = 0.0, max_tokens: int = 300
) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing in environment.")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    base_body = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Try with response_format first (best case)
    body = {**base_body, "response_format": {"type": "json_object"}}
    resp = requests.post(GROQ_API_URL, headers=headers, json=body, timeout=60)

    # Some Groq models may not support response_format → retry without it
    if resp.status_code == 400:
        # fallback: no response_format
        resp = requests.post(GROQ_API_URL, headers=headers, json=base_body, timeout=60)

    if resp.status_code == 200:
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    if resp.status_code == 429:
        raise RuntimeError(f"429 rate limit: {resp.text}")
    raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text}")


def classify_post_with_llm(
    posts_text: str, price_hint: Optional[int] = None
) -> Dict[str, Any]:
    hint = (
        f"\nHINT: Price candidate (IDR) from regex = {price_hint}\n"
        if price_hint is not None
        else ""
    )
    user_prompt = build_user_prompt(posts_text) + hint

    raw = groq_chat_complete(user_prompt, temperature=0.0, max_tokens=220)
    # Expect pure JSON; still guard against accidental fences:
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE | re.DOTALL
    )
    return json.loads(cleaned)


# ----------------------------
# Dataframe pipeline
# ----------------------------
def load_sheet_csv(url: str) -> pd.DataFrame:
    logging.info(f"Downloading sheet CSV: {url}")
    df = pd.read_csv(url)
    logging.info(f"Loaded shape: {df.shape}")
    return df


def _normalize_outputs(
    posts_text: str, result: Dict[str, Any], price_hint: Optional[int]
) -> Dict[str, Any]:
    """Validate/normalize LLM outputs with simple fallbacks."""
    # Category
    post_cat = str(result.get("Post Category", "")).strip()
    if post_cat not in CATEGORY_CHOICES:
        low = posts_text.lower()
        if any(k in low for k in ["wts", "jual", "sell"]):
            post_cat = "Selling"
        elif any(k in low for k in ["wtb", "beli", "buy"]):
            post_cat = "Buying"
        elif any(k in low for k in ["wtt", "tukar", "trade", "swap"]):
            post_cat = "Trading"
        else:
            post_cat = "Sharing"

    # Product Category
    prod_cat = str(result.get("Product Category", "")).strip()
    if prod_cat not in PRODUCT_CATEGORY_CHOICES:
        low = posts_text.lower()
        if any(k in low for k in ["ticket", "tiket"]):
            prod_cat = "Ticket"
        elif any(k in low for k in ["saldo", "topup", "top up", "balance"]):
            prod_cat = "Saldo"
        elif any(k in low for k in ["membership", "member card", "kartu anggota"]):
            prod_cat = "Membership Card"
        elif any(k in low for k in ["coin", "koin"]):
            prod_cat = "Coin"
        elif any(k in low for k in ["card", "trading card"]):
            prod_cat = "Card"
        elif any(k in low for k in ["complain", "komplain"]):
            prod_cat = "Complain"
        elif any(
            k in low
            for k in [
                "jual",
                "wts",
                "wtt",
                "wtb",
                "merch",
                "medal",
                "boneka",
                "figur",
                "merchandise",
            ]
        ):
            prod_cat = "Merchandise"
        else:
            prod_cat = "Sharing"

    # Product Name
    product_name = str(result.get("Product Name", "")).strip()

    # Type mirrors Category
    _type = str(result.get("Type", "")).strip() or post_cat

    # Price
    price_val = result.get("Price", 0)
    try:
        price_val = int(price_val)
    except Exception:
        price_val = 0
    if price_val == 0 and price_hint is not None:
        price_val = price_hint

    # --- NEW RULES ---
    low_posts = posts_text.lower()

    # If post mentions foil/metal/laser, enforce Product Category = Card
    if any(k in low_posts for k in ["foil", "metal", "laser"]):
        prod_cat = "Card"

    # If Category/Type is Sharing, blank out Product Name & Product Category
    # (use empty string as "null" to keep CSV/Sheet clean)
    if post_cat == "Sharing" or _type == "Sharing":
        product_name = ""
        prod_cat = ""

    return {
        "Post Category": post_cat,
        "Product Category": prod_cat,
        "Product Name": product_name,
        "Type": _type,
        "Price": price_val,
    }


def classify_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_columns(df).copy()

    # collect target row indices
    need_idx: List[int] = []
    for i, row in df.iterrows():
        if is_row_needs_llm(row):
            need_idx.append(i)

    logging.info(f"Rows needing LLM: {len(need_idx)}")

    last_call_ts = 0.0
    for i in need_idx:
        posts_text = str(df.at[i, "Posts"])
        price_hint = extract_price_regex(posts_text)

        # Respect RPM by sleeping between calls
        now = time.monotonic()
        wait = _GROQ_INTERVAL - (now - last_call_ts)
        if wait > 0:
            logging.info(
                f"Rate limiting: sleeping {wait:.2f} seconds to respect GROQ_MAX_RPM={GROQ_MAX_RPM}"
            )
            time.sleep(wait)

        # Log before calling LLM
        preview = posts_text.replace("\n", " ")[:80]
        logging.info(f"Calling LLM for row {i}: preview='{preview}...'")

        try:
            result = classify_post_with_llm(posts_text, price_hint=price_hint)
        except Exception as e:
            logging.error(f"LLM call failed for row {i}: {e}. Filling safe defaults.")
            # Fill safe defaults if call fails
            normalized = _normalize_outputs(posts_text, {}, price_hint)
        else:
            normalized = _normalize_outputs(posts_text, result, price_hint)

        # Write outputs into dataframe
        df.at[i, "Post Category"] = normalized["Post Category"]
        df.at[i, "Type"] = normalized["Type"]
        df.at[i, "Product Name"] = normalized["Product Name"]
        df.at[i, "Product Category"] = normalized["Product Category"]
        df.at[i, "Price"] = normalized["Price"]

        last_call_ts = time.monotonic()

    return df


def write_back_to_sheet(df: pd.DataFrame) -> None:
    if not WRITE_BACK:
        logging.info("WRITE_BACK=false → skip writing back to Google Sheet.")
        return
    if not HAS_GSPREAD:
        raise RuntimeError(
            "gspread not installed. `pip install gspread gspread-dataframe`"
        )

    logging.info("Writing classified dataframe back to Google Sheet...")
    gc = gspread.service_account(filename="creds.json")
    sh = gc.open_by_key(SHEET_ID)
    ws = sh.worksheet("Posts")
    ws.clear()
    set_with_dataframe(ws, df, include_index=False, include_column_header=True)
    logging.info("Finished updating Google Sheet.")


def run() -> pd.DataFrame:
    """Entry point to run the classification pipeline."""
    if not SHEET_ID or not GID:
        raise RuntimeError("SHEET_ID or gid is empty. Set them in your .env file.")
    if not GROQ_API_KEY:
        logging.warning("GROQ_API_KEY is empty. LLM calls will fail.")

    df = load_sheet_csv(CSV_URL)

    # Ensure expected column order (create if missing to avoid KeyErrors)
    desired_order = [
        "Date",
        "Post Category",
        "Product Name",
        "Product Category",
        "Price",
        "Type",
        "Mark",
        "Posts",
        "Member",
        "Comments",
        "Reactions",
        "Views",
        "Link",
        "is_from_merged_xml_xls",
    ]
    for c in desired_order:
        if c not in df.columns:
            df[c] = ""
    df = df[desired_order]

    # Classify missing rows
    df_cls = classify_dataframe(df)

    # Save locally
    out_csv = "classified_posts.csv"
    df_cls.to_csv(out_csv, index=False, encoding="utf-8-sig")
    logging.info(f"Saved: {out_csv}")

    # Optionally write back
    write_back_to_sheet(df_cls)
    return df_cls


# For direct execution: python classification_posts.py
if __name__ == "__main__":
    run()
