from bs4 import BeautifulSoup
import pandas as pd

with open("html_table.txt", "r", encoding="utf-8") as file:
    html_content = file.read()

soup = BeautifulSoup(html_content, "lxml")
posts = soup.find_all("a")

data = []

for post in posts:
    try:
        date_span = post.find_all("span")[0].text.strip()

        post_text_span = post.find(
            "span", style=lambda value: value and "line-clamp" in value
        )
        post_text = post_text_span.text.strip() if post_text_span else ""

        metrics = post.find_all("div", class_="xyqm7xq")
        comments = metrics[0].text.strip() if len(metrics) > 0 else "0"
        likes = metrics[1].text.strip() if len(metrics) > 1 else "0"
        impressions = metrics[2].text.strip() if len(metrics) > 2 else "0"

        data.append(
            {
                "Date": date_span,
                "Post": post_text,
                "Comments": comments,
                "Reactions": likes,
                "Views": impressions,
            }
        )
    except Exception as e:
        continue

df = pd.DataFrame(data)
# df["Post Category"] = df["post"].apply(lambda x: classify_post(x) if isinstance(x, str) and x.strip() else "Unknown")

print(df.head(50))

# df.to_csv("hasil_facebook_posts.csv", index=False)
