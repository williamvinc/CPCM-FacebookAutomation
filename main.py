import pandas as pd
from bs4 import BeautifulSoup
import pandas as pd
import gspread
from gspread_dataframe import set_with_dataframe
from dotenv import load_dotenv
import os

load_dotenv()

################################ Start Of XML Reader ################################

with open("html_table.txt", "r", encoding="utf-8") as file:
    html_content = file.read()

soup = BeautifulSoup(html_content, "lxml")
posts = soup.find_all("a")

data = []

for post in posts:
    try:
        date_span = post.find_all("span")[0].text.strip()

        post_text_span = post.find("span", style=lambda value: value and "line-clamp" in value)
        post_text = post_text_span.text.strip() if post_text_span else ""

        metrics = post.find_all("div", class_="xyqm7xq")
        comments = metrics[0].text.strip() if len(metrics) > 0 else "0"
        likes = metrics[1].text.strip() if len(metrics) > 1 else "0"
        impressions = metrics[2].text.strip() if len(metrics) > 2 else "0"

        data.append({
            "Date": date_span,
            "Posts": post_text,
            "Comments": comments,
            "Reactions": likes,
            "Views": impressions
        })
    except Exception as e:
        continue

df = pd.DataFrame(data)
################################ End Of XML Reader ################################

################################ Start Facebook XLS Reader ################################

file_path = 'Facebook_Group_Insights_7-30-2025.xlsx'
excel_file = pd.ExcelFile(file_path, engine="openpyxl")

# Step 2: Ambil semua sheet ke dalam dictionary
dataframes = {}
for sheet_name in excel_file.sheet_names:
    dataframes[sheet_name] = pd.read_excel(excel_file, sheet_name=sheet_name)

# Step 3: Assign ke variabel individual
daily_numbers_df = dataframes['Daily numbers']
popular_days_df = dataframes['Popular days']
popular_times_df = dataframes['Popular times']
top_posts_df = dataframes['Top posts (last 28 days)']
age_gender_df = dataframes['Age and gender of members']
town_city_df = dataframes['Town or city of members']
country_df = dataframes['Country of members']
contributors_df = dataframes['Contributors (last 28 days)']
admins_moderators_df = dataframes['Admins and moderators']

################################ End Facebook XLS Reader ################################

################################ Start Facebook Merge XLS  ################################

merged_xml_xls = pd.merge(
    top_posts_df,
    df[['Posts', 'Date']],
    on='Posts',
    how='left'
)

print(merged_xml_xls.head(10))

print("XML Columns: ", df.columns.tolist())
print("XLS Columns: ", top_posts_df.columns.tolist())
print("Merged Columns: ", merged_xml_xls.columns.tolist())
print("Merged DataFrame Shape:", merged_xml_xls.shape)
print("top_posts_df Shape:", top_posts_df.shape)

################################ End Facebook Merge XLS  ################################

################################ Start Read Posts Sheet  ################################

sheet_id = os.getenv("SHEET_ID", "")
gid = os.getenv("gid", "")
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
df_sheet_posts = pd.read_csv(url)

print("############################## Begin Merge Posts ################################")
print("Posts Sheet Columns: ", df_sheet_posts.columns.tolist())
print("Posts Sheet Shape: ", df_sheet_posts.shape)

def clean_text(text):
    return str(text).strip().lower()

df_sheet_posts['Posts_clean'] = df_sheet_posts['Posts'].apply(clean_text)
merged_xml_xls['Posts_clean'] = merged_xml_xls['Posts'].apply(clean_text)

columns_to_update = ['Date', 'Member', 'Comments', 'Reactions', 'Views', 'Link']

df_updated = df_sheet_posts.merge(
    merged_xml_xls[['Posts_clean'] + columns_to_update],
    on='Posts_clean',
    how='left',
    suffixes=('', '_new')
)

for col in columns_to_update:
    if f"{col}_new" in df_updated.columns:
        df_updated[col] = df_updated[f"{col}_new"].combine_first(df_updated[col])
        df_updated.drop(columns=[f"{col}_new"], inplace=True)

existing_posts = set(df_sheet_posts['Posts_clean'])
new_rows = merged_xml_xls[~merged_xml_xls['Posts_clean'].isin(existing_posts)].copy()

for col in df_updated.columns:
    if col not in new_rows.columns and col != 'Posts_clean':
        new_rows[col] = pd.NA
new_rows = new_rows[df_updated.columns]
final_merged_df = pd.concat([df_updated, new_rows], ignore_index=True)
final_merged_df.drop(columns=['Posts_clean'], inplace=True, errors='ignore')
desired_order = [
    'Date', 'Post Category', 'Product Name', 'Product Category', 'Price', 'Type', 'Mark',
    'Posts', 'Member', 'Comments', 'Reactions', 'Views', 'Link'
]
final_merged_df = final_merged_df[[col for col in desired_order if col in final_merged_df.columns]]

print("Final DataFrame shape:", final_merged_df.shape)
print("Columns:", final_merged_df.columns.tolist())
print(final_merged_df.head())

cols_to_int = ['Comments', 'Reactions', 'Views', 'Price']

for col in cols_to_int:
    if col in final_merged_df.columns:
        final_merged_df[col] = pd.to_numeric(final_merged_df[col], errors='coerce').astype('Int64')

final_merged_df['Date'] = pd.to_datetime(final_merged_df['Date'], format="%d/%m/%Y", errors='coerce')
final_merged_df = final_merged_df.sort_values(by='Date').reset_index(drop=True)
final_merged_df['Date'] = final_merged_df['Date'].dt.strftime('%d/%m/%Y')


final_merged_df.to_csv("final_merged_posts.csv", index=False, encoding='utf-8-sig', sep=';')

gc = gspread.service_account(filename="creds.json")
sheet = gc.open_by_key(sheet_id)
worksheet = sheet.worksheet("Posts")
worksheet.clear()
set_with_dataframe(worksheet, final_merged_df, include_index=False, include_column_header=True)

print("Success Write to Google Sheet 'Posts'")

################################ End of Write Posts Sheet  ################################