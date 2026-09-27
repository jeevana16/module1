# ============================================================
# ZEpto DATA PIPELINE - COMPLETE ONE CELL
# ============================================================

import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import os
from urllib.parse import urljoin


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = "https://books.toscrape.com/"
GBP_TO_INR = 105.50

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

DB_FILE = "books_catalog.db"
CSV_FILE = "books_cleaned.csv"
QUERY_OUTPUT_FILE = "sql_query_outputs.txt"


# ============================================================
# REQUEST SESSION
# ============================================================

session = requests.Session()
session.headers.update(HEADERS)


# ============================================================
# 1. GET CATEGORY URLS
# ============================================================

def get_category_urls():

    response = session.get(
        BASE_URL,
        timeout=20
    )

    print("Homepage status code:", response.status_code)

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    category_urls = {}

    for link in soup.select(
        "div.side_categories ul li ul li a"
    ):

        category_name = link.get_text(
            strip=True
        )

        href = link.get("href")

        if href:

            category_url = urljoin(
                BASE_URL,
                href
            )

            category_urls[
                category_name
            ] = category_url

    return category_urls


# ============================================================
# 2. SCRAPE ONE CATEGORY
# ============================================================

def scrape_category(
    category_name,
    category_url
):

    books = []

    current_url = category_url
    page_number = 1

    while current_url:

        print(
            f"\n{category_name} - page {page_number}"
        )

        response = session.get(
            current_url,
            timeout=20
        )

        print(
            "Status:",
            response.status_code
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        products = soup.select(
            "article.product_pod"
        )

        print(
            "Books found:",
            len(products)
        )

        for product in products:

            # ------------------------------------------------
            # TITLE
            # ------------------------------------------------

            title_tag = product.select_one(
                "h3 a"
            )

            if title_tag:

                title = title_tag.get(
                    "title"
                )

                if not title:

                    title = title_tag.get_text(
                        strip=True
                    )

            else:

                title = None


            # ------------------------------------------------
            # PRICE
            # ------------------------------------------------

            price_tag = product.select_one(
                "p.price_color"
            )

            if price_tag:

                price = price_tag.get_text(
                    strip=True
                )

            else:

                price = None


            # ------------------------------------------------
            # STAR RATING
            # ------------------------------------------------

            rating_tag = product.select_one(
                "p.star-rating"
            )

            star_rating = None

            if rating_tag:

                classes = rating_tag.get(
                    "class",
                    []
                )

                for rating in [
                    "One",
                    "Two",
                    "Three",
                    "Four",
                    "Five"
                ]:

                    if rating in classes:

                        star_rating = rating
                        break


            # ------------------------------------------------
            # AVAILABILITY
            # ------------------------------------------------

            availability_tag = product.select_one(
                "p.instock.availability"
            )

            if availability_tag:

                availability = availability_tag.get_text(
                    " ",
                    strip=True
                )

            else:

                availability_tag = product.select_one(
                    ".availability"
                )

                if availability_tag:

                    availability = availability_tag.get_text(
                        " ",
                        strip=True
                    )

                else:

                    availability = None


            # ------------------------------------------------
            # STORE BOOK
            # ------------------------------------------------

            books.append({

                "title": title,

                "price": price,

                "star_rating": star_rating,

                "availability": availability,

                "category": category_name

            })


        # ----------------------------------------------------
        # NEXT PAGE
        # ----------------------------------------------------

        next_button = soup.select_one(
            "li.next a"
        )

        if next_button:

            next_href = next_button.get(
                "href"
            )

            current_url = urljoin(
                current_url,
                next_href
            )

            page_number += 1

        else:

            current_url = None


    return books


# ============================================================
# 3. GET ALL CATEGORIES
# ============================================================

print("\n" + "=" * 70)
print("GETTING CATEGORY LINKS")
print("=" * 70)

category_urls = get_category_urls()


print("\nCategories found:")

for category, url in category_urls.items():

    print(
        f"{category} -> {url}"
    )


# ============================================================
# 4. SELECT 4 CATEGORIES
# ============================================================

wanted_categories = [
    "Travel",
    "Mystery",
    "Historical Fiction",
    "Science Fiction"
]


selected_categories = [

    category

    for category in wanted_categories

    if category in category_urls

]


# Fallback

if len(selected_categories) < 3:

    selected_categories = list(
        category_urls.keys()
    )[:4]


print("\nSelected categories:")

print(
    selected_categories
)


# ============================================================
# 5. SCRAPE
# ============================================================

print("\n" + "=" * 70)
print("STARTING SCRAPING")
print("=" * 70)


all_books = []


for category in selected_categories:

    category_books = scrape_category(

        category,

        category_urls[category]

    )

    print(
        f"\n{category}: "
        f"{len(category_books)} books"
    )

    all_books.extend(
        category_books
    )


# ============================================================
# 6. DATAFRAME
# ============================================================

df = pd.DataFrame(
    all_books
)


if df.empty:

    raise ValueError(
        "No books were scraped."
    )


print("\n" + "=" * 70)
print("RAW DATA")
print("=" * 70)

print(
    "Total rows:",
    len(df)
)

print(
    "\nFirst 10 rows:"
)

print(
    df.head(10)
)


# ============================================================
# 7. CLEAN PRICE
# ============================================================

def clean_price(value):

    if value is None:

        return None

    try:

        value = str(value).strip()

        value = value.replace(
            "£",
            ""
        )

        value = value.replace(
            "Â",
            ""
        )

        value = value.strip()

        return float(value)

    except (
        ValueError,
        TypeError
    ):

        return None


df["price_gbp"] = (
    df["price"]
    .apply(clean_price)
)


print("\n" + "=" * 70)
print("PRICE CHECK")
print("=" * 70)

print(
    df["price"].head(10).tolist()
)

print(
    df["price_gbp"].head(10).tolist()
)

print(
    "Invalid prices:",
    df["price_gbp"].isna().sum()
)


# ============================================================
# 8. CLEAN RATING
# ============================================================

rating_map = {

    "One": 1,

    "Two": 2,

    "Three": 3,

    "Four": 4,

    "Five": 5

}


df["rating"] = (
    df["star_rating"]
    .map(rating_map)
)


# ============================================================
# 9. CLEAN AVAILABILITY
# ============================================================

def clean_stock(value):

    if value is None:

        return None

    value = str(value).lower()

    if "in stock" in value:

        return True

    if "out of stock" in value:

        return False

    return None


df["in_stock"] = (
    df["availability"]
    .apply(clean_stock)
)


# ============================================================
# 10. MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

print(
    df[
        [
            "price_gbp",
            "rating",
            "in_stock"
        ]
    ].isna().sum()
)


# ============================================================
# 11. MEDIAN IMPUTATION
# ============================================================

# PRICE

if df["price_gbp"].isna().any():

    median_price = (
        df["price_gbp"]
        .median()
    )

    if pd.isna(median_price):

        raise ValueError(
            "All prices failed to parse."
        )

    df["price_gbp"] = (
        df["price_gbp"]
        .fillna(median_price)
    )


# RATING

if df["rating"].isna().any():

    median_rating = (
        df["rating"]
        .median()
    )

    if pd.isna(median_rating):

        raise ValueError(
            "All ratings failed to parse."
        )

    df["rating"] = (
        df["rating"]
        .fillna(
            round(median_rating)
        )
    )


# ============================================================
# 12. DROP INVALID REQUIRED ROWS
# ============================================================

df = df.dropna(

    subset=[
        "title",
        "price_gbp",
        "rating",
        "in_stock",
        "category"
    ]

)


# ============================================================
# 13. DATA TYPES
# ============================================================

df["price_gbp"] = (
    df["price_gbp"]
    .astype(float)
)

df["rating"] = (
    df["rating"]
    .astype(int)
)

df["in_stock"] = (
    df["in_stock"]
    .astype(bool)
)


# ============================================================
# 14. GBP → INR
# ============================================================

df["price_inr"] = (

    df["price_gbp"]

    * GBP_TO_INR

).round(2)


# ============================================================
# 15. FINAL DATAFRAME
# ============================================================

df = df[

    [
        "title",
        "price_gbp",
        "price_inr",
        "rating",
        "in_stock",
        "category"
    ]

].reset_index(
    drop=True
)


# ============================================================
# 16. VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET")
print("=" * 70)

print(
    "Total books:",
    len(df)
)

print(
    "Total categories:",
    df["category"].nunique()
)

print(
    "\nData types:"
)

print(
    df.dtypes
)

print(
    "\nCategory counts:"
)

print(
    df["category"].value_counts()
)

print(
    "\nFirst 10 rows:"
)

print(
    df.head(10)
)


if len(df) < 60:

    raise ValueError(
        f"Only {len(df)} books found. "
        "Requirement is at least 60."
    )


if df["category"].nunique() < 3:

    raise ValueError(
        "Requirement is at least 3 categories."
    )


# ============================================================
# 17. SAVE CSV
# ============================================================

df.to_csv(
    CSV_FILE,
    index=False
)

print(
    "\nCSV saved:",
    CSV_FILE
)


# ============================================================
# 18. CREATE SQLITE DATABASE
# ============================================================

if os.path.exists(DB_FILE):

    os.remove(
        DB_FILE
    )


conn = sqlite3.connect(
    DB_FILE
)

cursor = conn.cursor()


# Enable foreign keys

cursor.execute(
    "PRAGMA foreign_keys = ON;"
)


# ============================================================
# 19. CATEGORIES TABLE
# ============================================================

cursor.execute("""

CREATE TABLE categories (

    category_id INTEGER PRIMARY KEY AUTOINCREMENT,

    category_name TEXT UNIQUE NOT NULL

);

""")


# ============================================================
# 20. BOOKS TABLE
# ============================================================

cursor.execute("""

CREATE TABLE books (

    book_id INTEGER PRIMARY KEY AUTOINCREMENT,

    title TEXT NOT NULL,

    price_gbp REAL NOT NULL,

    price_inr REAL NOT NULL,

    rating INTEGER NOT NULL,

    in_stock INTEGER NOT NULL,

    category_id INTEGER NOT NULL,

    FOREIGN KEY (category_id)

        REFERENCES categories(category_id)

);

""")


# ============================================================
# 21. NORMALIZE CATEGORY
# ============================================================

df["category"] = (
    df["category"]
    .astype(str)
    .str.strip()
)


# ============================================================
# 22. INSERT CATEGORIES
# ============================================================

categories = sorted(
    df["category"].unique()
)


for category in categories:

    cursor.execute(

        """

        INSERT OR IGNORE INTO categories

        (category_name)

        VALUES (?)

        """,

        (category,)

    )


conn.commit()


# ============================================================
# 23. CORRECT CATEGORY LOOKUP
# ============================================================

cursor.execute("""

SELECT
    category_id,
    category_name

FROM categories

""")


category_rows = cursor.fetchall()


# VERY IMPORTANT:
#
# Database returns:
#
# [(1, "Travel"), (2, "Mystery")]
#
# We create:
#
# {"Travel": 1, "Mystery": 2}


category_lookup = {

    category_name: category_id

    for category_id, category_name
    in category_rows

}


print("\n" + "=" * 70)
print("CATEGORY LOOKUP")
print("=" * 70)

print(
    category_lookup
)


# ============================================================
# 24. INSERT BOOKS
# ============================================================

for _, row in df.iterrows():

    category_name = (
        str(row["category"])
        .strip()
    )


    if category_name not in category_lookup:

        raise ValueError(
            f"Category not found: "
            f"{category_name}"
        )


    category_id = category_lookup[
        category_name
    ]


    cursor.execute(

        """

        INSERT INTO books

        (

            title,

            price_gbp,

            price_inr,

            rating,

            in_stock,

            category_id

        )

        VALUES (?, ?, ?, ?, ?, ?)

        """,

        (

            row["title"],

            float(row["price_gbp"]),

            float(row["price_inr"]),

            int(row["rating"]),

            int(row["in_stock"]),

            int(category_id)

        )

    )


conn.commit()


print(
    "\nBooks inserted successfully."
)


# ============================================================
# 25. CHECK DATABASE
# ============================================================

cursor.execute(
    "SELECT COUNT(*) FROM books"
)

database_book_count = (
    cursor.fetchone()[0]
)


cursor.execute(
    "SELECT COUNT(*) FROM categories"
)

database_category_count = (
    cursor.fetchone()[0]
)


print(
    "\nBooks in database:",
    database_book_count
)

print(
    "Categories in database:",
    database_category_count
)


# ============================================================
# 26. SQL QUERIES
# ============================================================

queries = {

    # SELECT + WHERE

    "Query 1 - SELECT WHERE": """

    SELECT
        title,
        price_gbp,
        rating

    FROM books

    WHERE rating >= 4;

    """,


    # ORDER BY + LIMIT

    "Query 2 - ORDER BY LIMIT": """

    SELECT
        title,
        price_gbp,
        rating

    FROM books

    ORDER BY price_gbp DESC

    LIMIT 10;

    """,


    # DISTINCT

    "Query 3 - DISTINCT": """

    SELECT DISTINCT
        category_name

    FROM categories

    ORDER BY category_name;

    """,


    # BETWEEN

    "Query 4 - BETWEEN": """

    SELECT
        title,
        price_gbp,
        price_inr

    FROM books

    WHERE price_gbp BETWEEN 20 AND 40

    ORDER BY price_gbp;

    """,


    # IN

    "Query 5 - IN": """

    SELECT
        title,
        rating,
        in_stock

    FROM books

    WHERE rating IN (4, 5)

    ORDER BY rating DESC;

    """,


    # JOIN

    "Query 6 - JOIN": """

    SELECT
        b.title,
        c.category_name,
        b.price_gbp,
        b.price_inr,
        b.rating,
        b.in_stock

    FROM books AS b

    JOIN categories AS c

        ON b.category_id = c.category_id

    ORDER BY
        b.rating DESC,
        b.price_gbp DESC

    LIMIT 10;

    """
}


# ============================================================
# 27. EXECUTE AND SAVE SQL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SQL QUERY RESULTS")
print("=" * 70)


with open(
    QUERY_OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:


    for name, query in queries.items():

        print(
            "\n" + "=" * 70
        )

        print(
            name
        )

        print(
            "=" * 70
        )

        print(
            query.strip()
        )


        result = pd.read_sql(
            query,
            conn
        )


        print(
            "\nOUTPUT:"
        )

        print(
            result.to_string(
                index=False
            )
        )


        # Save query

        file.write(
            "\n" + "=" * 70 + "\n"
        )

        file.write(
            name + "\n"
        )

        file.write(
            "=" * 70 + "\n"
        )

        file.write(
            query.strip() + "\n\n"
        )


        # Save output

        file.write(
            result.to_string(
                index=False
            )
        )

        file.write(
            "\n\n"
        )


print(
    "\nSQL outputs saved to:",
    QUERY_OUTPUT_FILE
)


# ============================================================
# 28. pd.read_sql()
# ============================================================

df_sql_1 = pd.read_sql(

    queries[
        "Query 1 - SELECT WHERE"
    ],

    conn

)


df_sql_2 = pd.read_sql(

    queries[
        "Query 2 - ORDER BY LIMIT"
    ],

    conn

)


df_sql_join = pd.read_sql(

    queries[
        "Query 6 - JOIN"
    ],

    conn

)


print("\n" + "=" * 70)
print("pd.read_sql() - QUERY 1")
print("=" * 70)

print(
    df_sql_1
)


print("\n" + "=" * 70)
print("pd.read_sql() - QUERY 2")
print("=" * 70)

print(
    df_sql_2
)


print("\n" + "=" * 70)
print("pd.read_sql() - JOIN")
print("=" * 70)

print(
    df_sql_join
)


# ============================================================
# 29. REPRODUCE JOIN USING pd.merge()
# ============================================================

print("\n" + "=" * 70)
print("REPRODUCING JOIN USING pd.merge()")
print("=" * 70)


# ------------------------------------------------------------
# Books DataFrame
# ------------------------------------------------------------

books_df = df.copy()


books_df["category"] = (
    books_df["category"]
    .astype(str)
    .str.strip()
)


# ------------------------------------------------------------
# Add category_id
# ------------------------------------------------------------

books_df["category_id"] = (

    books_df["category"]

    .map(
        category_lookup
    )

)


# Check

if books_df["category_id"].isna().any():

    print(
        "\nMissing category IDs:"
    )

    print(
        books_df[
            books_df["category_id"].isna()
        ]["category"].unique()
    )

    raise ValueError(
        "Category mapping failed."
    )


books_df["category_id"] = (
    books_df["category_id"]
    .astype(int)
)


# ------------------------------------------------------------
# Categories DataFrame
# ------------------------------------------------------------

categories_df = pd.DataFrame(

    category_rows,

    columns=[
        "category_id",
        "category_name"
    ]

)


categories_df["category_id"] = (
    categories_df["category_id"]
    .astype(int)
)


categories_df["category_name"] = (
    categories_df["category_name"]
    .astype(str)
    .str.strip()
)


# ------------------------------------------------------------
# PANDAS MERGE
# ------------------------------------------------------------

df_merge_join = pd.merge(

    books_df,

    categories_df,

    on="category_id",

    how="inner"

)


# ------------------------------------------------------------
# Select same columns as SQL
# ------------------------------------------------------------

df_merge_join = df_merge_join[

    [
        "title",
        "category_name",
        "price_gbp",
        "price_inr",
        "rating",
        "in_stock"
    ]

]


# ------------------------------------------------------------
# Same ORDER BY and LIMIT
# ------------------------------------------------------------

df_merge_join = (

    df_merge_join

    .sort_values(

        by=[
            "rating",
            "price_gbp"
        ],

        ascending=[
            False,
            False
        ]

    )

    .head(10)

    .reset_index(
        drop=True
    )

)


# ============================================================
# 30. PREPARE SQL JOIN
# ============================================================

df_sql_join = (
    df_sql_join
    .reset_index(
        drop=True
    )
)


# SQLite BOOLEAN = INTEGER

df_sql_join["in_stock"] = (
    df_sql_join["in_stock"]
    .astype(bool)
)


# ============================================================
# 31. DISPLAY BOTH
# ============================================================

print("\n" + "=" * 70)
print("SQL JOIN RESULT")
print("=" * 70)

print(
    df_sql_join.to_string(
        index=False
    )
)


print("\n" + "=" * 70)
print("pd.merge() RESULT")
print("=" * 70)

print(
    df_merge_join.to_string(
        index=False
    )
)


# ============================================================
# 32. COMPARE RESULTS
# ============================================================

sql_compare = df_sql_join.copy()

merge_compare = df_merge_join.copy()


# Make data types identical

sql_compare["price_gbp"] = (
    sql_compare["price_gbp"]
    .astype(float)
)

merge_compare["price_gbp"] = (
    merge_compare["price_gbp"]
    .astype(float)
)


sql_compare["price_inr"] = (
    sql_compare["price_inr"]
    .astype(float)
)

merge_compare["price_inr"] = (
    merge_compare["price_inr"]
    .astype(float)
)


sql_compare["rating"] = (
    sql_compare["rating"]
    .astype(int)
)

merge_compare["rating"] = (
    merge_compare["rating"]
    .astype(int)
)


sql_compare["in_stock"] = (
    sql_compare["in_stock"]
    .astype(bool)
)

merge_compare["in_stock"] = (
    merge_compare["in_stock"]
    .astype(bool)
)


# Same columns

merge_compare = merge_compare[
    sql_compare.columns
]


# Reset index

sql_compare = (
    sql_compare
    .reset_index(drop=True)
)

merge_compare = (
    merge_compare
    .reset_index(drop=True)
)


# ============================================================
# 33. FINAL COMPARISON
# ============================================================

equivalent = sql_compare.equals(
    merge_compare
)


print("\n" + "=" * 70)
print(
    "SQL JOIN == pd.merge():",
    equivalent
)
print("=" * 70)


if equivalent:

    print(
        "\nSUCCESS!"
    )

    print(
        "SQL JOIN and pd.merge() "
        "produced equivalent output."
    )

else:

    print(
        "\nWARNING!"
    )

    print(
        "SQL JOIN and pd.merge() "
        "are not equivalent."
    )


# ============================================================
# 34. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)

print(
    "Total books:",
    len(df)
)

print(
    "Total categories:",
    df["category"].nunique()
)

print(
    "Database books:",
    database_book_count
)

print(
    "Database categories:",
    database_category_count
)

print(
    "Currency rate:",
    "1 GBP = 105.50 INR"
)

print(
    "CSV file:",
    CSV_FILE
)

print(
    "SQLite file:",
    DB_FILE
)

print(
    "SQL output file:",
    QUERY_OUTPUT_FILE
)

print(
    "JOIN comparison:",
    equivalent
)


# ============================================================
# 35. CLOSE DATABASE
# ============================================================

conn.close()

print(
    "\nAll tasks completed successfully."
)