 Model 1 - Data Pipeline
 --------------------------------

 hello ,  This is my Module 1 submission for  the data pipeline  task. 

 The task was to scrape live product data from a public practice site,
clean it up, convert prices into INR using a fixed rate, store it in a
SQLite database, and then query it two ways - with SQL and with pandas -
to show both give the same answer.

The full pipeline is: scrape, clean, convert, store, query.

1. What to install :---
1.You need Python 3.9 or newer and three libraries:
2.pip install requests beautifulsoup4 pandas
3.SQLite comes with Python already, so nothing else to install.

2. How to run it :--------
All commands are run from inside the data_pipeline folder:
cd data_pipeline
    python scrape.py
    python clean_and_load.py
    python load_db.py
    python queries.py
    python task6_compare.py
Run them in order. Each script rebuilds its own output from scratch.


3. What is in this folder :------------------

scrape.py   :---- scrapes 5 catalogue listing pages from books.toscrape.com
clean_and_load.py :--cleans raw fields, converts GBP to INR, saves cleaned CSV
load_db.py :----creates the two table SQLite schema and inserts data
queries.py  :--- runs 6 SQL queries covering every required clause
task6_compare.py  :---compares the SQL join result against pandas merge
raw_books.csv    :---raw scrape output, auto generated
cleaned_books.csv  :----cleaned dataset, auto generated
books.db      :---the final SQLite database



4. Design decisions:-------------
1 Currency conversion, the required fixed rate :--1 GBP = 105.50 INR

This is a project defined constant, not a real market rate. It does not
need an API call, does not need a date, does not need a network. I just
multiply:

 df["price_inr"] = df["price_gbp"] * 105.50

That exact rate is what gets graded, so that is what I used.

2 How I clean each field

price_gbp
    Pull the first number out with regex (\d+(?:\.\d+)?). This ignores
    the pound symbol, the mojibake version of it, commas, and spaces.
    If it fails, I use median imputation. If every row fails, I raise
    an error instead of silently saving garbage.

rating
    Map One through Five to 1 through 5. If it fails, median imputation.

in_stock
    True if the availability text contains the phrase "in stock",
    case insensitive. Otherwise False.

title
    Strip whitespace.If missing or empty, drop the row.

category
    Strip whitespace. If missing or empty, drop the row.

Why median for numbers but drop for title and category?
---------------------------------------------------------

Median keeps the row count and is robust to weird outliers for numbers
like price or ordinal values like rating. But title and category are
identity fields. If I made them up, I would be inventing fake products,
so those rows get dropped instead.

3 The UTF-8 issue:-
-----------------------------------------

When I first scraped the site, the pound symbol came through as the
mojibake version instead of the real character. So the cleaner ignores
the symbol entirely and uses regex to pull out the digits. That way it
works no matter how the currency symbol is encoded. I also set the
encoding to utf-8 in the scraper to be safe.

5. The database schema:-
-------------------------------------------
Two tables linked by a primary and foreign key. Category names are
stored once, and books point at them.

    CREATE TABLE categories (
        category_id   INTEGER PRIMARY KEY AUTOINCREMENT,
        category_name TEXT UNIQUE NOT NULL
    );

    CREATE TABLE books (
        book_id     INTEGER PRIMARY KEY AUTOINCREMENT,
        title       TEXT NOT NULL,
        price_gbp   REAL NOT NULL,
        price_inr   REAL NOT NULL,
        rating      INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
        in_stock    INTEGER NOT NULL,
        category_id INTEGER NOT NULL,
        FOREIGN KEY (category_id) REFERENCES categories(category_id)
    );

29 categories, 100 books, all linked correctly

6. The 6 SQL queries
--------------------
Q1  First 10 books with title, price, rating. Uses SELECT and LIMIT.
Q2  All unique category names. Uses SELECT DISTINCT and ORDER BY.
Q3  In stock books rated 4 or higher. Uses WHERE and ORDER BY.
Q4  Books priced between 20 and 40 pounds. Uses WHERE, BETWEEN, ORDER BY.
Q5  Books in Travel, Mystery, or Poetry. Uses IN, JOIN, ORDER BY.
Q6  Top rated book or books per category. Uses JOIN, a correlated
    subquery, and ORDER BY.

Run python queries.py to see all six SQL strings and their output
printed to the terminal.


7. Proving pandas and SQL agree
-------------------------------
task6_compare.py does the same join two ways. First with pd.read_sql.
Second with books_df.merge(cats_df, on="category_id"). Then it checks
both results are identical using pd.testing.assert_frame_equal.

The output was:

    Shapes: sql=(100, 5), merge=(100, 5)
    EQUIVALENT: pd.read_sql and pd.merge produce identical output.

Same rows, same columns, same values.


8. Acceptance criteria
----------------------
Runs end to end, at least 60 books, at least 3 categories
    scrape.py gives 100 books across 29 categories

price_gbp is float, rating is int 1 to 5, in_stock is bool,
price_inr is float
    done in clean_and_load.py

Fixed rate 1 GBP = 105.50 INR stated
    this README, section 4.1, plus clean_and_load.py

SQLite database with two table PK FK schema plus regeneration script
    books.db plus load_db.py

At least 5 SQL queries covering all clauses plus a JOIN
    queries.py, 6 queries

pd.read_sql and pd.merge shown equal
    task6_compare.py

README with install run steps and decisions
    this file
