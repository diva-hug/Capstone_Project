import sqlite3
import pandas as pd

# path of databse 

DB_PATH ="book.db"

def load_to_sqlite(df: pd.DataFrame, db_path: str = DB_PATH):
    conn =sqlite3.connect(db_path)
    cur =conn.cursor()


    cur.executescript(""" 
    DROP TABLE IF EXISTS books;
    DROP TABLE IF EXISTS categories;

    CREATE TABLE categories(
    category_id INTEGER PRIMARY KEY,
    category_name TEXT UNIQUE );
    

    CREATE TABLE books (
    book_id INTEGER PRIMARY KEY  AUTOINCREMENT,
    title TEXT NOT NULL,
    price_gbp REAL NOT NULL,
    price_inr REAL  NOT NULL,
    rating INTEGER  NOT NULL CHECK (rating BETWEEN 1 AND 5),
    in_stock INTEGER NOT NULL,
    category_id INTEGER  NOT NULL ,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
    )

    """)


    # INSERT THE CATEGORIES DATA

    for  cat in sorted(df["category"].unique()):
        cur.execute("INSERT OR IGNORE INTO categories(category_name) VALUES(?)",(cat,))


    # build the name id map 
    cur.execute("SELECT category_id, category_name FROM categories")
    cat_map = {name: cid for cid, name in cur.fetchall()}

    # insert the books here row by row

    for _,row in df.iterrows():
        cur.execute(
            """ INSERT INTO  books 
            (title, price_gbp, price_inr, rating, in_stock, category_id) VALUES(?,?,?,?,?,?)""",
            (
                row["title"],
                float(row["price_gbp"]),
                float(row["price_inr"]),
                int(row["rating"]),
                int(bool(row["in_stock"])),
                cat_map[row["category"]],
            ),
        )
    conn.commit()

    #  check
    n_cats = cur.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
    n_books = cur.execute("SELECT COUNT(*) FROM books").fetchone()[0]
    conn.close()
    print(f"Loaded {n_books} books into {n_cats} categories → {db_path}")


if __name__ == "__main__":
    cleaned = pd.read_csv("cleaned_books.csv")
    load_to_sqlite(cleaned)