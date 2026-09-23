'''
 SQL queries that cover SELECT/LIMIT, DISTINCT, WHERE, ORDER BY, BETWEEN, IN, and JOIN
'''
import sqlite3
import pandas as pd

DB_PATH = "book.db"


QUERIES = {
    "Q1_select limit":
    """ SELECT title , price_gbp,rating 
    FROM  books  LIMIT 10 ;
    """,

    "Q2_distinct_categories":""" SELECT DISTINCT category_name
        FROM categories
        ORDER BY category_name;
    """,
    "Q3 _where order": """ SELECT title , rating ,in_stock FROM books 
    WHERE  in_stock = 1 AND rating >=4 
    ORDER BY rating DESC , title ASC; """,

    "Q4_between Price ": """ SELECT title ,price_gbp ,price_inr FROM books 
    WHERE price_gbp BETWEEN 20 AND 40  ORDER BY price_gbp ASC""",

    "Q5_in_categories": """
        SELECT b.title,c.category_name,b.rating
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        WHERE c.category_name IN('Travel', 'Mystery', 'Poetry')
        ORDER BY c.category_name, b.rating DESC;
    """,

    "Q6_join_top_rated_per_category": """
        SELECT c.category_name, b.title, b.rating, b.price_gbp, b.price_inr
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        WHERE b.rating = (
            SELECT MAX(b2.rating)
            FROM books b2
            WHERE b2.category_id = b.category_id
        )
        ORDER BY c.category_name, b.title;
    """,}

  
def run_all(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    results = {}
    for name, sql in QUERIES.items():
        print("=" * 70)
        print(f"--- {name} ---")
        print(sql.strip())
        df = pd.read_sql(sql, conn)
        print("\nOUTPUT:")
        print(df.to_string(index=False))
        print()
        results[name] = df
    conn.close()
    return results

if __name__ == "__main__":
    run_all()