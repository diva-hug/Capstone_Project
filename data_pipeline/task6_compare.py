# pd.read sql  and pd.merge 

import sqlite3
import pandas as pd 


DB_PATH = "book.db"

conn =sqlite3.connect(DB_PATH)

JOIN_SQL =""" SELECT c.category_name, b.title, b.rating, b.price_gbp, b.price_inr
FROM books b 
JOIN categories c  on b.category_id =c.category_id 
ORDER  BY  c.category_name, b.title;"""

# sql join 

sql_result =pd.read_sql(JOIN_SQL,conn)

# pandas merge 

books_df =pd.read_sql("SELECT * FROM books",conn)
cats_df =pd.read_sql("SELECT * FROM categories",conn)


merged = books_df.merge(cats_df,on="category_id",how="inner")
merged= merged[["category_name", "title", "rating", "price_gbp", "price_inr"]]
merged=merged.sort_values(["category_name", "title"]).reset_index(drop=True)

sql_result = sql_result.reset_index(drop=True)
conn.close()

print("="*70)
print("SQL JOIN RESULT (first 10):")
print(sql_result.head(10).to_string(index=False))


print("\n" + "=" * 70)
print("PD.MERGE RESULT (first 10):")
print(merged.head(10).to_string(index=False))

print("\n" + "=" * 70)
print(f"Shapes: sql={sql_result.shape}, merge={merged.shape}")


try:
    pd.testing.assert_frame_equal(sql_result, merged, check_dtype=False)
    print("EQUIVALENT: pd.read_sql and pd.merge produce identical output.")
except AssertionError as e:
    print("NOT equivalent:")
    print(e)