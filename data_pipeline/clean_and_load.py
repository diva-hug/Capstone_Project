import numpy as np
import pandas as pd 
import re 

RATING_MAP ={
    "One": 1,
    "Two":2,
    "Three":3,
    "Four":4,
    "Five":5
}
# Fixed project base line
GBP_TO_INR =105.50


def parse_price_to_float(s):
    """Extract the first number from any string"""
    if pd.isna(s):
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", str(s))
    return float(m.group(1)) if m else None

def clean_books(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    #Strip the currency symbol from  price_gbp.
    df["price_gbp"]=df["price"].apply(parse_price_to_float)

    #Convert the text star rating to int rating 1–5
    df["rating"]=df["star_rating"].map(RATING_MAP)

    #Parse the  availability text into a boolean column INSTOCK
    df["in_stock"]=(
        df["availability"].astype(str)
                    .str.strip()
                    .str.lower()
                    .str.contains("in stock"))
    
    # If any field fails to parse for a given row
    n_missing_price = df["price_gbp"].isna().sum()
    if 0< n_missing_price <len(df):
        median_price =df["price_gbp"].median()
        df['price_gbp']=df['price_gbp'].fillna(median_price)
        print(f"Imputed {n_missing_price} missing price with medain {median_price}")
    elif n_missing_price == len(df):
        raise ValueError("All price failed to prase - check the row books columns")
    


    if df["rating"].isna().any():
        median_rating=df['rating'].median()
        df["rating"]=df['rating'].fillna(median_rating)
        print(f"imputed missing rating with median{median_rating}")

    #Convert price_gbp to a price_inr
    df["price_inr"] =df["price_gbp"]*GBP_TO_INR


    # This fixed-rate conversion is the required

    df["rating"]=df["rating"].astype(int)
    df["in_stock"]=df["in_stock"].astype(bool)
    df['price_gbp']=df["price_gbp"].astype(float)
    df["price_inr"]=df["price_inr"].astype(float)

    # keep only need columns

    df=df[["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]]
    return df.reset_index(drop=True)

if __name__=="__main__":
    raw =pd.read_csv("raw_books.csv")
    cleaned = clean_books(raw)
    cleaned.to_csv("cleaned_books.csv", index=False)
    print(f"\n Cleaned {len(cleaned)} rows")
    print(cleaned.dtypes)
    print(cleaned.head())