# import libs
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

BASE_URL = "http://books.toscrape.com/"
CATALOGUE_URL = "http://books.toscrape.com/catalogue/page-{}.html"

RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def scrape_catalogue_pages(num_pages=5):
    """Scrape first N catalogue listing pages (All products)."""
    books = []
    for page in range(1, num_pages + 1):
        url = CATALOGUE_URL.format(page)
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for article in soup.select("article.product_pod"):
            title = article.h3.a["title"].strip()

            # price like"£51.77"
            price_text = article.select_one("p.price_color").text.strip()

            # rating is a class like star-rating Three
            star_class = article.select_one("p.star-rating")["class"]
            rating_text = [c for c in star_class if c != "star-rating"][0]

            # availability text like in stock
            avail_text = article.select_one("p.instock.availability").text.strip()

            # category not on listing page — go to detail page
            detail_link = article.h3.a["href"]
            detail_url = BASE_URL + "catalogue/" + detail_link.replace("../", "")
            detail_resp = requests.get(detail_url, timeout=10)
            detail_soup = BeautifulSoup(detail_resp.text, "html.parser")

            breadcrumb = detail_soup.select("ul.breadcrumb li a")
            category = breadcrumb[2].text.strip() if len(breadcrumb) >= 3 else "Unknown"

            books.append({
                "title": title,
                "price": price_text,
                "star_rating": rating_text,
                "availability": avail_text,
                "category": category,
            })
            time.sleep(0.05) 

        print(f"Page {page} scraped. Total books so far: {len(books)}")

    return pd.DataFrame(books)

if __name__ == "__main__":
    df = scrape_catalogue_pages(num_pages=5)
    df.to_csv("raw_books.csv", index=False)
    print(f"\n Scraped {len(df)} books across {df['category'].nunique()} categories")
    print(df.head())