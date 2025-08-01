import requests
from bs4 import BeautifulSoup
import pandas as pd
from time import sleep
import csv

def extract_reviews(page_url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(page_url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    review_cards = soup.find_all('article')
    reviews_data = []

    for card in review_cards:
        try:
            review_text = card.find('p').get_text(strip=True)
            review_date = card.find('time').get_text(strip=True)
            rating_tag = card.find('div', attrs={"data-service-review-rating": True})
            rating = rating_tag.get("data-service-review-rating") if rating_tag else None
            author_tag = card.select_one("span[data-consumer-name-typography]")
            author = author_tag.get_text(strip=True) if author_tag else "Anonyme"


            reviews_data.append({
                "Auteur": author,
                "Date": review_date,
                "Texte": review_text,
                "Note": rating
            })

        except Exception:
            continue

    return reviews_data

def extract_all_reviews(base_url, from_page=1, to_page=20):
    all_reviews = []
    for page in range(from_page, to_page + 1):
        page_url = f"{base_url}?page={page}"
        print(f"Scraping : {page_url}")
        all_reviews.extend(extract_reviews(page_url))
        sleep(1)
    return pd.DataFrame(all_reviews)

def clean_and_export(df, output_csv="avis_trustpilot_propre.csv"):
    # Nettoyage
    df = df.dropna(subset=["Auteur", "Texte", "Date", "Note"])
    df = df[df["Texte"].str.strip() != ""]
    df = df[df["Note"].apply(lambda x: str(x).isdigit())]
    df["Note"] = df["Note"].astype(int)

    # Export CSV propre : gérer les guillemets pour texte multi-ligne
    df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
        sep=";",
        quoting=csv.QUOTE_ALL  # pour que Excel/LibreOffice respecte les colonnes
    )
    print(f"✅ Fichier généré : {output_csv}")

# ---- MAIN ----
if __name__ == "__main__":
    base_url = "https://fr.trustpilot.com/review/circular.xyz"
    df_reviews = extract_all_reviews(base_url, from_page=1, to_page=100)
    clean_and_export(df_reviews)
