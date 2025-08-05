import sys
import os

# Vérification de l'environnement virtuel (optionnel)
# Décommentez les lignes suivantes si vous voulez forcer l'utilisation de l'environnement virtuel
# if sys.prefix == sys.base_prefix:
#     print("\033[91m[ERREUR] Veuillez activer l'environnement virtuel avant de lancer ce script :\033[0m")
#     print("  source venv/bin/activate && python test/webscraping.py")
#     sys.exit(1)

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
import csv

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def extract_reviews(driver):
    base_url = "https://www.trustpilot.com/review/ouraring.com?page="
    all_reviews = []
    page = 1

    while True:
        print(f"Scraping page {page}...")
        driver.get(base_url + str(page))
        time.sleep(3)

        review_blocks = driver.find_elements(By.TAG_NAME, "article")
        if not review_blocks:
            print("Plus d'avis détectés. Fin du scraping.")
            break

        for review in review_blocks:
            try:
                title = review.find_element(By.TAG_NAME, "h2").text
            except:
                title = None

            try:
                # Essayer plusieurs sélecteurs pour capturer le contenu complet
                content_selectors = [
                    "p[data-service-review-card-typography='true']",
                    "p[data-service-review-card-typography]",
                    "p[class*='review']",
                    "p"
                ]
                
                content = None
                for selector in content_selectors:
                    try:
                        content_element = review.find_element(By.CSS_SELECTOR, selector)
                        # Essayer textContent d'abord pour récupérer le texte original
                        content = content_element.get_attribute('textContent')
                        if not content:
                            content = content_element.text
                        if content:
                            break
                    except:
                        continue
                
                # Nettoyer le contenu si trouvé
                if content:
                    content = content.strip()
                    # Supprimer les "See more" et autres textes de troncature
                    if "... See more" in content:
                        content = content.replace("... See more", "")
                    if "See more" in content:
                        content = content.replace("See more", "")
            except:
                content = None

            try:
                # Nouveau sélecteur plus robuste pour la note
                rating_img = review.find_element(By.CSS_SELECTOR, 'img[alt*="Rated"]')
                alt_text = rating_img.get_attribute('alt')  # Ex: "Rated 5 out of 5 stars"
                rating = int(alt_text.split(" ")[1])
            except:
                rating = None

            try:
                date = review.find_element(By.TAG_NAME, "time").get_attribute("datetime")
            except:
                date = None

            try:
                # Sélecteur basé sur l'attribut data-consumer-name-typography
                author = review.find_element(By.CSS_SELECTOR, 'span[data-consumer-name-typography="true"]').text
            except:
                author = None

            try:
                # Pays de l'utilisateur
                country = review.find_element(By.CSS_SELECTOR, 'span[data-consumer-country-typography="true"]').text
            except:
                country = None

            try:
                # Nombre de commentaires déjà laissés
                reviews_count_span = review.find_element(By.CSS_SELECTOR, 'span[data-consumer-reviews-count-typography="true"]').text
                # Extraction du nombre (ex: "3 reviews" -> 3)
                reviews_count = int(reviews_count_span.split()[0])
            except:
                reviews_count = None

            all_reviews.append({
                "Title": title,
                "Content": content,
                "Rating": rating,
                "Date": date,
                "Author": author,
                "Country": country,
                "ReviewsCount": reviews_count
            })

        page += 1
        time.sleep(2)

    return all_reviews

def main():
    driver = setup_driver()
    reviews = extract_reviews(driver)
    driver.quit()

    df = pd.DataFrame(reviews)
    print(f"\nTotal d'avis récupérés : {len(df)}\n")
    print(df.head())

    df.to_csv("trustpilot_ouraring.csv", index=False, quoting=csv.QUOTE_ALL)
    print("\nExport CSV terminé.")

if __name__ == "__main__":
    main()
