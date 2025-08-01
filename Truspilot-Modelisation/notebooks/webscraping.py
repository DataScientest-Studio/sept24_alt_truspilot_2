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
    base_url = "https://www.trustpilot.com/review/ringconn.com?page="
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
                content = review.find_element(By.CSS_SELECTOR, "p").text
            except:
                content = None

            try:
                rating_img = review.find_element(By.CSS_SELECTOR, 'div.star-rating_starRating__sdbkn img')
                alt_text = rating_img.get_attribute('alt')  # Ex: "Rated 4 out of 5 stars"
                rating = int(alt_text.split(" ")[1])
            except:
                rating = None

            try:
                date = review.find_element(By.TAG_NAME, "time").get_attribute("datetime")
            except:
                date = None

            try:
                author = review.find_element(By.CSS_SELECTOR, "span.typography_heading-xxs__QKBS8").text
            except:
                author = None 

            all_reviews.append({
                "Title": title,
                "Content": content,
                "Rating": rating,
                "Date": date,
                "Author": author
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

    df.to_csv("trustpilot_ringconn_v3.csv", index=False, quoting=csv.QUOTE_ALL)
    print("\nExport CSV terminé.")

if __name__ == "__main__":
    main()
