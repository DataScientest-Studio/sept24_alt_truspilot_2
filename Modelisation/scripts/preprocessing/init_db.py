import sqlite3
import pandas as pd
from pathlib import Path

CSV_PATH = "data/processed/trustpilot_dataset_final_cleaned.csv"
DB_PATH = "data/trustpilot.db"
TABLE_NAME = "reviews"

def init_database():
    if not Path(CSV_PATH).exists():
        raise FileNotFoundError(f"CSV introuvable : {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    conn = sqlite3.connect(DB_PATH)
    df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
    conn.close()

    print("Base SQLite créée avec succès.")
    print(f"Fichier DB : {DB_PATH}")
    print(f"Table : {TABLE_NAME}")
    print(f"Nombre de lignes importées : {len(df)}")
    print("Colonnes importées :")
    print(list(df.columns))

if __name__ == "__main__":
    init_database()