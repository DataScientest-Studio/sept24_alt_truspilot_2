import subprocess

# Liste des scripts à exécuter dans l'ordre
scripts = [
    "preprocessing_regex.py",
    "preprocessing_advanced.py",
    "feature_engineering.py",
    "train_sentiment.py"
]

for script in scripts:
    print(f"\n🚀 Lancement de {script}...")
    subprocess.run(["python", f"Modelisation/notebooks/{script}"], check=True)
    print(f"✅ {script} terminé avec succès.")

print("\n🎉 Pipeline complet exécuté avec succès !")
