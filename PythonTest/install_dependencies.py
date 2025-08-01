#!/usr/bin/env python3
"""
Script d'installation automatique des dépendances pour le projet d'analyse de sentiment
"""

import subprocess
import sys
import os

def install_package(package):
    """Installe un package avec pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installé avec succès")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Erreur lors de l'installation de {package}")
        return False

def main():
    print("🚀 Installation des dépendances pour le projet d'analyse de sentiment...")
    print("=" * 60)
    
    # Liste des packages à installer
    packages = [
        "pandas>=2.0.0",
        "scikit-learn>=1.0.0", 
        "joblib>=1.0.0",
        "nltk>=3.8.0",
        "spacy>=3.8.0",
        "beautifulsoup4>=4.12.0",
        "requests>=2.28.0"
    ]
    
    # Installation des packages
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n📊 Résultat : {success_count}/{len(packages)} packages installés")
    
    # Installation du modèle spaCy français
    print("\n🇫🇷 Installation du modèle français pour spaCy...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "fr_core_news_sm"])
        print("✅ Modèle français installé avec succès")
    except subprocess.CalledProcessError:
        print("❌ Erreur lors de l'installation du modèle français")
    
    # Téléchargement des ressources NLTK
    print("\n📚 Téléchargement des ressources NLTK...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ Ressources NLTK téléchargées")
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement NLTK : {e}")
    
    print("\n🎉 Installation terminée !")
    print("Vous pouvez maintenant exécuter vos scripts Python.")

if __name__ == "__main__":
    main() 