#!/usr/bin/env python3
"""
Script d'analyse des résultats de prédiction
Calcule le pourcentage de réussite et les métriques de performance
"""

import pandas as pd
import numpy as np

def analyser_resultats():
    """Analyse les résultats de prédiction"""
    
    # Lecture du fichier de résultats
    try:
        df = pd.read_csv("Baseline(5)/resultats_test_predictions.csv", sep=';', encoding='utf-8-sig')
        print("📊 Analyse des résultats de prédiction")
        print("=" * 50)
        
        # Statistiques générales
        total_predictions = len(df)
        predictions_correctes = df['Correct'].sum()
        pourcentage_reussite = (predictions_correctes / total_predictions) * 100
        
        print(f"📈 Statistiques générales:")
        print(f"   • Total des prédictions: {total_predictions}")
        print(f"   • Prédictions correctes: {predictions_correctes}")
        print(f"   • Prédictions incorrectes: {total_predictions - predictions_correctes}")
        print(f"   • Pourcentage de réussite: {pourcentage_reussite:.2f}%")
        
        # Analyse par sentiment
        print(f"\n🎯 Analyse par sentiment:")
        
        # Sentiment positif (1)
        positif_vrai = df[df['Vrai'] == 1]
        positif_correct = positif_vrai[positif_vrai['Correct'] == True]
        precision_positif = len(positif_correct) / len(positif_vrai) * 100 if len(positif_vrai) > 0 else 0
        
        print(f"   • Sentiment positif (1):")
        print(f"     - Total réel: {len(positif_vrai)}")
        print(f"     - Prédits correctement: {len(positif_correct)}")
        print(f"     - Précision: {precision_positif:.2f}%")
        
        # Sentiment négatif (0)
        negatif_vrai = df[df['Vrai'] == 0]
        negatif_correct = negatif_vrai[negatif_vrai['Correct'] == True]
        precision_negatif = len(negatif_correct) / len(negatif_vrai) * 100 if len(negatif_vrai) > 0 else 0
        
        print(f"   • Sentiment négatif (0):")
        print(f"     - Total réel: {len(negatif_vrai)}")
        print(f"     - Prédits correctement: {len(negatif_correct)}")
        print(f"     - Précision: {precision_negatif:.2f}%")
        
        # Analyse des erreurs
        print(f"\n❌ Analyse des erreurs:")
        erreurs = df[df['Correct'] == False]
        
        if len(erreurs) > 0:
            print(f"   • Erreurs totales: {len(erreurs)}")
            
            # Erreurs par type
            faux_positifs = erreurs[(erreurs['Vrai'] == 0) & (erreurs['Prévu'] == 1)]
            faux_negatifs = erreurs[(erreurs['Vrai'] == 1) & (erreurs['Prévu'] == 0)]
            
            print(f"   • Faux positifs (prédit positif, réel négatif): {len(faux_positifs)}")
            print(f"   • Faux négatifs (prédit négatif, réel positif): {len(faux_negatifs)}")
            
            # Exemples d'erreurs
            print(f"\n📝 Exemples d'erreurs:")
            for i, (_, row) in enumerate(erreurs.head(3).iterrows()):
                print(f"   {i+1}. Texte: {row['Texte'][:100]}...")
                print(f"      Vrai: {row['Vrai']}, Prédit: {row['Prévu']}")
                print()
        
        # Interprétation
        print(f"\n💡 Interprétation:")
        if pourcentage_reussite >= 90:
            print(f"   🎉 Excellent! Le modèle a une performance très élevée ({pourcentage_reussite:.2f}%)")
        elif pourcentage_reussite >= 80:
            print(f"   👍 Bon! Le modèle a une bonne performance ({pourcentage_reussite:.2f}%)")
        elif pourcentage_reussite >= 70:
            print(f"   ⚠️  Moyen. Le modèle a une performance acceptable ({pourcentage_reussite:.2f}%)")
        else:
            print(f"   ❌ Faible. Le modèle nécessite des améliorations ({pourcentage_reussite:.2f}%)")
        
        # Recommandations
        print(f"\n🔧 Recommandations:")
        if precision_positif < 80 or precision_negatif < 80:
            print(f"   • Améliorer la précision par classe (actuellement {min(precision_positif, precision_negatif):.2f}%)")
        if len(erreurs) > 0:
            print(f"   • Analyser les cas d'erreur pour améliorer le modèle")
        print(f"   • Augmenter la taille du dataset d'entraînement")
        print(f"   • Essayer d'autres techniques de vectorisation (TF-IDF vs CountVectorizer)")
        
        return {
            'total': total_predictions,
            'correctes': predictions_correctes,
            'pourcentage': pourcentage_reussite,
            'precision_positif': precision_positif,
            'precision_negatif': precision_negatif,
            'erreurs': len(erreurs)
        }
        
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse: {e}")
        return None

if __name__ == "__main__":
    analyser_resultats() 