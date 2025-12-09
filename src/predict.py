import joblib
import sys
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ajout du dossier root au path pour pouvoir importer src.preprocess
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocess import clean_text, process_text

def load_artifacts():
    """Charge le modèle et le vectorizer sauvegardés."""
    print("Chargement des artefacts (modèle + vectorizer)...")
    if not os.path.exists('models/model.joblib') or not os.path.exists('models/tfidf_vectorizer.joblib'):
        raise FileNotFoundError("Les fichiers modèles sont introuvables. Lancez 'src/train.py' d'abord.")
        
    model = joblib.load('models/model.joblib')
    vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
    return model, vectorizer

def make_prediction(text, model, vectorizer):
    """Effectue une prédiction sur un texte brut."""
    # 1. Initialisation des outils NLP
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # 2. Prétraitement (Même pipeline que pour l'entraînement !)
    cleaned_txt = clean_text(text)
    processed_txt = process_text(cleaned_txt, stop_words, lemmatizer)
    
    # 3. Vectorisation
    vectorized_txt = vectorizer.transform([processed_txt])
    
    # 4. Prédiction
    prediction = model.predict(vectorized_txt)
    
    # On retourne la classe prédite (c'est un entier 0-19 pour 20newsgroups)
    # Idéalement on aurait la map des noms de classes, mais le dataset sklearn les donne pas direct ici
    return prediction[0]

if __name__ == "__main__":
    # Exemple de test
    sample_text = "The CPU and GPU temperature is too high during gaming."
    
    try:
        model, vectorizer = load_artifacts()
        prediction = make_prediction(sample_text, model, vectorizer)
        
        print("-" * 30)
        print(f"Texte : {sample_text}")
        print(f"Prédiction (Class ID) : {prediction}")
        print("-" * 30)
        
        # Mapping explicite 20 Newsgroups (pour info)
        # 0: alt.atheism, 1: comp.graphics, 2: comp.os.ms-windows.misc, ...
        # (Pour un vrai script, on sauvegarderait target_names dans un json aussi)
        
    except Exception as e:
        print(f"Erreur : {e}")
