import re
import nltk
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os

"""
Pré-traitement des données (Preprocessing).

Ce script télécharge les "20 Newsgroups", nettoie le texte, et prépare
les jeux d'entraînement, de validation et de test pour le modèle.
"""

# Téléchargement des ressources NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

def clean_text(text):
    """
    Nettoyage basique du texte.
    - Minuscules
    - Suppression ponctuation et chiffres
    """
    # Minuscules pour normaliser
    text = text.lower()
    
    # Suppression ponctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Suppression chiffres
    text = re.sub(r'\d+', '', text)
    
    # Suppression espaces superflus
    text = text.strip()
    return text

def process_text(text, stop_words, lemmatizer):
    """
    Traitement linguistique.
    - Tokenisation
    - Suppression stopwords
    - Lemmatisation
    """
    # Tokenisation
    tokens = word_tokenize(text)
    
    # Garder les mots significatifs et les lemmatiser
    processed_tokens = [
        lemmatizer.lemmatize(word) 
        for word in tokens 
        if word not in stop_words
    ]
    
    return ' '.join(processed_tokens)

def main():
    print("Chargement du dataset 20 Newsgroups...")
    # Chargement sans les métadonnées pour éviter les biais
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    
    df = pd.DataFrame({
        'text': newsgroups.data,
        'target': newsgroups.target
    })
    
    print("Initialisation NLP...")
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    print("Nettoyage du texte en cours...")
    # 1. Nettoyage simple
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # 2. Traitement NLP complet
    df['processed_text'] = df['cleaned_text'].apply(lambda x: process_text(x, stop_words, lemmatizer))
    
    # Suppression des textes vides après nettoyage
    df = df[df['processed_text'].str.len() > 0].copy()
    
    print("Séparation des données...")
    # Split 70% Train, 15% Val, 15% Test
    
    # Séparation train / reste
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['target'])
    
    # Séparation validation / test
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['target'])
    
    os.makedirs('data/processed', exist_ok=True)
    
    print("Sauvegarde CSV...")
    train_df.to_csv('data/processed/train.csv', index=False)
    val_df.to_csv('data/processed/validation.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    
    print(f"Terminé ! Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

if __name__ == "__main__":
    main()
