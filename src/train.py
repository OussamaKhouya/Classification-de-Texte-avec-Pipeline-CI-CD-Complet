import pandas as pd
import joblib
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score, confusion_matrix

def main():
    # Configuration MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Text_Classification_Projet9")

    print("Chargement des données nettoyées...")
    try:
        train_df = pd.read_csv('data/processed/train.csv')
        test_df = pd.read_csv('data/processed/test.csv')
    except FileNotFoundError:
        print("Erreur: Fichiers introuvables. Lancez 'src/preprocess.py' d'abord.")
        return

    train_df = train_df.dropna(subset=['processed_text'])
    test_df = test_df.dropna(subset=['processed_text'])

    # Paramètres du modèle
    MAX_FEATURES = 10000
    N_ESTIMATORS = 200
    
    # Démarrage du run MLflow
    with mlflow.start_run() as run:
        print(f"MLflow Run ID: {run.info.run_id}")
        
        # Log des hyperparamètres
        mlflow.log_param("max_features", MAX_FEATURES)
        mlflow.log_param("n_estimators", N_ESTIMATORS)

        print("Vectorisation (TF-IDF)...")
        # TF-IDF : on apprend uniquement sur le train
        vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
        X_train = vectorizer.fit_transform(train_df['processed_text'])
        X_test = vectorizer.transform(test_df['processed_text'])
        
        y_train = train_df['target']
        y_test = test_df['target']

        print("Entraînement (RandomForest)...")
        model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        print("Évaluation...")
        y_pred = model.predict(X_test)
        
        # Calcul des métriques
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (Weighted): {precision:.4f}")
        print(f"Recall (Weighted): {recall:.4f}")
        print(f"F1-Score (Weighted): {f1:.4f}")
        
        # Log des métriques dans MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Rapport détaillé
        print("\nRapport détaillé :")
        report = classification_report(y_test, y_pred)
        print(report)
        
        # Sauvegarde des résultats locaux (Pour CML)
        os.makedirs('models', exist_ok=True)
        os.makedirs('reports', exist_ok=True)

        # 1. Sauvegarde des métriques en JSON
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        with open("reports/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
            
        # 2. Sauvegarde du rapport texte
        with open("reports/classification_report.txt", "w") as f:
            f.write(report)
            
        # 3. Génération et sauvegarde de la matrice de confusion
        print("Génération de la matrice de confusion...")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
        plt.title('Matrice de Confusion')
        plt.ylabel('Vrai label')
        plt.xlabel('Label prédit')
        plt.savefig('reports/confusion_matrix.png')
        plt.close()
        
        # Log de la matrice de confusion dans MLflow
        mlflow.log_artifact('reports/confusion_matrix.png')

        # 4. Sauvegarde modèle et vectorizer (Local + MLflow)
        joblib.dump(model, 'models/model.joblib')
        joblib.dump(vectorizer, 'models/tfidf_vectorizer.joblib')
        
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact('models/tfidf_vectorizer.joblib')
        
        print("Terminé ! Modèle, métriques, rapport et matrice de confusion sauvegardés (Local + MLflow).")

if __name__ == "__main__":
    main()
