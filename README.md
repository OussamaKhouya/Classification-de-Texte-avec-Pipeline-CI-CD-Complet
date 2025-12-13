> Projet de classification de texte avec une chaîne MLOps complète (prétraitement, entraînement, API FastAPI, Docker, CI/CD GitHub Actions, reporting CML).

# 1. Aperçu du projet

Ce dépôt implémente une pipeline de classification de texte sur le jeu de données **20 Newsgroups**, avec :

- Prétraitement des données (`src/preprocess.py`) : nettoyage du texte, tokenisation, stopwords, lemmatisation, splits train/val/test.
- Entraînement et évaluation (`src/train.py`) : TF‑IDF + RandomForest, métriques, matrice de confusion, logging MLflow.
- Service d’inférence (`src/app.py`) : API REST FastAPI exposant `/health` et `/predict`.
- Tests automatisés (`tests/`) via `pytest`.
- Containerisation (`Dockerfile`) pour déployer l’API dans un conteneur.
- CI/CD GitHub Actions (`.github/workflows/`) incluant build, tests, build/push Docker, reporting CML et déploiement staging → production.

Visuels principaux (générés par le projet) :

- Diagramme du pipeline de données  
  `![Pipeline de données](.technical-report/snapshots/pipeline.png)`
- Diagramme du pipeline d’entraînement / évaluation  
  `![Pipeline entraînement](.technical-report/snapshots/train_eval.png)`
- Matrice de confusion du modèle  
  `![Matrice de confusion](reports/confusion_matrix.png)`

# 2. Prérequis

- **Python** 3.9 (ou compatible avec les dépendances du projet).
- **pip** (gestionnaire de paquets Python).
- Optionnel mais recommandé : `virtualenv` ou `venv`.
- Pour la partie Docker (optionnelle) : **Docker** installé et fonctionnel.

# 3. Installation locale

Dans un terminal, à la racine du projet :

```bash
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Les ressources NLTK nécessaires (stopwords, wordnet, etc.) seront téléchargées automatiquement lors du premier lancement des scripts.

# 4. Préparation des données (prétraitement)

Lance le pipeline de données :

```bash
python src/preprocess.py
```

Ce script :

- télécharge le dataset **20 Newsgroups**,
- construit un `DataFrame` avec les colonnes `text` (brut) et `target`,
- applique le nettoyage (`clean_text`) : minuscules, suppression de la ponctuation et des chiffres,
- applique le traitement linguistique (`process_text`) : tokenisation, retrait des stopwords anglais, lemmatisation,
- filtre les lignes dont `processed_text` est vide,
- découpe en train / validation / test (70 % / 15 % / 15 %, stratifié),
- enregistre :
  - `data/processed/train.csv`
  - `data/processed/validation.csv`
  - `data/processed/test.csv`

Si besoin, tu peux vérifier rapidement la présence des fichiers :

```bash
ls data/processed
```

# 5. Entraînement et évaluation du modèle

Une fois les données prétraitées, entraîne le modèle :

```bash
python src/train.py
```

Ce script :

- charge `train.csv` et `test.csv`,
- supprime les lignes avec `processed_text` manquant,
- vectorise les textes avec **TF‑IDF** (max 5000 features),
- entraîne un **RandomForestClassifier** (100 arbres),
- prédit sur le jeu de test,
- calcule les métriques : accuracy, précision, rappel, F1 pondérés,
- génère :
  - `reports/metrics.json`
  - `reports/classification_report.txt`
  - `reports/confusion_matrix.png`
- sauvegarde les artefacts du modèle :
  - `models/model.joblib`
  - `models/tfidf_vectorizer.joblib`
- logge l’expérience dans **MLflow** (`mlruns/`).

Tu peux ouvrir les fichiers de rapport pour voir le détail :

```bash
cat reports/metrics.json
cat reports/classification_report.txt
```

Et visualiser la matrice de confusion (par exemple dans un explorateur de fichiers ou un viewer d’images) :

- `reports/confusion_matrix.png`

## 5.1. Visualisation des expériences avec MLflow

Pour explorer les runs MLflow via une interface web locale :

```bash
mlflow ui --backend-store-uri file:./mlruns
```

Puis ouvre dans ton navigateur : <http://127.0.0.1:5000>.

# 6. Lancer l’API FastAPI (local)

Assure-toi d’abord que les artefacts du modèle existent (`models/model.joblib`, `models/tfidf_vectorizer.joblib`). Ensuite, lance l’API :

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

### 6.1. Vérifier l’état du service

Health check :

```bash
curl http://localhost:8000/health
```

Réponse attendue (exemple) :

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### 6.2. Faire une requête de prédiction

Exemple en `curl` :

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Computer graphics and GPU acceleration"}'
```

Réponse typique :

```json
{
  "text": "Computer graphics and GPU acceleration",
  "prediction_class_id": 3,
  "status": "success"
}
```

Tu peux aussi utiliser l’interface interactive générée par FastAPI :

- Documentation OpenAPI : <http://localhost:8000/docs>
- Documentation alternative : <http://localhost:8000/redoc>

# 7. Utilisation avec Docker

## 7.1. Construire l’image Docker

Après avoir entraîné le modèle et généré les artefacts :

```bash
docker build -t text-classifier .
```

## 7.2. Lancer le conteneur

```bash
docker run -p 8000:8000 text-classifier
```

Puis utilise les mêmes URLs que pour l’API locale :

- `http://localhost:8000/health`
- `http://localhost:8000/predict`

# 8. Tests automatisés

Le projet contient des tests pour :

- le prétraitement (`tests/test_preprocess.py`),
- la présence / validité des artefacts (`tests/test_train.py`),
- l’API FastAPI (`tests/test_api.py`).

Pour exécuter l’ensemble des tests :

```bash
pytest tests -v
```

# 9. CI/CD sur GitHub Actions

Trois workflows principaux sont définis dans `.github/workflows/` :

- `docker.yaml` — **Docker Build & Push**  
  - déclenché sur `push` et `pull_request` vers `master`,  
  - installe les dépendances, prétraite les données, entraîne le modèle, lance les tests,  
  - construit et (hors PR) pousse l’image Docker vers GitHub Container Registry.

- `cml.yaml` — **CML Report**  
  - réexécute prétraitement et entraînement,  
  - génère un `report.md` avec les métriques, le rapport de classification et la matrice de confusion,  
  - publie ce rapport en commentaire sur la pull request via **CML**.

- `deploy.yaml` — **Deploy Pipeline (Staging → Production)**  
  - se déclenche quand le workflow Docker build/push est terminé sur `master`,  
  - déploie l’image en staging, exécute des tests d’intégration sur `/health` et `/predict`,  
  - si la validation réussit, simule un déploiement en production et prévoit un scénario de rollback automatique en cas d’échec.

# 10. Démonstration rapide (check‑list)

Pour démontrer le projet à un évaluateur externe, tu peux suivre ce scénario :

1. **Cloner le dépôt et installer les dépendances** (section 3).  
2. **Prétraiter les données** (`python src/preprocess.py`) et montrer les CSV générés.  
3. **Entraîner le modèle** (`python src/train.py`) et ouvrir :
   - `reports/metrics.json`,
   - `reports/confusion_matrix.png`.
4. **Lancer MLflow UI** (`mlflow ui`) et montrer les runs, hyperparamètres et métriques.  
5. **Démarrer l’API FastAPI** (`uvicorn src.app:app ...`) et tester :
   - `/health` dans le navigateur,
   - `/docs` pour l’interface interactive,
   - une requête `/predict` (via `curl` ou l’UI).  
6. (Optionnel) **Montrer le conteneur Docker** :
   - `docker build -t text-classifier .`
   - `docker run -p 8000:8000 text-classifier`
7. Sur GitHub, montrer les workflows Actions (`docker.yaml`, `cml.yaml`, `deploy.yaml`) et, si possible, une exécution réussie (build + tests + rapport CML).

Ce guide donne le parcours complet pour un utilisateur externe : installation, exécution du pipeline, lancement du service, tests, et aperçu de la CI/CD. 
