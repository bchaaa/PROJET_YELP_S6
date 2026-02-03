# Analyse et Classification d'Avis Clients (Projet Yelp)

Ce projet de SAE S6 combine une analyse statistique rigoureuse des données Yelp et une approche comparative d'Intelligence Artificielle (Machine Learning vs Deep Learning vs GenAI).

## 📂 Architecture du Projet

### 1. Analyse Exploratoire (Approche Statistique)
Cette phase valide les hypothèses métiers avant la modélisation.
- **Notebook 1a (Business)** : Test de Mann-Whitney montrant l'impact de la popularité sur la note.
- **Notebook 1b (Reviewers)** : Test de Student sur le biais de sévérité des experts "Elite".
- **Notebook 1c (Texte)** : Analyse TF-IDF des mots-clés par niveau de satisfaction.

### 2. Intelligence Artificielle (Approche Comparative)
- **Notebook 2 (ML Classique)** : Régression Logistique.
  - *Performance* : ~94% Accuracy.
  - *Point fort* : Gestion du déséquilibre des classes (Recall > 90% sur les avis négatifs).
- **Notebook 3 (Deep Learning)** : Réseau de Neurones (MLP).
  - *Performance* : Convergence optimisée.
  - *Technique* : Utilisation de l'Early Stopping pour éviter le sur-apprentissage.
- **Notebook 4 (GenAI / LLM)** : Modèle BERT (Zero-Shot).
  - *Innovation* : Capable de classer les plaintes (Service vs Cuisine) sans aucun entraînement préalable.

## 🛠️ Structure Technique

```text
PROJET_YELP_S6/
├── data/               # Données brutes (exclues du git)
├── models/             # Modèles entraînés (.pkl)
│   ├── classic_model.pkl
│   ├── deep_model.pkl
│   └── vectorizer.pkl
├── notebooks/          # Les 5 notebooks d'analyse
│   ├── 0_Setup_Data.ipynb
│   ├── 1a_Exploration_Business.ipynb
│   ├── 1b_Exploration_Reviewers.ipynb
│   ├── 1c_Exploration_Text.ipynb
│   ├── 2_Classic_ML.ipynb
│   ├── 3_Deep_Learning.ipynb
│   └── 4_GenAI_Experiments.ipynb
├── src/                # Code source modulaire
│   ├── data_loader.py  # Chargement optimisé des JSON
│   ├── preprocessing.py
│   └── models.py
└── requirements.txt    # Dépendances Python