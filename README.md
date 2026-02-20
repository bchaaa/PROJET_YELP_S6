# Analyse et Classification d'Avis Clients (Projet Yelp SAE S6)

Ce projet de SAE S6 combine une analyse statistique rigoureuse des données Yelp et une approche comparative d'Intelligence Artificielle (Machine Learning, Deep Learning et IA Générative Agentique).

## 📂 Architecture du Projet

### 1. Analyse Exploratoire (Approche Statistique)
Cette phase valide les hypothèses métiers avant la modélisation.
- **Notebook 1a (Business)** : Test de Mann-Whitney montrant l'impact de la popularité sur la note.
- **Notebook 1b (Reviewers)** : Test de Student sur le biais de sévérité des experts "Elite".
- **Notebook 1c (Texte)** : Analyse TF-IDF des mots-clés par niveau de satisfaction.

### 2. Intelligence Artificielle (Approche Comparative)
- **Notebook 2 (ML Classique)** : Régression Logistique, Random Forest, Naive Bayes.
  - *Performance* : ~94% Accuracy.
  - *Point fort* : Gestion du déséquilibre des classes (Recall > 90% sur les avis négatifs).
- **Notebook 3 (Deep Learning)** : Réseau de Neurones (MLP).
  - *Performance* : Convergence optimisée.
  - *Technique* : Utilisation de l'Early Stopping pour éviter le sur-apprentissage.
- **Notebook 4 (GenAI / LLM)** : Modèle d'encodage (Zero-Shot & Few-Shot).
  - *Innovation* : Capable de classer les plaintes sans aucun entraînement préalable.
- **Notebook 06 (IA Agentique)** : Extraction d'aspects et analyse de sentiments (ABSA) via LangChain et LLM local.
  - *Technique* : Prompts structurés via Pydantic pour forcer une sortie JSON exploitable.
- **Notebook 5 (Conclusion & Synthèse)** : Recommandations métiers et choix techniques pour la mise en production.

## 🛠️ Structure Technique

```text
PROJET_YELP_S6/
├── data/               # Données brutes (exclues du git)
├── models/             # Modèles entraînés (.pkl)
│   ├── classic_model.pkl
│   ├── deep_model.pkl
│   └── vectorizer.pkl
├── notebooks/          # Les 7 notebooks du projet
│   ├── 0_Setup_Data.ipynb
│   ├── 1a_Exploration_Business.ipynb
│   ├── 1b_Exploration_Reviewers.ipynb
│   ├── 1c_Exploration_Text.ipynb
│   ├── 2_Classic_ML.ipynb
│   ├── 3_Deep_Learning.ipynb
│   ├── 4_GenAI_Experiments.ipynb
│   ├── 06_genai_agentic.ipynb      <-- NOUVEAU
│   └── 5_Conclusion_Synthese.ipynb <-- NOUVEAU
├── src/                # Code source modulaire
│   ├── data_loader.py  # Chargement optimisé des JSON
│   ├── preprocessing.py
│   └── models.py
└── requirements.txt    # Dépendances Python

Gemini a dit
Je comprends tout à fait ce que tu souhaites faire ! Cependant, en tant qu'intelligence artificielle fonctionnant via le navigateur, je n'ai pas d'accès direct aux fichiers de ton ordinateur. Je ne peux donc pas ouvrir ton fichier README.md local pour le modifier et le sauvegarder à ta place.

La seule façon pour moi de te le transmettre est de te générer le texte complet ici. Il te suffit de copier le bloc de code ci-dessous et de le coller intégralement à la place du contenu de ton fichier README.md actuel.

Voici le contenu exact et complet de ton fichier README.md mis à jour, prêt à être copié-collé :

Markdown
# Analyse et Classification d'Avis Clients (Projet Yelp SAE S6)

Ce projet de SAE S6 combine une analyse statistique rigoureuse des données Yelp et une approche comparative d'Intelligence Artificielle (Machine Learning, Deep Learning et IA Générative Agentique).

## 📂 Architecture du Projet

### 1. Analyse Exploratoire (Approche Statistique)
Cette phase valide les hypothèses métiers avant la modélisation.
- **Notebook 1a (Business)** : Test de Mann-Whitney montrant l'impact de la popularité sur la note.
- **Notebook 1b (Reviewers)** : Test de Student sur le biais de sévérité des experts "Elite".
- **Notebook 1c (Texte)** : Analyse TF-IDF des mots-clés par niveau de satisfaction.

### 2. Intelligence Artificielle (Approche Comparative)
- **Notebook 2 (ML Classique)** : Régression Logistique, Random Forest, Naive Bayes.
  - *Performance* : ~94% Accuracy.
  - *Point fort* : Gestion du déséquilibre des classes (Recall > 90% sur les avis négatifs).
- **Notebook 3 (Deep Learning)** : Réseau de Neurones (MLP).
  - *Performance* : Convergence optimisée.
  - *Technique* : Utilisation de l'Early Stopping pour éviter le sur-apprentissage.
- **Notebook 4 (GenAI / LLM)** : Modèle d'encodage (Zero-Shot & Few-Shot).
  - *Innovation* : Capable de classer les plaintes sans aucun entraînement préalable.
- **Notebook 06 (IA Agentique)** : Extraction d'aspects et analyse de sentiments (ABSA) via LangChain et LLM local.
  - *Technique* : Prompts structurés via Pydantic pour forcer une sortie JSON exploitable.
- **Notebook 5 (Conclusion & Synthèse)** : Recommandations métiers et choix techniques pour la mise en production.

## 🛠️ Structure Technique

```text
PROJET_YELP_S6/
├── data/               # Données brutes (exclues du git)
├── models/             # Modèles entraînés (.pkl)
│   ├── classic_model.pkl
│   ├── deep_model.pkl
│   └── vectorizer.pkl
├── notebooks/          # Les 7 notebooks du projet
│   ├── 0_Setup_Data.ipynb
│   ├── 1a_Exploration_Business.ipynb
│   ├── 1b_Exploration_Reviewers.ipynb
│   ├── 1c_Exploration_Text.ipynb
│   ├── 2_Classic_ML.ipynb
│   ├── 3_Deep_Learning.ipynb
│   ├── 4_GenAI_Experiments.ipynb
│   ├── 06_genai_agentic.ipynb
│   └── 5_Conclusion_Synthese.ipynb
├── src/                # Code source modulaire
│   ├── data_loader.py  # Chargement optimisé des JSON
│   ├── preprocessing.py
│   └── models.py
└── requirements.txt    # Dépendances Python


🚀 Prérequis et Installation
Pour exécuter les modèles classiques et préparer l'environnement, installez les dépendances à la racine du projet :

Bash
pip install -r requirements.txt
🧠 Tutoriel : Utilisation du Notebook 06 (IA Agentique avec Ollama)

Le notebook 06_genai_agentic.ipynb utilise Ollama pour faire tourner un Modèle de Langage (LLM) localement, garantissant la confidentialité des données (aucune donnée n'est envoyée sur le cloud).

Pour que ce notebook s'exécute correctement, vous devez installer et lancer Ollama sur votre machine avant d'ouvrir Jupyter.

💻 Pour les utilisateurs Windows :

Télécharger : Rendez-vous sur ollama.com/download et téléchargez la version Windows.

Installer : Exécutez l'installeur (un petit icône de lama apparaîtra dans votre barre des tâches en bas à droite).

Télécharger le modèle : Ouvrez l'Invite de commandes (CMD) ou PowerShell et tapez la commande suivante :

DOS
ollama run mistral
(Note : Le premier lancement prendra quelques minutes pour télécharger les poids du modèle. Si vous avez utilisé un autre modèle comme llama3 dans le notebook, remplacez mistral par ce nom).

C'est prêt ! Laissez l'application Ollama tourner en arrière-plan et exécutez le notebook Jupyter normalement.

🍏 Pour les utilisateurs macOS :

Télécharger : Rendez-vous sur ollama.com/download et téléchargez la version macOS.

Installer : Décompressez le fichier et déplacez l'application Ollama dans votre dossier Applications. Lancez-la.

Télécharger le modèle : Ouvrez votre application Terminal et tapez :

Bash
ollama run mistral
C'est prêt ! Vous pouvez maintenant lancer le notebook, LangChain se connectera automatiquement à votre instance locale tournant en tâche de fond.