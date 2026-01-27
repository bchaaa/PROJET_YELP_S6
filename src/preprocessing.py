import string
import re
import nltk
from nltk.corpus import stopwords

# On telecharge la liste des mots vides (le, la, a, is, the...)
# Le try/except evite de le re-telecharger a chaque fois si on l'a deja
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# On garde les stopwords en memoire pour aller vite
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
    Nettoie un texte brut pour le Machine Learning.
    - Minuscules
    - Suppression ponctuation
    - Suppression chiffres
    - Suppression stopwords (mots vides)
    """
    # Securite si le texte est vide ou pas un string
    if not isinstance(text, str):
        return ""
    
    # 1. Mise en minuscule
    text = text.lower()
    
    # 2. Suppression de la ponctuation (.,!?)
    # On remplace tout ce qui est ponctuation par rien
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 3. Suppression des chiffres 
    text = re.sub(r'\d+', '', text)
    
    # 4. Suppression des stopwords et espaces en trop
    words = text.split()
    # On ne garde que les mots qui ne sont PAS dans la liste STOPWORDS
    clean_words = [w for w in words if w not in STOPWORDS]
    
    # On recolle les morceaux
    return " ".join(clean_words)
