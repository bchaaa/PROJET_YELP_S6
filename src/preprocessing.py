import re
import string

def clean_text(text):
    """
    Nettoie le texte : minuscule, suppression ponctuation et caractères spéciaux.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Mise en minuscule
    text = text.lower()
    
    # 2. Suppression de la ponctuation
    # On remplace la ponctuation par des espaces
    text = text.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    
    # 3. Suppression des retours à la ligne et espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text