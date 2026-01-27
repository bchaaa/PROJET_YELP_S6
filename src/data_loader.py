import pandas as pd
import json
import os

def load_yelp_sample(filepath, n_rows=10000):
    # Charge les n premieres lignes du fichier json
    # Sert a tester le code sans charger tout le dataset qui est trop lourd
    data = []
    
    # On verifie que le fichier est bien la
    if not os.path.exists(filepath):
        print(f"Erreur : le fichier {filepath} n'existe pas.")
        print("Verifie qu'il est bien dans data/raw/")
        return None

    print(f"Chargement de {n_rows} lignes depuis {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n_rows:
                break
            
            try:
                # Lecture ligne par ligne
                data.append(json.loads(line))
            except:
                continue # Si une ligne plante on passe a la suivante
                
    return pd.DataFrame(data)

def load_yelp_chunk(filepath, chunk_size=100000):
    # Generateur pour lire les gros fichiers par morceaux
    # Evite de saturer la ram du pc
    with open(filepath, 'r', encoding='utf-8') as f:
        data = []
        for i, line in enumerate(f):
            data.append(json.loads(line))
            
            # Quand on atteint la taille limite on renvoie le bloc
            if len(data) >= chunk_size:
                yield pd.DataFrame(data)
                data = []
        
        # On renvoie le reste a la fin
        if data:
            yield pd.DataFrame(data)

# Test rapide si on lance le fichier directement
if __name__ == "__main__":
    review_path = os.path.join("data", "raw", "review.json")
    
    # On teste avec juste 1000 lignes
    df = load_yelp_sample(review_path, n_rows=1000)
    
    if df is not None:
        print("Chargement reussi.")
        print(f"Taille du dataframe : {df.shape}")
        print("Apercu des donnees :")
        print(df[['text', 'stars']].head(2))
    else:
        print("Le chargement a echoue.")