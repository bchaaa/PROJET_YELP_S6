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
    # 1. Construction du chemin vers data/raw
    # On part du dossier src/, on remonte (..), puis on va dans data/raw
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(current_dir, "..", "data", "raw")
    
    # On normalise le chemin (enlève les ".." pour faire propre)
    raw_data_dir = os.path.normpath(raw_data_dir)

    print(f"--- Recherche de fichiers dans : {raw_data_dir} ---\n")

    # 2. Vérification que le dossier existe
    if not os.path.exists(raw_data_dir):
        print(f"Erreur : Le dossier {raw_data_dir} n'existe pas.")
    else:
        # 3. On liste tous les fichiers du dossier
        all_files = os.listdir(raw_data_dir)
        
        # On filtre pour ne garder que les .json et .jsonl 
        # (évite de planter sur .DS_Store ou .gitignore)
        json_files = [f for f in all_files if f.endswith(('.json', '.jsonl'))]

        if not json_files:
            print("Aucun fichier JSON ou JSONL trouvé dans ce dossier.")
        
        # 4. Boucle sur chaque fichier trouvé
        for filename in json_files:
            full_path = os.path.join(raw_data_dir, filename)
            print(f"👉 Test du fichier : {filename}")
            
            # On charge un petit échantillon (ex: 100 lignes)
            df = load_yelp_sample(full_path, n_rows=100)
            
            if df is not None and not df.empty:
                print("   ✅ Chargement réussi.")
                print(f"   📏 Taille : {df.shape}")
                # Affiche les 5 premières colonnes dispo pour vérifier le contenu
                cols = list(df.columns[:5])
                print(f"   👀 Colonnes (extrait) : {cols}")
            else:
                print("   ❌ Échec ou fichier vide.")
            
            print("-" * 50)