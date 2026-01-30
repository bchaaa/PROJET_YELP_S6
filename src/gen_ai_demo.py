import torch
from transformers import pipeline
import os
import random
from preprocessing import clean_text
from dataset import YelpDataset, Vocabulary 
from models import YelpClassifier

def clean_output(text):
    """Coupe le texte au dernier signe de ponctuation pour éviter les phrases non finies."""
    last_punc = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
    if last_punc != -1:
        return text[:last_punc+1]
    return text

def main():
    print("--- Démarrage de la démo IA Générative + Classification ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. GÉNÉRATION DE TEXTE (Guidée)
    print("\n1. Chargement du générateur (GPT-2)...")
    generator = pipeline('text-generation', model='gpt2', framework='pt')
    
    # ON GUIDE L'IA : On choisit aléatoirement un scénario clair
    scenarios = [
        {"type": "Positif", "prompt": "The food at this restaurant was absolutely amazing because"},
        {"type": "Négatif", "prompt": "I will never go back to this restaurant because the food was"},
        {"type": "Mitigé", "prompt": "The restaurant was okay but the service was"}
    ]
    
    scenario = random.choice(scenarios)
    prompt = scenario["prompt"]
    print(f"   Scénario choisi : {scenario['type']}")
    print(f"   Prompt : '{prompt}'")
    
    # Paramètres ajustés pour plus de cohérence
    result = generator(
        prompt, 
        max_new_tokens=40,       # Pas trop long pour qu'il ne perde pas le fil
        do_sample=True, 
        temperature=0.7,         # Créativité modérée
        top_k=50,                # Se limite aux 50 mots les plus probables
        top_p=0.95,              # Nucleus sampling (évite les mots bizarres)
        repetition_penalty=1.2,  # Évite les répétitions
        truncation=True,
        pad_token_id=generator.tokenizer.eos_token_id
    )
    
    raw_text = result[0]['generated_text']
    # On nettoie pour avoir une phrase finie
    generated_review = clean_output(raw_text)
    
    print(f"\n📝 Review générée par l'IA :\n\"{generated_review}\"")
    
    # 2. ANALYSE PAR TON MODÈLE LSTM
    print("\n2. Chargement de ton modèle entraîné...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "yelp_lstm_model.pth")
    
    if not os.path.exists(model_path):
        print("ERREUR : Lance d'abord main.py pour sauvegarder le modèle !")
        return

    checkpoint = torch.load(model_path, map_location=device)
    vocab_stoi = checkpoint['vocab_stoi']
    
    # Reconstruction rapide du vocabulaire pour l'inférence
    vocab = Vocabulary()
    vocab.stoi = vocab_stoi
    
    EMBED_DIM = checkpoint['embed_dim']
    HIDDEN_DIM = checkpoint['hidden_dim']
    OUTPUT_DIM = checkpoint['output_dim']
    INPUT_DIM = len(vocab_stoi)
    
    model = YelpClassifier(INPUT_DIM, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval() 
    
    print("   ✅ Modèle chargé.")

    # Prédiction
    cleaned = clean_text(generated_review)
    indices = [vocab.stoi.get(token, vocab.stoi.get("<UNK>", 1)) for token in cleaned.split()]
    
    if not indices: indices = [0] 
        
    tensor_input = torch.tensor(indices).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(tensor_input)
        predicted_class = prediction.argmax(dim=1).item() + 1
        
    print(f"\n⭐ Note prédite par ton LSTM : {predicted_class}/5")
    
    # Vérification de cohérence entre le scénario et la note
    print(f"   (Contexte attendu : {scenario['type']})")
    
    if (scenario['type'] == "Positif" and predicted_class >= 4) or \
       (scenario['type'] == "Négatif" and predicted_class <= 2):
         print("   ✅ L'IA et le Classifieur sont d'accord !")
    else:
         print("   ⚠️ Légère divergence (Le texte généré est peut-être ambigu).")

if __name__ == "__main__":
    main()