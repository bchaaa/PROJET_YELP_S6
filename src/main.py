import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Import des modules locaux
from data_loader import load_yelp_sample
from preprocessing import clean_text
from dataset import YelpDataset
from models import YelpClassifier
from train_utils import train_one_epoch, evaluate, plot_confusion_matrix

def main():
    # --- CONFIGURATION ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Utilisation du device : {device}")
    
    BATCH_SIZE = 64
    # On réduit le taux d'apprentissage pour éviter les sautes d'humeur du modèle
    LEARNING_RATE = 0.0005 
    EPOCHS = 8  # On augmente un peu car on apprend plus lentement
    EMBED_DIM = 100
    HIDDEN_DIM = 64
    N_ROWS = 50000 
    
    # --- 1. CHARGEMENT ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "..", "data", "raw", "yelp_academic_reviews4students.jsonl")
    
    print(f"Chargement de {N_ROWS} lignes...")
    df = load_yelp_sample(file_path, n_rows=N_ROWS)
    if df is None: return

    # --- 2. EQUILIBRAGE ---
    print("\nCalcul des poids de classes...")
    y_labels = df['stars'].values - 1
    classes_uniques = np.unique(y_labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes_uniques, y=y_labels)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)
    print(f"Poids : {weights}")

    # --- 3. DATASET ---
    print("\nPréparation des données...")
    df['text'] = df['text'].apply(clean_text)
    full_dataset = YelpDataset(df)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # --- 4. MODÈLE ---
    INPUT_DIM = len(full_dataset.vocab.stoi)
    OUTPUT_DIM = 5 
    model = YelpClassifier(INPUT_DIM, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor).to(device)
    
    # --- 5. ENTRAÎNEMENT AVEC SAUVEGARDE INTELLIGENTE ---
    print("\nDémarrage de l'entraînement stabilisé...")
    
    best_valid_loss = float('inf') # On commence avec une perte infinie
    model_save_path = os.path.join(current_dir, "yelp_lstm_model.pth")
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val Acc: {val_acc*100:.2f}%')
        
        # SI le modèle est meilleur que le précédent, on sauvegarde !
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab_stoi': full_dataset.vocab.stoi,
                'embed_dim': EMBED_DIM,
                'hidden_dim': HIDDEN_DIM,
                'output_dim': OUTPUT_DIM
            }, model_save_path)
            print(f"   --> Nouveau record ! Modèle sauvegardé (Loss: {val_loss:.3f})")

    print("\nEntraînement terminé.")
    
    # --- 6. CHARGEMENT DU MEILLEUR MODÈLE POUR LES TESTS ---
    # On recharge le "champion" pour être sûr d'utiliser la meilleure version pour la matrice et les tests
    print("\nChargement du meilleur modèle pour évaluation...")
    checkpoint = torch.load(model_save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # --- 7. VISUALISATION ---
    classes = ["1 étoile", "2 étoiles", "3 étoiles", "4 étoiles", "5 étoiles"]
    try:
        plot_confusion_matrix(model, val_loader, device, classes)
    except:
        pass
    
    # --- 8. TEST MANUEL ---
    print("\n--- Test manuel (avec le meilleur modèle) ---")
    reviews_test = [
        "The service was slow and the food was cold.", 
        "Just okay. Nothing special but edible.", 
        "Absolutely fantastic! Best pizza ever." 
    ]
    model.eval()
    for rev in reviews_test:
        cleaned = clean_text(rev)
        indices = [full_dataset.vocab.stoi.get(t, 1) for t in cleaned.split()] # 1 = UNK
        if not indices: indices = [0]
        tensor_input = torch.tensor(indices).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(tensor_input).argmax(dim=1).item() + 1
        print(f"'{rev}' -> {pred}/5")

if __name__ == "__main__":
    main()