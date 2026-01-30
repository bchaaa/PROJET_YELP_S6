import torch

def train_one_epoch(model, iterator, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for texts, labels in iterator:
        texts = texts.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        predictions = model(texts)
        loss = criterion(predictions, labels)
        loss.backward()
        
        # --- LA SÉCURITÉ (NOUVEAU) ---
        # On empêche les gradients d'exploser. Cela stabilise énormément les LSTM.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # -----------------------------
        
        optimizer.step()
        
        epoch_loss += loss.item()
        preds = predictions.argmax(dim=1)
        correct = (preds == labels).float()
        epoch_acc += correct.sum() / len(correct)
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# ... (Laisse la fonction evaluate telle quelle) ...
def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad():
        for texts, labels in iterator:
            texts = texts.to(device)
            labels = labels.to(device)
            
            predictions = model(texts)
            loss = criterion(predictions, labels)
            
            epoch_loss += loss.item()
            preds = predictions.argmax(dim=1)
            correct = (preds == labels).float()
            epoch_acc += correct.sum() / len(correct)
            
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# ... (Laisse plot_confusion_matrix tel quel) ...
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(model, iterator, device, classes):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for texts, labels in iterator:
            texts = texts.to(device)
            labels = labels.to(device)
            predictions = model(texts)
            preds = predictions.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Prédit')
    plt.ylabel('Réel')
    plt.title('Matrice de Confusion Normalisée')
    plt.savefig('confusion_matrix.png')
    print("Matrice de confusion sauvegardée sous 'confusion_matrix.png'")