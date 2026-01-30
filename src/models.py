import torch
import torch.nn as nn

class YelpClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(YelpClassifier, self).__init__()
        
        # 1. Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 2. LSTM (Long Short-Term Memory)
        # batch_first=True signifie que l'entrée est [batch, seq_len, features]
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        
        # 3. Couche linéaire
        # On prend la sortie cachée du LSTM
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        # text shape: [batch_size, sent_len]
        
        embedded = self.embedding(text)
        # embedded shape: [batch_size, sent_len, embed_dim]
        
        # Passage dans le LSTM
        # output : contient les états de chaque mot
        # (hidden, cell) : contient l'état final (résumé de la phrase)
        _, (hidden, cell) = self.lstm(embedded)
        
        # hidden shape: [1, batch_size, hidden_dim]
        # On prend le dernier état caché (le résumé final de la phrase)
        last_hidden = hidden[-1]
        
        return self.fc(last_hidden)