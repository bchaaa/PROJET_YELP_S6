import torch
from torch.utils.data import Dataset
from collections import Counter

class Vocabulary:
    def __init__(self, freq_threshold=2):
        # Index 0 pour le padding, 1 pour les mots inconnus (UNK)
        self.itos = {0: "<PAD>", 1: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<UNK>": 1}
        self.freq_threshold = freq_threshold

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 2
        
        for sentence in sentence_list:
            for word in sentence.split():
                frequencies[word] += 1
                
                # On ajoute le mot au vocabulaire s'il apparait assez souvent
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
                    
    def numericalize(self, text):
        tokenized_text = text.split()
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

class YelpDataset(Dataset):
    def __init__(self, df, vocab=None, max_len=100):
        self.df = df
        self.texts = df['text'].tolist()
        # Les stars vont de 1 à 5, on décale à 0-4 pour PyTorch
        self.labels = [int(s) - 1 for s in df['stars'].tolist()]
        self.max_len = max_len
        
        # Si aucun vocabulaire n'est fourni, on le construit
        if vocab is None:
            self.vocab = Vocabulary()
            self.vocab.build_vocabulary(self.texts)
        else:
            self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        
        # Conversion texte -> liste d'entiers
        numericalized_text = self.vocab.numericalize(text)
        
        # Padding (remplissage) ou Truncating (coupe) pour avoir une taille fixe
        if len(numericalized_text) < self.max_len:
            padding = [self.vocab.stoi["<PAD>"]] * (self.max_len - len(numericalized_text))
            numericalized_text += padding
        else:
            numericalized_text = numericalized_text[:self.max_len]
            
        return torch.tensor(numericalized_text), torch.tensor(label)