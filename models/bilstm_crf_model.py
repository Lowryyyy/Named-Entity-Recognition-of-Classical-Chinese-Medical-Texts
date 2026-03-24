import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
from tqdm import tqdm
import numpy as np

from config import Config
from utils import save_pickle, load_pickle
from .crf_layer import CRF


class BiLSTMCRFDataset(Dataset):
    def __init__(self, sentences: List[Tuple[List[str], List[str]]], word2id: Dict[str, int], tag2id: Dict[str, int], max_len: int):
        self.word2id = word2id
        self.tag2id = tag2id
        self.max_len = max_len
        self.data = []
        
        for words, tags in sentences:
            word_ids = [word2id.get(w, word2id['<UNK>']) for w in words]
            tag_ids = [tag2id.get(t, tag2id['O']) for t in tags]
            
            original_len = len(word_ids)
            if original_len > max_len:
                word_ids = word_ids[:max_len]
                tag_ids = tag_ids[:max_len]
                original_len = max_len
            else:
                padding_len = max_len - original_len
                word_ids += [word2id['<PAD>']] * padding_len
                tag_ids += [tag2id['O']] * padding_len
            
            mask = [1] * original_len + [0] * (max_len - original_len)
            
            self.data.append((
                torch.tensor(word_ids, dtype=torch.long),
                torch.tensor(tag_ids, dtype=torch.long),
                torch.tensor(mask, dtype=torch.bool)
            ))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class BiLSTMCRFModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int, num_tags: int, dropout: float):
        super(BiLSTMCRFModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_tags)
        self.crf = CRF(num_tags, batch_first=True)
    
    def forward(self, x, tags=None, mask=None):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        emissions = self.fc(x)
        
        if tags is not None and mask is not None:
            loss = self.crf(emissions, tags, mask)
            return loss
        else:
            return emissions
    
    def decode(self, emissions, mask=None):
        return self.crf.decode(emissions, mask)


class BiLSTMCRFNER:
    def __init__(self):
        self.config = Config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.word2id = {'<PAD>': 0, '<UNK>': 1}
        self.tag2id = Config.TAG2ID
        self.id2tag = Config.ID2TAG
        
        self.model = None
        self.is_trained = False
    
    def _build_vocab(self, sentences: List[Tuple[List[str], List[str]]], min_freq: int = 2):
        from collections import defaultdict
        word_freq = defaultdict(int)
        for words, _ in sentences:
            for word in words:
                word_freq[word] += 1
        
        for word, freq in word_freq.items():
            if freq >= min_freq and word not in self.word2id:
                self.word2id[word] = len(self.word2id)
    
    def train(self, train_sentences: List[Tuple[List[str], List[str]]], val_sentences: List[Tuple[List[str], List[str]]] = None):
        self._build_vocab(train_sentences)
        
        train_dataset = BiLSTMCRFDataset(train_sentences, self.word2id, self.tag2id, self.config.MAX_SEQ_LEN)
        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        
        val_loader = None
        if val_sentences:
            val_dataset = BiLSTMCRFDataset(val_sentences, self.word2id, self.tag2id, self.config.MAX_SEQ_LEN)
            val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        self.model = BiLSTMCRFModel(
            vocab_size=len(self.word2id),
            embedding_dim=self.config.EMBEDDING_DIM,
            hidden_dim=self.config.BILSTM_HIDDEN_DIM,
            num_layers=self.config.BILSTM_NUM_LAYERS,
            num_tags=self.config.NUM_TAGS,
            dropout=0.5
        ).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.EPOCHS):
            self.model.train()
            total_loss = 0
            
            for word_ids, tag_ids, mask in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.config.EPOCHS}'):
                word_ids = word_ids.to(self.device)
                tag_ids = tag_ids.to(self.device)
                mask = mask.to(self.device)
                
                optimizer.zero_grad()
                loss = self.model(word_ids, tag_ids, mask)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_loader)
            print(f'Train Loss: {avg_train_loss:.4f}')
            
            if val_loader:
                val_loss = self._validate(val_loader)
                print(f'Val Loss: {val_loss:.4f}')
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                        print(f'Early stopping at epoch {epoch + 1}')
                        break
        
        self.is_trained = True
    
    def _validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for word_ids, tag_ids, mask in val_loader:
                word_ids = word_ids.to(self.device)
                tag_ids = tag_ids.to(self.device)
                mask = mask.to(self.device)
                
                loss = self.model(word_ids, tag_ids, mask)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def predict(self, words: List[str]) -> List[str]:
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.eval()
        
        word_ids = [self.word2id.get(w, self.word2id['<UNK>']) for w in words]
        original_len = len(word_ids)
        
        if original_len > self.config.MAX_SEQ_LEN:
            word_ids = word_ids[:self.config.MAX_SEQ_LEN]
            original_len = self.config.MAX_SEQ_LEN
        else:
            word_ids += [self.word2id['<PAD>']] * (self.config.MAX_SEQ_LEN - original_len)
        
        mask = [1] * original_len + [0] * (self.config.MAX_SEQ_LEN - original_len)
        
        with torch.no_grad():
            word_ids_tensor = torch.tensor([word_ids], dtype=torch.long).to(self.device)
            mask_tensor = torch.tensor([mask], dtype=torch.bool).to(self.device)
            
            emissions = self.model(word_ids_tensor)
            predicted_ids = self.model.decode(emissions, mask_tensor)
        
        predicted_tags = [self.id2tag[tid] for tid in predicted_ids[0]]
        return predicted_tags
    
    def predict_batch(self, sentences: List[List[str]]) -> List[List[str]]:
        return [self.predict(words) for words in sentences]
    
    def save(self, model_path: str):
        model_data = {
            'word2id': self.word2id,
            'model_state_dict': self.model.state_dict() if self.model else None,
            'is_trained': self.is_trained
        }
        save_pickle(model_data, model_path)
    
    def load(self, model_path: str):
        model_data = load_pickle(model_path)
        self.word2id = model_data['word2id']
        self.is_trained = model_data['is_trained']
        
        if model_data['model_state_dict']:
            self.model = BiLSTMCRFModel(
                vocab_size=len(self.word2id),
                embedding_dim=self.config.EMBEDDING_DIM,
                hidden_dim=self.config.BILSTM_HIDDEN_DIM,
                num_layers=self.config.BILSTM_NUM_LAYERS,
                num_tags=self.config.NUM_TAGS,
                dropout=0.5
            ).to(self.device)
            self.model.load_state_dict(model_data['model_state_dict'])
