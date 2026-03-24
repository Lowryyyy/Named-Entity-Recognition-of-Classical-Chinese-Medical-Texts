import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from typing import List, Tuple, Dict
from tqdm import tqdm
import numpy as np

from config import Config
from utils import save_pickle, load_pickle
from .crf_layer import CRF


class BERTBiLSTMCRFDataset(Dataset):
    def __init__(self, sentences: List[Tuple[List[str], List[str]]], tokenizer: BertTokenizer, tag2id: Dict[str, int], max_len: int):
        self.tokenizer = tokenizer
        self.tag2id = tag2id
        self.max_len = max_len
        self.data = []
        
        for words, tags in sentences:
            encoding = tokenizer(
                words,
                is_split_into_words=True,
                max_length=max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
            
            label_ids = []
            word_ids = encoding.word_ids()
            previous_word_idx = None
            mask = []
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(tag2id['O'])
                    mask.append(0)
                elif word_idx != previous_word_idx:
                    if word_idx < len(tags):
                        label_ids.append(tag2id.get(tags[word_idx], tag2id['O']))
                    else:
                        label_ids.append(tag2id['O'])
                    mask.append(1)
                    previous_word_idx = word_idx
                else:
                    label_ids.append(tag2id['O'])
                    mask.append(0)
            
            self.data.append((
                input_ids,
                attention_mask,
                torch.tensor(label_ids, dtype=torch.long),
                torch.tensor(mask, dtype=torch.bool)
            ))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class BERTBiLSTMCRFModel(nn.Module):
    def __init__(self, bert_model_name: str, hidden_dim: int, num_layers: int, num_tags: int, dropout: float):
        super(BERTBiLSTMCRFModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(
            self.bert.config.hidden_size,
            hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim, num_tags)
        self.crf = CRF(num_tags, batch_first=True)
    
    def forward(self, input_ids, attention_mask=None, tags=None, mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        lstm_output, _ = self.lstm(sequence_output)
        lstm_output = self.dropout(lstm_output)
        
        emissions = self.fc(lstm_output)
        
        if tags is not None and mask is not None:
            loss = self.crf(emissions, tags, mask)
            return loss
        else:
            return emissions
    
    def decode(self, emissions, mask=None):
        return self.crf.decode(emissions, mask)


class BERTBiLSTMCRFNER:
    def __init__(self):
        self.config = Config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = BertTokenizer.from_pretrained(self.config.BERT_MODEL_NAME)
        self.tag2id = Config.TAG2ID
        self.id2tag = Config.ID2TAG
        
        self.model = None
        self.is_trained = False
    
    def train(self, train_sentences: List[Tuple[List[str], List[str]]], val_sentences: List[Tuple[List[str], List[str]]] = None):
        train_dataset = BERTBiLSTMCRFDataset(train_sentences, self.tokenizer, self.tag2id, self.config.MAX_SEQ_LEN)
        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        
        val_loader = None
        if val_sentences:
            val_dataset = BERTBiLSTMCRFDataset(val_sentences, self.tokenizer, self.tag2id, self.config.MAX_SEQ_LEN)
            val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        self.model = BERTBiLSTMCRFModel(
            bert_model_name=self.config.BERT_MODEL_NAME,
            hidden_dim=self.config.BILSTM_HIDDEN_DIM,
            num_layers=self.config.BILSTM_NUM_LAYERS,
            num_tags=self.config.NUM_TAGS,
            dropout=self.config.DROPOUT_RATE
        ).to(self.device)
        
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.EPOCHS):
            self.model.train()
            total_loss = 0
            
            for input_ids, attention_mask, label_ids, mask in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.config.EPOCHS}'):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                label_ids = label_ids.to(self.device)
                mask = mask.to(self.device)
                
                optimizer.zero_grad()
                loss = self.model(input_ids, attention_mask, label_ids, mask)
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
            for input_ids, attention_mask, label_ids, mask in val_loader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                label_ids = label_ids.to(self.device)
                mask = mask.to(self.device)
                
                loss = self.model(input_ids, attention_mask, label_ids, mask)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def predict(self, words: List[str]) -> List[str]:
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.eval()
        
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.config.MAX_SEQ_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        word_ids = encoding.word_ids()
        mask = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is not None and word_idx != previous_word_idx:
                mask.append(1)
                previous_word_idx = word_idx
            else:
                mask.append(0)
        
        mask_tensor = torch.tensor([mask], dtype=torch.bool).to(self.device)
        
        with torch.no_grad():
            emissions = self.model(input_ids, attention_mask)
            predicted_ids = self.model.decode(emissions, mask_tensor)
        
        predicted_tags = []
        for idx, is_valid in enumerate(mask):
            if is_valid:
                predicted_tags.append(self.id2tag[predicted_ids[0][idx]])
        
        return predicted_tags[:len(words)]
    
    def predict_batch(self, sentences: List[List[str]]) -> List[List[str]]:
        return [self.predict(words) for words in sentences]
    
    def save(self, model_path: str):
        model_data = {
            'model_state_dict': self.model.state_dict() if self.model else None,
            'is_trained': self.is_trained
        }
        save_pickle(model_data, model_path)
        
        model_dir = os.path.splitext(model_path)[0] + '_bert'
        os.makedirs(model_dir, exist_ok=True)
        self.model.bert.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
    
    def load(self, model_path: str):
        model_data = load_pickle(model_path)
        self.is_trained = model_data['is_trained']
        
        if model_data['model_state_dict']:
            model_dir = os.path.splitext(model_path)[0] + '_bert'
            self.model = BERTBiLSTMCRFModel(
                bert_model_name=model_dir,
                hidden_dim=self.config.BILSTM_HIDDEN_DIM,
                num_layers=self.config.BILSTM_NUM_LAYERS,
                num_tags=self.config.NUM_TAGS,
                dropout=self.config.DROPOUT_RATE
            ).to(self.device)
            self.model.load_state_dict(model_data['model_state_dict'])
