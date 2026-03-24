import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForTokenClassification
from typing import List, Tuple, Dict
from tqdm import tqdm
import numpy as np

from config import Config
from utils import save_pickle, load_pickle


class BERTDataset(Dataset):
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
                return_tensors='pt',
                return_offsets_mapping=True
            )
            
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
            offset_mapping = encoding['offset_mapping'].squeeze(0)
            
            label_ids = []
            word_ids = encoding.word_ids()
            previous_word_idx = None
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    if word_idx < len(tags):
                        label_ids.append(tag2id.get(tags[word_idx], tag2id['O']))
                    else:
                        label_ids.append(-100)
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            
            self.data.append((
                input_ids,
                attention_mask,
                torch.tensor(label_ids, dtype=torch.long)
            ))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class BERTNER:
    def __init__(self):
        self.config = Config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = BertTokenizer.from_pretrained(self.config.BERT_MODEL_NAME)
        self.tag2id = Config.TAG2ID
        self.id2tag = Config.ID2TAG
        
        self.model = None
        self.is_trained = False
    
    def train(self, train_sentences: List[Tuple[List[str], List[str]]], val_sentences: List[Tuple[List[str], List[str]]] = None):
        train_dataset = BERTDataset(train_sentences, self.tokenizer, self.tag2id, self.config.MAX_SEQ_LEN)
        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        
        val_loader = None
        if val_sentences:
            val_dataset = BERTDataset(val_sentences, self.tokenizer, self.tag2id, self.config.MAX_SEQ_LEN)
            val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        self.model = BertForTokenClassification.from_pretrained(
            self.config.BERT_MODEL_NAME,
            num_labels=self.config.NUM_TAGS,
            hidden_dropout_prob=self.config.DROPOUT_RATE
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
            
            for input_ids, attention_mask, label_ids in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.config.EPOCHS}'):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                label_ids = label_ids.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=label_ids
                )
                loss = outputs.loss
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
            for input_ids, attention_mask, label_ids in val_loader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                label_ids = label_ids.to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=label_ids
                )
                loss = outputs.loss
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
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        predictions = predictions[0].cpu().numpy()
        word_ids = encoding.word_ids()
        previous_word_idx = None
        predicted_tags = []
        
        for idx, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx != previous_word_idx:
                predicted_tags.append(self.id2tag[predictions[idx]])
                previous_word_idx = word_idx
        
        return predicted_tags[:len(words)]
    
    def predict_batch(self, sentences: List[List[str]]) -> List[List[str]]:
        return [self.predict(words) for words in sentences]
    
    def save(self, model_path: str):
        model_dir = os.path.splitext(model_path)[0]
        os.makedirs(model_dir, exist_ok=True)
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        save_pickle({'is_trained': self.is_trained}, model_path)
    
    def load(self, model_path: str):
        model_dir = os.path.splitext(model_path)[0]
        self.model = BertForTokenClassification.from_pretrained(model_dir).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        model_data = load_pickle(model_path)
        self.is_trained = model_data['is_trained']
