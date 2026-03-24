import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import numpy as np

from config import Config
from utils import save_pickle, load_pickle
from .crf_layer import CRF


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)
        self.W_O = nn.Linear(hidden_dim, hidden_dim)
    
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size = Q.shape[0]
        
        Q = self.split_heads(self.W_Q(Q))
        K = self.split_heads(self.W_K(K))
        V = self.split_heads(self.W_V(V))
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        output = self.W_O(output)
        
        return output


class KnowledgeEnhancedDataset(Dataset):
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
            
            kg_embedding = torch.zeros(max_len, Config.KG_EMBEDDING_DIM)
            
            self.data.append((
                input_ids,
                attention_mask,
                torch.tensor(label_ids, dtype=torch.long),
                torch.tensor(mask, dtype=torch.bool),
                kg_embedding
            ))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class KnowledgeEnhancedModel(nn.Module):
    def __init__(self, bert_model_name: str, hidden_dim: int, num_layers: int, num_tags: int, kg_emb_dim: int, num_heads: int, dropout: float):
        super(KnowledgeEnhancedModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        
        self.kg_projection = nn.Linear(kg_emb_dim, self.bert.config.hidden_size)
        
        self.multi_head_attn = MultiHeadAttention(self.bert.config.hidden_size, num_heads)
        self.layer_norm = nn.LayerNorm(self.bert.config.hidden_size)
        
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
    
    def forward(self, input_ids, attention_mask=None, tags=None, mask=None, kg_embeddings=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        e_bert = outputs.last_hidden_state
        e_bert = self.dropout(e_bert)
        
        if kg_embeddings is not None:
            h_kg = self.kg_projection(kg_embeddings)
            h_kg = self.dropout(h_kg)
            
            attn_output = self.multi_head_attn(e_bert, h_kg, h_kg)
            h_att = self.layer_norm(e_bert + attn_output)
        else:
            h_att = e_bert
        
        lstm_output, _ = self.lstm(h_att)
        lstm_output = self.dropout(lstm_output)
        
        emissions = self.fc(lstm_output)
        
        if tags is not None and mask is not None:
            loss = self.crf(emissions, tags, mask)
            return loss, emissions
        else:
            return emissions
    
    def decode(self, emissions, mask=None):
        return self.crf.decode(emissions, mask)


class KnowledgeEnhancedNER:
    def __init__(self):
        self.config = Config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = BertTokenizer.from_pretrained(self.config.BERT_MODEL_NAME)
        self.tag2id = Config.TAG2ID
        self.id2tag = Config.ID2TAG
        
        self.model = None
        self.is_trained = False
    
    def train(self, train_sentences: List[Tuple[List[str], List[str]]], val_sentences: List[Tuple[List[str], List[str]]] = None):
        train_dataset = KnowledgeEnhancedDataset(train_sentences, self.tokenizer, self.tag2id, self.config.MAX_SEQ_LEN)
        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        
        val_loader = None
        if val_sentences:
            val_dataset = KnowledgeEnhancedDataset(val_sentences, self.tokenizer, self.tag2id, self.config.MAX_SEQ_LEN)
            val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        self.model = KnowledgeEnhancedModel(
            bert_model_name=self.config.BERT_MODEL_NAME,
            hidden_dim=self.config.BILSTM_HIDDEN_DIM,
            num_layers=self.config.BILSTM_NUM_LAYERS,
            num_tags=self.config.NUM_TAGS,
            kg_emb_dim=self.config.KG_EMBEDDING_DIM,
            num_heads=8,
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
            
            for input_ids, attention_mask, label_ids, mask, kg_embeddings in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.config.EPOCHS}'):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                label_ids = label_ids.to(self.device)
                mask = mask.to(self.device)
                kg_embeddings = kg_embeddings.to(self.device)
                
                optimizer.zero_grad()
                crf_loss, emissions = self.model(input_ids, attention_mask, label_ids, mask, kg_embeddings)
                
                pred_probs = F.softmax(emissions, dim=-1)
                knowledge_prior = torch.ones_like(pred_probs.shape) / self.config.NUM_TAGS
                knowledge_prior = knowledge_prior.to(self.device)
                kl_loss = F.kl_div(
                    torch.log(pred_probs + 1e-10),
                    knowledge_prior,
                    reduction='batchmean'
                )
                
                total_loss_val = self.config.ALPHA * crf_loss + (1 - self.config.ALPHA) * kl_loss
                total_loss_val.backward()
                optimizer.step()
                
                total_loss += total_loss_val.item()
            
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
            for input_ids, attention_mask, label_ids, mask, kg_embeddings in val_loader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                label_ids = label_ids.to(self.device)
                mask = mask.to(self.device)
                kg_embeddings = kg_embeddings.to(self.device)
                
                crf_loss, emissions = self.model(input_ids, attention_mask, label_ids, mask, kg_embeddings)
                
                pred_probs = F.softmax(emissions, dim=-1)
                knowledge_prior = torch.ones_like(pred_probs.shape) / self.config.NUM_TAGS
                knowledge_prior = knowledge_prior.to(self.device)
                kl_loss = F.kl_div(
                    torch.log(pred_probs + 1e-10),
                    knowledge_prior,
                    reduction='batchmean'
                )
                
                total_loss_val = self.config.ALPHA * crf_loss + (1 - self.config.ALPHA) * kl_loss
                total_loss += total_loss_val.item()
        
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
        kg_embedding = torch.zeros(1, self.config.MAX_SEQ_LEN, self.config.KG_EMBEDDING_DIM).to(self.device)
        
        with torch.no_grad():
            emissions = self.model(input_ids, attention_mask, kg_embeddings=kg_embedding)
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
            self.model = KnowledgeEnhancedModel(
                bert_model_name=model_dir,
                hidden_dim=self.config.BILSTM_HIDDEN_DIM,
                num_layers=self.config.BILSTM_NUM_LAYERS,
                num_tags=self.config.NUM_TAGS,
                kg_emb_dim=self.config.KG_EMBEDDING_DIM,
                num_heads=8,
                dropout=self.config.DROPOUT_RATE
            ).to(self.device)
            self.model.load_state_dict(model_data['model_state_dict'])
