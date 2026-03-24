import os
import re
import json
import pickle
import random
import numpy as np
from typing import List, Tuple, Dict, Any
from config import Config


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def load_conll_data(file_path: str) -> List[Tuple[List[str], List[str]]]:
    sentences = []
    words = []
    tags = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if words:
                    sentences.append((words, tags))
                    words = []
                    tags = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    words.append(parts[0])
                    tags.append(parts[-1])
    
    if words:
        sentences.append((words, tags))
    
    return sentences


def save_conll_data(sentences: List[Tuple[List[str], List[str]]], file_path: str):
    with open(file_path, 'w', encoding='utf-8') as f:
        for words, tags in sentences:
            for word, tag in zip(words, tags):
                f.write(f'{word}\t{tag}\n')
            f.write('\n')


def split_data(sentences: List, train_ratio: float = 0.8, val_ratio: float = 0.1):
    random.shuffle(sentences)
    n = len(sentences)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return sentences[:train_end], sentences[train_end:val_end], sentences[val_end:]


def build_vocab(sentences: List[Tuple[List[str], List[str]]], min_freq: int = 2):
    word2id = {'<PAD>': 0, '<UNK>': 1}
    word_freq = {}
    
    for words, _ in sentences:
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    for word, freq in word_freq.items():
        if freq >= min_freq and word not in word2id:
            word2id[word] = len(word2id)
    
    id2word = {idx: word for word, idx in word2id.items()}
    return word2id, id2word


def save_pickle(obj: Any, file_path: str):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(file_path: str) -> Any:
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_json(obj: Any, file_path: str):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(file_path: str) -> Any:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
