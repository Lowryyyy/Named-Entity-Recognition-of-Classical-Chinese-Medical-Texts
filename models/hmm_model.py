import os
import numpy as np
from typing import List, Tuple
from collections import defaultdict
from config import Config
from utils import save_pickle, load_pickle


class HMMNER:
    def __init__(self):
        self.tag2id = Config.TAG2ID
        self.id2tag = Config.ID2TAG
        self.num_tags = Config.NUM_TAGS
        
        self.initial_prob = None
        self.transition_prob = None
        self.emission_prob = None
        
        self.word2id = {'<UNK>': 0}
        self.is_trained = False
    
    def _build_vocab(self, sentences: List[Tuple[List[str], List[str]]]):
        word_freq = defaultdict(int)
        for words, _ in sentences:
            for word in words:
                word_freq[word] += 1
        
        for word in word_freq:
            if word not in self.word2id:
                self.word2id[word] = len(self.word2id)
    
    def train(self, sentences: List[Tuple[List[str], List[str]]]):
        self._build_vocab(sentences)
        num_words = len(self.word2id)
        
        initial_count = np.zeros(self.num_tags)
        transition_count = np.zeros((self.num_tags, self.num_tags))
        emission_count = np.zeros((self.num_tags, num_words))
        
        for words, tags in sentences:
            if not words:
                continue
            
            tag_ids = [self.tag2id[t] if t in self.tag2id else self.tag2id['O'] for t in tags]
            word_ids = [self.word2id[w] if w in self.word2id else self.word2id['<UNK>'] for w in words]
            
            initial_count[tag_ids[0]] += 1
            
            for i in range(len(tag_ids) - 1):
                transition_count[tag_ids[i]][tag_ids[i + 1]] += 1
            
            for tag_id, word_id in zip(tag_ids, word_ids):
                emission_count[tag_id][word_id] += 1
        
        initial_sum = initial_count.sum()
        self.initial_prob = np.log((initial_count + 1) / (initial_sum + self.num_tags))
        
        transition_sum = transition_count.sum(axis=1, keepdims=True)
        self.transition_prob = np.log((transition_count + 1) / (transition_sum + self.num_tags))
        
        emission_sum = emission_count.sum(axis=1, keepdims=True)
        self.emission_prob = np.log((emission_count + 1) / (emission_sum + num_words))
        
        self.is_trained = True
    
    def predict(self, words: List[str]) -> List[str]:
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        word_ids = [self.word2id[w] if w in self.word2id else self.word2id['<UNK>'] for w in words]
        n = len(word_ids)
        
        if n == 0:
            return []
        
        viterbi = np.zeros((n, self.num_tags))
        backpointer = np.zeros((n, self.num_tags), dtype=int)
        
        viterbi[0] = self.initial_prob + self.emission_prob[:, word_ids[0]]
        
        for t in range(1, n):
            for tag_j in range(self.num_tags):
                scores = viterbi[t - 1] + self.transition_prob[:, tag_j]
                best_tag_i = np.argmax(scores)
                viterbi[t][tag_j] = scores[best_tag_i] + self.emission_prob[tag_j][word_ids[t]]
                backpointer[t][tag_j] = best_tag_i
        
        best_path = []
        best_last_tag = np.argmax(viterbi[-1])
        best_path.append(best_last_tag)
        
        for t in range(n - 1, 0, -1):
            best_last_tag = backpointer[t][best_last_tag]
            best_path.append(best_last_tag)
        
        best_path = best_path[::-1]
        return [self.id2tag[tid] for tid in best_path]
    
    def predict_batch(self, sentences: List[List[str]]) -> List[List[str]]:
        return [self.predict(words) for words in sentences]
    
    def save(self, model_path: str):
        model_data = {
            'initial_prob': self.initial_prob,
            'transition_prob': self.transition_prob,
            'emission_prob': self.emission_prob,
            'word2id': self.word2id,
            'is_trained': self.is_trained
        }
        save_pickle(model_data, model_path)
    
    def load(self, model_path: str):
        model_data = load_pickle(model_path)
        self.initial_prob = model_data['initial_prob']
        self.transition_prob = model_data['transition_prob']
        self.emission_prob = model_data['emission_prob']
        self.word2id = model_data['word2id']
        self.is_trained = model_data['is_trained']
