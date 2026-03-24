import os
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from typing import List, Tuple
from config import Config
from utils import extract_crf_features, save_pickle, load_pickle


class CRFNER:
    def __init__(self):
        self.tag2id = Config.TAG2ID
        self.id2tag = Config.ID2TAG
        
        self.model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        self.is_trained = False
    
    def train(self, sentences: List[Tuple[List[str], List[str]]]):
        X_train = []
        y_train = []
        
        for words, tags in sentences:
            features = extract_crf_features(words)
            X_train.append(features)
            y_train.append(tags)
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
    
    def predict(self, words: List[str]) -> List[str]:
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        features = extract_crf_features(words)
        return self.model.predict_single(features)
    
    def predict_batch(self, sentences: List[List[str]]) -> List[List[str]]:
        X_test = [extract_crf_features(words) for words in sentences]
        return self.model.predict(X_test)
    
    def save(self, model_path: str):
        save_pickle(self.model, model_path)
    
    def load(self, model_path: str):
        self.model = load_pickle(model_path)
        self.is_trained = True
