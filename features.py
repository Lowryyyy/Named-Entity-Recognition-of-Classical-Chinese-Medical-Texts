import jieba
from typing import List, Dict, Any


def extract_char_features(sentence: List[str], idx: int) -> Dict[str, Any]:
    word = sentence[idx]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    
    if idx > 0:
        prev_word = sentence[idx - 1]
        features.update({
            '-1:word.lower()': prev_word.lower(),
            '-1:word.istitle()': prev_word.istitle(),
            '-1:word.isupper()': prev_word.isupper(),
        })
    else:
        features['BOS'] = True
    
    if idx < len(sentence) - 1:
        next_word = sentence[idx + 1]
        features.update({
            '+1:word.lower()': next_word.lower(),
            '+1:word.istitle()': next_word.istitle(),
            '+1:word.isupper()': next_word.isupper(),
        })
    else:
        features['EOS'] = True
    
    return features


def extract_crf_features(sentence: List[str]) -> List[Dict[str, Any]]:
    return [extract_char_features(sentence, i) for i in range(len(sentence))]


def get_pos_tags(sentence: List[str]) -> List[str]:
    pos_tags = []
    text = ''.join(sentence)
    words = jieba.lcut(text)
    word_pos = 0
    for word in words:
        word_len = len(word)
        for i in range(word_len):
            if i == 0:
                pos_tags.append('B')
            else:
                pos_tags.append('I')
        word_pos += word_len
    return pos_tags[:len(sentence)]


def extract_hmm_features(sentence: List[str]) -> List[Dict[str, Any]]:
    pos_tags = get_pos_tags(sentence)
    features = []
    for i, (word, pos) in enumerate(zip(sentence, pos_tags)):
        feat = {
            'word': word,
            'pos': pos,
            'prefix1': word[0] if len(word) > 0 else '',
            'prefix2': word[:2] if len(word) > 1 else '',
            'suffix1': word[-1] if len(word) > 0 else '',
            'suffix2': word[-2:] if len(word) > 1 else '',
        }
        features.append(feat)
    return features
