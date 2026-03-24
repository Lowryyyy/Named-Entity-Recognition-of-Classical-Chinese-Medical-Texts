import os
import argparse
import sys
from typing import Dict, Any

from config import Config
from utils import (
    set_seed,
    load_conll_data,
    save_conll_data,
    split_data,
    compute_metrics,
    print_metrics
)
from models import (
    HMMNER,
    CRFNER,
    BiLSTMNER,
    BiLSTMCRFNER,
    BERTNER,
    BERTCRFNER,
    BERTBiLSTMCRFNER,
    KnowledgeEnhancedNER
)


MODEL_MAP = {
    'hmm': HMMNER,
    'crf': CRFNER,
    'bilstm': BiLSTMNER,
    'bilstm_crf': BiLSTMCRFNER,
    'bert': BERTNER,
    'bert_crf': BERTCRFNER,
    'bert_bilstm_crf': BERTBiLSTMCRFNER,
    'knowledge_enhanced': KnowledgeEnhancedNER
}


def create_sample_data(data_dir: str):
    sample_sentences = [
        (['麻', '黄', '汤', '主', '之'], ['B-方剂', 'I-方剂', 'I-方剂', 'O', 'O']),
        (['治', '太', '阳', '病', '头', '痛'], ['O', 'B-疾病', 'I-疾病', 'I-疾病', 'B-证候', 'I-证候']),
        (['发', '热', '恶', '寒'], ['B-证候', 'I-证候', 'B-证候', 'I-证候']),
        (['桂', '枝', '三', '两'], ['B-药物', 'I-药物', 'O', 'O']),
        (['甘', '草', '一', '两', '炙'], ['B-药物', 'I-药物', 'O', 'O', 'O']),
        (['水', '七', '升', '煮', '取', '三', '升'], ['O', 'O', 'O', 'B-煎服法', 'I-煎服法', 'O', 'O']),
        (['去', '滓', '温', '服', '一', '升'], ['O', 'O', 'B-煎服法', 'I-煎服法', 'O', 'O']),
        (['其', '他', '药', '物', '暂', '不', '列'], ['O', 'O', 'O', 'O', 'O', 'O', 'O']),
    ]
    
    train_path = os.path.join(data_dir, 'train.conll')
    val_path = os.path.join(data_dir, 'val.conll')
    test_path = os.path.join(data_dir, 'test.conll')
    
    save_conll_data(sample_sentences[:5], train_path)
    save_conll_data(sample_sentences[5:7], val_path)
    save_conll_data(sample_sentences[7:], test_path)
    
    print(f'Sample data created at: {data_dir}')
    return sample_sentences[:5], sample_sentences[5:7], sample_sentences[7:]


def train_model(model_name: str, train_data: list, val_data: list, model_path: str):
    if model_name not in MODEL_MAP:
        raise ValueError(f'Unknown model: {model_name}. Available models: {list(MODEL_MAP.keys())}')
    
    print(f'Training {model_name} model...')
    model = MODEL_MAP[model_name]()
    
    if model_name in ['hmm', 'crf']:
        model.train(train_data)
    else:
        model.train(train_data, val_data)
    
    model.save(model_path)
    print(f'Model saved to: {model_path}')
    return model


def evaluate_model(model, test_data: list):
    true_tags_list = [tags for _, tags in test_data]
    words_list = [words for words, _ in test_data]
    
    pred_tags_list = model.predict_batch(words_list)
    
    metrics = compute_metrics(true_tags_list, pred_tags_list)
    print_metrics(metrics)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='中医古籍命名实体识别')
    parser.add_argument('--model', type=str, required=True, choices=list(MODEL_MAP.keys()),
                        help='模型名称: hmm, crf, bilstm, bilstm_crf, bert, bert_crf, bert_bilstm_crf, knowledge_enhanced')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'eval', 'both'],
                        help='运行模式: train, eval, both')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='数据目录 (包含 train.conll, val.conll, test.conll)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型保存/加载路径')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    Config.ensure_dirs()
    
    if args.data_dir is None:
        args.data_dir = Config.DATA_DIR
    
    if args.model_path is None:
        args.model_path = os.path.join(Config.MODEL_DIR, f'{args.model}.pkl')
    
    print('=' * 80)
    print('中医古籍命名实体识别系统')
    print('=' * 80)
    
    train_data, val_data, test_data = None, None, None
    
    if os.path.exists(os.path.join(args.data_dir, 'train.conll')):
        print(f'Loading data from {args.data_dir}...')
        train_data = load_conll_data(os.path.join(args.data_dir, 'train.conll'))
        val_data = load_conll_data(os.path.join(args.data_dir, 'val.conll'))
        test_data = load_conll_data(os.path.join(args.data_dir, 'test.conll'))
    else:
        print('Creating sample data...')
        train_data, val_data, test_data = create_sample_data(args.data_dir)
    
    print(f'Train sentences: {len(train_data)}')
    print(f'Val sentences: {len(val_data)}')
    print(f'Test sentences: {len(test_data)}')
    
    model = None
    
    if args.mode in ['train', 'both']:
        model = train_model(args.model, train_data, val_data, args.model_path)
    
    if args.mode in ['eval', 'both']:
        if model is None:
            if not os.path.exists(args.model_path):
                raise FileNotFoundError(f'Model not found at {args.model_path}')
            print(f'Loading model from {args.model_path}...')
            model = MODEL_MAP[args.model]()
            model.load(args.model_path)
        
        print('\nEvaluating on test set...')
        evaluate_model(model, test_data)
    
    print('=' * 80)
    print('Done!')
    print('=' * 80)


if __name__ == '__main__':
    main()
