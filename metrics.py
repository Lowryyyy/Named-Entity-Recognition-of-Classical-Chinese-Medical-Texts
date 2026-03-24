from typing import List, Tuple, Dict, Set
from collections import defaultdict
from config import Config


def get_entities(tags: List[str]) -> List[Tuple[str, int, int]]:
    entities = []
    entity_start = None
    entity_type = None
    
    for i, tag in enumerate(tags):
        if tag.startswith('B-'):
            if entity_start is not None:
                entities.append((entity_type, entity_start, i - 1))
            entity_start = i
            entity_type = tag[2:]
        elif tag.startswith('I-'):
            if entity_start is not None and tag[2:] == entity_type:
                continue
            else:
                if entity_start is not None:
                    entities.append((entity_type, entity_start, i - 1))
                entity_start = None
                entity_type = None
        else:
            if entity_start is not None:
                entities.append((entity_type, entity_start, i - 1))
            entity_start = None
            entity_type = None
    
    if entity_start is not None:
        entities.append((entity_type, entity_start, len(tags) - 1))
    
    return entities


def compute_metrics(true_tags_list: List[List[str]], pred_tags_list: List[List[str]]) -> Dict[str, Dict[str, float]]:
    true_entities = defaultdict(set)
    pred_entities = defaultdict(set)
    
    for idx, (true_tags, pred_tags) in enumerate(zip(true_tags_list, pred_tags_list)):
        for etype, start, end in get_entities(true_tags):
            true_entities[etype].add((idx, start, end))
        for etype, start, end in get_entities(pred_tags):
            pred_entities[etype].add((idx, start, end))
    
    all_entity_types = set(true_entities.keys()) | set(pred_entities.keys())
    metrics = {}
    
    for etype in all_entity_types:
        true_set = true_entities.get(etype, set())
        pred_set = pred_entities.get(etype, set())
        
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[etype] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    total_tp = sum(len(true_entities.get(et, set()) & pred_entities.get(et, set())) for et in all_entity_types)
    total_fp = sum(len(pred_entities.get(et, set()) - true_entities.get(et, set())) for et in all_entity_types)
    total_fn = sum(len(true_entities.get(et, set()) - pred_entities.get(et, set())) for et in all_entity_types)
    
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    
    metrics['micro'] = {
        'precision': micro_precision,
        'recall': micro_recall,
        'f1': micro_f1
    }
    
    if all_entity_types:
        macro_precision = sum(metrics[et]['precision'] for et in all_entity_types) / len(all_entity_types)
        macro_recall = sum(metrics[et]['recall'] for et in all_entity_types) / len(all_entity_types)
        macro_f1 = sum(metrics[et]['f1'] for et in all_entity_types) / len(all_entity_types)
        
        metrics['macro'] = {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1
        }
    
    return metrics


def print_metrics(metrics: Dict[str, Dict[str, float]]):
    print('\n' + '=' * 80)
    print(f'{"Entity Type":<15} {"Precision":<12} {"Recall":<12} {"F1":<12}')
    print('-' * 80)
    
    for etype in sorted(metrics.keys()):
        if etype in ['micro', 'macro']:
            continue
        m = metrics[etype]
        print(f'{etype:<15} {m["precision"]:>10.4f}   {m["recall"]:>10.4f}   {m["f1"]:>10.4f}')
    
    print('-' * 80)
    if 'macro' in metrics:
        m = metrics['macro']
        print(f'{"Macro Average":<15} {m["precision"]:>10.4f}   {m["recall"]:>10.4f}   {m["f1"]:>10.4f}')
    if 'micro' in metrics:
        m = metrics['micro']
        print(f'{"Micro Average":<15} {m["precision"]:>10.4f}   {m["recall"]:>10.4f}   {m["f1"]:>10.4f}')
    print('=' * 80 + '\n')
