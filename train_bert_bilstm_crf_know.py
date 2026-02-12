import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report
import json

from bert_bilstm_crf_know import BertBiLstmCrfKnow
from processors.ner_seq import ner_processors, convert_examples_to_features
from metrics.ner_metrics import SeqEntityScore

class TCMNERDataset(Dataset):
    """中医药NER数据集"""
    def __init__(self, examples, tokenizer, label_list, max_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.max_length = max_length
        self.label2id = {label: i for i, label in enumerate(label_list)}
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # 编码文本
        encoding = self.tokenizer(
            example.text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # 处理标签
        labels = [self.label2id.get(tag, 0) for tag in example.labels]
        labels = labels[:self.max_length]
        
        # 填充标签
        if len(labels) < self.max_length:
            labels += [0] * (self.max_length - len(labels))
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'token_type_ids': encoding['token_type_ids'].squeeze(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

def train_model(args):
    """训练模型"""
    # 初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    
    # 加载数据
    processor = ner_processors[args.task_name]()
    label_list = processor.get_labels()
    
    # 训练数据
    train_examples = processor.get_train_examples(args.data_dir)
    train_dataset = TCMNERDataset(train_examples, tokenizer, label_list, args.max_seq_length)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    
    # 验证数据
    dev_examples = processor.get_dev_examples(args.data_dir)
    dev_dataset = TCMNERDataset(dev_examples, tokenizer, label_list, args.max_seq_length)
    dev_loader = DataLoader(dev_dataset, batch_size=args.eval_batch_size)
    
    # 初始化模型
    config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=len(label_list))
    model = BertBiLstmCrfKnow.from_pretrained(
        args.model_name_or_path,
        config=config,
        num_labels=len(label_list)
    )
    model.set_tokenizer(tokenizer)
    model.to(device)
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # 学习率调度器
    total_steps = len(train_loader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    
    # 训练循环
    best_f1 = 0
    for epoch in range(args.num_train_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_train_epochs}")
        
        # 训练阶段
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 前向传播
            outputs = model(**batch)
            loss = outputs['loss']
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # 验证阶段
        model.eval()
        eval_metric = SeqEntityScore(label_list)
        
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc="Evaluating"):
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop('labels')
                
                outputs = model(**batch)
                predictions = outputs['predictions']
                
                # 更新指标
                for pred, label in zip(predictions, labels.cpu().numpy()):
                    # 移除填充
                    mask = label != -100
                    pred_filtered = pred[:sum(mask)]
                    label_filtered = label[mask]
                    
                    eval_metric.update(pred_filtered, label_filtered)
        
        # 计算指标
        eval_result = eval_metric.result()
        f1_score = eval_result['f1']
        
        print(f"Validation F1: {f1_score:.4f}")
        
        # 保存最佳模型
        if f1_score > best_f1:
            best_f1 = f1_score
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"Best model saved with F1: {best_f1:.4f}")
    
    print(f"\nTraining completed. Best F1: {best_f1:.4f}")

def predict(args, model, tokenizer, text):
    """预测函数"""
    model.eval()
    device = next(model.parameters()).device
    
    # 编码文本
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=args.max_seq_length,
        padding='max_length',
        return_tensors='pt'
    )
    
    encoding = {k: v.to(device) for k, v in encoding.items()}
    
    with torch.no_grad():
        outputs = model(**encoding)
        predictions = outputs['predictions'][0]
        
        # 转换标签
        tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
        pred_labels = [label_list[p] for p in predictions]
        
        # 提取实体
        entities = []
        current_entity = None
        current_label = None
        
        for i, (token, label) in enumerate(zip(tokens, pred_labels)):
            if label.startswith('B-'):
                if current_entity:
                    entities.append((current_entity, current_label))
                current_entity = token
                current_label = label[2:]
            elif label.startswith('I-') and current_entity:
                current_entity += token.replace('##', '')
            elif label == 'O' and current_entity:
                entities.append((current_entity, current_label))
                current_entity = None
                current_label = None
        
        if current_entity:
            entities.append((current_entity, current_label))
    
    return entities

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    
    # 数据参数
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-chinese")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--task_name", type=str, default="tcm_ner")
    
    # 训练参数
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 训练模型
    train_model(args)