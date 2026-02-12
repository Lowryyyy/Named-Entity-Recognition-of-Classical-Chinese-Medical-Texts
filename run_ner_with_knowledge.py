#!/usr/bin/env python3
"""
运行BERT+BiLSTM+CRF+KNOW模型
"""

import os
import sys
import argparse
import torch
from transformers import BertTokenizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_bert_bilstm_crf_know import train_model, predict
from bert_bilstm_crf_know import BertBiLstmCrfKnow
from config import ModelConfig, TrainingConfig

def main():
    parser = argparse.ArgumentParser(description="中医药NER模型训练和预测")
    
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # 训练命令
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("--data_dir", type=str, required=True, help="数据目录")
    train_parser.add_argument("--output_dir", type=str, default="./output", help="输出目录")
    train_parser.add_argument("--model_name", type=str, default="bert-base-chinese", help="预训练模型")
    train_parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    train_parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    train_parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    
    # 预测命令
    predict_parser = subparsers.add_parser("predict", help="预测文本")
    predict_parser.add_argument("--model_dir", type=str, required=True, help="模型目录")
    predict_parser.add_argument("--text", type=str, required=True, help="待预测文本")
    predict_parser.add_argument("--output_file", type=str, help="输出文件")
    
    # 评估命令
    eval_parser = subparsers.add_parser("evaluate", help="评估模型")
    eval_parser.add_argument("--model_dir", type=str, required=True, help="模型目录")
    eval_parser.add_argument("--test_data", type=str, required=True, help="测试数据")
    
    args = parser.parse_args()
    
    if args.command == "train":
        # 训练模型
        train_args = argparse.Namespace(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            model_name_or_path=args.model_name,
            num_train_epochs=args.epochs,
            train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            eval_batch_size=32,
            max_seq_length=128,
            weight_decay=0.01,
            max_grad_norm=1.0,
            task_name="tcm_ner"
        )
        
        print("开始训练模型...")
        print(f"数据目录: {args.data_dir}")
        print(f"输出目录: {args.output_dir}")
        print(f"预训练模型: {args.model_name}")
        print(f"训练轮数: {args.epochs}")
        print(f"批次大小: {args.batch_size}")
        print(f"学习率: {args.learning_rate}")
        
        train_model(train_args)
        
    elif args.command == "predict":
        # 加载模型
        print(f"加载模型从: {args.model_dir}")
        
        tokenizer = BertTokenizer.from_pretrained(args.model_dir)
        model = BertBiLstmCrfKnow.from_pretrained(args.model_dir)
        model.eval()
        
        if torch.cuda.is_available():
            model.cuda()
        
        # 预测
        print(f"预测文本: {args.text}")
        entities = predict(args, model, tokenizer, args.text)
        
        # 输出结果
        print("\n识别到的实体:")
        for entity, label in entities:
            print(f"  {entity} -> {label}")
        
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                for entity, label in entities:
                    f.write(f"{entity}\t{label}\n")
            print(f"\n结果已保存到: {args.output_file}")
    
    elif args.command == "evaluate":
        # 评估模型
        print("评估功能待实现...")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()