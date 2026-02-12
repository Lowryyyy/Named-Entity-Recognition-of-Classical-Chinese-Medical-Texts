import os

class ModelConfig:
    """模型配置"""
    # BERT配置
    bert_model = "bert-base-chinese"
    hidden_size = 768
    num_hidden_layers = 12
    num_attention_heads = 12
    
    # 知识图谱配置
    knowledge_dim = 200
    kg_path = "./data/tcm_knowledge_graph.json"  # 知识图谱文件路径
    
    # BiLSTM配置
    lstm_hidden_size = 384  # hidden_size // 2
    lstm_num_layers = 2
    lstm_dropout = 0.3
    lstm_bidirectional = True
    
    # CRF配置
    crf_dropout = 0.1
    
    # 训练配置
    learning_rate = 2e-5
    crf_learning_rate = 1e-3
    weight_decay = 0.01
    warmup_proportion = 0.1
    
    # 知识融合配置
    fusion_hidden_dim = 512
    attention_heads = 8
    attention_dropout = 0.1
    
    # 损失权重
    alpha = 0.7  # CRF损失权重
    beta = 0.3   # 知识对齐损失权重

class TrainingConfig:
    """训练配置"""
    # 数据配置
    data_dir = "./data/tcm_ner"
    max_seq_length = 128
    train_batch_size = 16
    eval_batch_size = 32
    test_batch_size = 32
    
    # 训练参数
    num_train_epochs = 10
    gradient_accumulation_steps = 1
    max_grad_norm = 1.0
    logging_steps = 100
    save_steps = 500
    eval_steps = 500
    
    # 早停
    early_stopping_patience = 5
    early_stopping_threshold = 0.001
    
    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = False
    local_rank = -1
    
    # 输出
    output_dir = "./output"
    overwrite_output_dir = True
    seed = 42

class DataConfig:
    """数据配置"""
    # 实体类别
    entity_types = {
        "Drug": "药物",
        "Disease": "疾病",
        "Syndrome": "症状",
        "Decoction": "煎煮方法",
        "Formula": "方剂",
        "Other": "其他医学术语"
    }
    
    # BIO标签
    bio_labels = ["O"]
    for label in entity_types.keys():
        bio_labels.extend([f"B-{label}", f"I-{label}"])
    
    # 数据集分割
    train_ratio = 0.7
    dev_ratio = 0.15
    test_ratio = 0.15
    
    # 数据预处理
    convert_traditional_to_simplified = True
    remove_noise = True
    deduplicate = True