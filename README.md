# 中医古籍命名实体识别模型

本项目实现了8种命名实体识别算法，用于中医古籍文本的实体抽取，支持六类实体：药物、疾病、证候、煎服法、方剂、其他。

## 项目结构

```
ddddd/
├── config.py                 # 配置文件
├── requirements.txt          # 依赖包
├── train_eval.py             # 训练和评估主入口
├── utils/                    # 工具模块
│   ├── __init__.py
│   ├── data_utils.py         # 数据处理工具
│   ├── metrics.py            # 评估指标
│   └── features.py           # 特征提取
└── models/                   # 模型实现
    ├── __init__.py
    ├── crf_layer.py          # CRF层实现
    ├── hmm_model.py          # HMM模型
    ├── crf_model.py          # CRF模型
    ├── bilstm_model.py       # BiLSTM模型
    ├── bilstm_crf_model.py   # BiLSTM+CRF模型
    ├── bert_model.py         # BERT模型
    ├── bert_crf_model.py     # BERT+CRF模型
    ├── bert_bilstm_crf_model.py  # BERT+BiLSTM+CRF模型
    └── knowledge_enhanced_model.py  # 知识增强模型
```

## 环境要求

- Python 3.7+
- PyTorch 1.10+
- Transformers 4.20+

## 安装依赖

```bash
pip install -r requirements.txt
```

## 模型说明

1. **HMM** - 隐马尔可夫模型
2. **CRF** - 条件随机场模型
3. **BiLSTM** - 双向长短期记忆网络
4. **BiLSTM+CRF** - BiLSTM编码 + CRF解码
5. **BERT** - BERT微调序列标注
6. **BERT+CRF** - BERT编码 + CRF解码
7. **BERT+BiLSTM+CRF** - BERT编码 → BiLSTM → CRF
8. **Knowledge-Enhanced** - 知识增强模型（融合知识图谱）

## 使用方法

### 1. 数据格式

数据需要使用CoNLL格式，每行包含一个词和对应的标签，空行分隔句子：

```
麻    B-方剂
黄    I-方剂
汤    I-方剂
主    O
之    O

治    O
太    B-疾病
阳    I-疾病
病    I-疾病
头    B-证候
痛    I-证候
```

### 2. 准备数据

在 `data/` 目录下放置三个文件：
- `train.conll` - 训练集
- `val.conll` - 验证集
- `test.conll` - 测试集

如果没有数据，程序会自动生成示例数据。

### 3. 训练模型

```bash
# 训练HMM模型
python train_eval.py --model hmm --mode train

# 训练BERT模型
python train_eval.py --model bert --mode train

# 训练知识增强模型
python train_eval.py --model knowledge_enhanced --mode train
```

### 4. 评估模型

```bash
# 评估已训练的模型
python train_eval.py --model bert --mode eval --model_path models/bert.pkl
```

### 5. 训练并评估

```bash
python train_eval.py --model bert_bilstm_crf --mode both
```

### 6. 命令行参数

- `--model`: 模型名称（必需）
- `--mode`: 运行模式：train/eval/both（必需）
- `--data_dir`: 数据目录（默认：data/）
- `--model_path`: 模型保存/加载路径（默认：models/{model}.pkl）
- `--seed`: 随机种子（默认：42）

## 实体类型

- `药物` - 中药名称
- `疾病` - 疾病名称
- `证候` - 证候描述
- `煎服法` - 煎药和服药方法
- `方剂` - 方剂名称
- `其他` - 其他实体

## 评估指标

- 精确率（Precision）
- 召回率（Recall）
- F1值
- 支持实体级别评估（CoNLL-2003标准）

## 配置说明

主要配置项在 `config.py` 中：

- `MAX_SEQ_LEN`: 最大序列长度
- `BATCH_SIZE`: 批大小
- `LEARNING_RATE`: 学习率
- `EPOCHS`: 训练轮数
- `EARLY_STOPPING_PATIENCE`: 早停耐心值
- `BERT_MODEL_NAME`: BERT模型名称

## 引用

本项目实现了以下算法：
- 隐马尔可夫模型（HMM）
- 条件随机场（CRF）
- BiLSTM
- BiLSTM+CRF
- BERT
- BERT+CRF
- BERT+BiLSTM+CRF
- 知识增强模型（融合知识图谱）
