import os

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    LOG_DIR = os.path.join(BASE_DIR, 'logs')
    
    ENTITY_TYPES = ['药物', '疾病', '证候', '煎服法', '方剂', '其他']
    
    BIO_TAGS = ['O']
    for et in ENTITY_TYPES:
        BIO_TAGS.append(f'B-{et}')
        BIO_TAGS.append(f'I-{et}')
    
    TAG2ID = {tag: idx for idx, tag in enumerate(BIO_TAGS)}
    ID2TAG = {idx: tag for idx, tag in enumerate(BIO_TAGS)}
    NUM_TAGS = len(BIO_TAGS)
    
    MAX_SEQ_LEN = 128
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    DROPOUT_RATE = 0.3
    EARLY_STOPPING_PATIENCE = 5
    
    EMBEDDING_DIM = 100
    BILSTM_HIDDEN_DIM = 256
    BILSTM_NUM_LAYERS = 2
    
    BERT_MODEL_NAME = 'bert-base-chinese'
    BERT_HIDDEN_DIM = 768
    
    KG_EMBEDDING_DIM = 100
    ALPHA = 0.7
    
    @classmethod
    def ensure_dirs(cls):
        for dir_path in [cls.DATA_DIR, cls.MODEL_DIR, cls.LOG_DIR]:
            os.makedirs(dir_path, exist_ok=True)
