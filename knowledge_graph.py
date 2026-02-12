import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import json

class KnowledgeGraphEmbedding:
    """中医药知识图谱嵌入"""
    def __init__(self, kg_path=None, embedding_dim=200):
        """
        Args:
            kg_path: 知识图谱文件路径
            embedding_dim: 知识嵌入维度
        """
        self.embedding_dim = embedding_dim
        self.entity2id = {}
        self.id2entity = {}
        self.relation2id = {}
        self.id2relation = {}
        
        # 如果没有提供知识图谱，使用预训练的嵌入或随机初始化
        if kg_path and os.path.exists(kg_path):
            self.load_kg(kg_path)
            self.initialize_embeddings()
        else:
            # 使用预训练的中医药知识嵌入或随机初始化
            self.embeddings = nn.Embedding(10000, embedding_dim)
            
    def load_kg(self, kg_path):
        """加载知识图谱"""
        with open(kg_path, 'r', encoding='utf-8') as f:
            kg_data = json.load(f)
        
        # 构建实体和关系的映射
        entities = kg_data.get('entities', [])
        relations = kg_data.get('relations', [])
        triples = kg_data.get('triples', [])
        
        for i, entity in enumerate(entities):
            self.entity2id[entity] = i
            self.id2entity[i] = entity
            
        for i, relation in enumerate(relations):
            self.relation2id[relation] = i
            self.id2relation[i] = relation
            
        self.triples = triples
        
    def initialize_embeddings(self):
        """初始化实体和关系的嵌入"""
        # 使用TransE初始化
        n_entities = len(self.entity2id)
        n_relations = len(self.relation2id)
        
        # 随机初始化
        entity_emb = nn.Embedding(n_entities, self.embedding_dim)
        relation_emb = nn.Embedding(n_relations, self.embedding_dim)
        
        # 归一化
        entity_emb.weight.data = entity_emb.weight.data / torch.norm(
            entity_emb.weight.data, p=2, dim=1, keepdim=True
        )
        
        self.entity_embeddings = entity_emb
        self.relation_embeddings = relation_emb
        
    def get_entity_embedding(self, entity_name):
        """获取实体的嵌入"""
        if entity_name in self.entity2id:
            entity_id = torch.tensor([self.entity2id[entity_name]])
            return self.entity_embeddings(entity_id)
        else:
            # 返回零向量
            return torch.zeros(1, self.embedding_dim)
    
    def get_token_knowledge(self, token_text, token_context=None):
        """获取token对应的知识嵌入"""
        # 这里可以根据具体需求实现更复杂的知识检索
        # 例如：根据上下文检索相关的知识
        if token_text in self.entity2id:
            return self.get_entity_embedding(token_text)
        else:
            # 尝试近似匹配或返回相关知识的平均
            similar_entities = self.find_similar_entities(token_text)
            if similar_entities:
                embeddings = [self.get_entity_embedding(e) for e in similar_entities[:3]]
                return torch.mean(torch.stack(embeddings), dim=0)
            else:
                return torch.zeros(1, self.embedding_dim)
    
    def find_similar_entities(self, token_text, top_k=3):
        """查找相似的实体"""
        # 这里可以实现基于字符串相似度的查找
        # 或使用预训练的语义相似度模型
        similar = []
        for entity in self.entity2id.keys():
            if token_text in entity or entity in token_text:
                similar.append(entity)
            if len(similar) >= top_k:
                break
        return similar

class TCMKnowledgeGraph(KnowledgeGraphEmbedding):
    """中医药知识图谱专用类"""
    def __init__(self, embedding_dim=200):
        super().__init__(embedding_dim=embedding_dim)
        
        # 初始化中医药特定实体
        self.initialize_tcm_entities()
        
    def initialize_tcm_entities(self):
        """初始化中医药常见实体"""
        # 这里可以预定义一些中医药实体
        tcm_entities = [
            # 药物
            "桂枝", "麻黄", "附子", "甘草", "人参", "黄芪", "当归", "白芍",
            # 疾病
            "太阳病", "少阳病", "阳明病", "伤寒", "温病", "消渴", "中风",
            # 症状
            "发热", "恶寒", "头痛", "咳嗽", "脉浮", "脉沉",
            # 方剂
            "桂枝汤", "麻黄汤", "小柴胡汤", "大承气汤",
        ]
        
        for i, entity in enumerate(tcm_entities):
            self.entity2id[entity] = i
            self.id2entity[i] = entity
            
        # 随机初始化嵌入
        n_entities = len(self.entity2id)
        self.entity_embeddings = nn.Embedding(n_entities, self.embedding_dim)
        
    def get_token_knowledge_enhanced(self, token_text, context_tokens=None):
        """增强的知识获取，考虑上下文"""
        # 基础实体匹配
        if token_text in self.entity2id:
            return self.get_entity_embedding(token_text)
        
        # 复合实体识别（如"桂枝汤"）
        if context_tokens:
            # 检查是否构成复合实体
            window_size = 3
            start_idx = max(0, len(context_tokens) - window_size)
            context_window = context_tokens[start_idx:]
            
            # 尝试组合词
            for i in range(len(context_window)):
                for j in range(i+1, len(context_window)+1):
                    compound = ''.join(context_window[i:j])
                    if compound in self.entity2id:
                        return self.get_entity_embedding(compound)
        
        return torch.zeros(1, self.embedding_dim)