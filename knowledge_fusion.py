import torch
import torch.nn as nn
import torch.nn.functional as F

class KnowledgeFusionModule(nn.Module):
    """知识融合模块"""
    def __init__(self, bert_dim=768, knowledge_dim=200, hidden_dim=512):
        super().__init__()
        
        # 知识对齐矩阵
        self.knowledge_projection = nn.Linear(knowledge_dim, bert_dim)
        
        # 跨模态注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=bert_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(bert_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, bert_dim)
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(bert_dim)
        
    def forward(self, bert_output, knowledge_embeddings, attention_mask=None):
        """
        Args:
            bert_output: [batch_size, seq_len, bert_dim]
            knowledge_embeddings: [batch_size, seq_len, knowledge_dim]
            attention_mask: [batch_size, seq_len]
        Returns:
            fused_output: [batch_size, seq_len, bert_dim]
        """
        batch_size, seq_len, _ = bert_output.shape
        
        # 1. 投影知识嵌入到BERT空间
        knowledge_projected = self.knowledge_projection(knowledge_embeddings)  # [B, L, bert_dim]
        
        # 2. 跨模态注意力
        if attention_mask is not None:
            # 扩展注意力掩码
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            attention_mask = (1.0 - attention_mask) * -10000.0
            
        cross_output, cross_weights = self.cross_attention(
            query=bert_output,
            key=knowledge_projected,
            value=knowledge_projected,
            key_padding_mask=None,
            attn_mask=None
        )
        
        # 3. 特征拼接
        concatenated = torch.cat([bert_output, cross_output], dim=-1)  # [B, L, 2*bert_dim]
        
        # 4. 融合层
        fused = self.fusion_layer(concatenated)
        
        # 5. 残差连接和层归一化
        fused_output = self.layer_norm(fused + bert_output)
        
        return fused_output, cross_weights

class KnowledgeAlignmentLoss(nn.Module):
    """知识对齐损失"""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        
    def forward(self, entity_embeddings, knowledge_embeddings, entity_pairs):
        """
        Args:
            entity_embeddings: 模型输出的实体嵌入
            knowledge_embeddings: 知识图谱中的实体嵌入
            entity_pairs: 相关实体对列表 [(entity1_idx, entity2_idx, relation_type), ...]
        """
        loss = 0.0
        positive_pairs = []
        negative_pairs = []
        
        for i, j, relation in entity_pairs:
            # 正样本对
            emb_i = entity_embeddings[i]
            emb_j = entity_embeddings[j]
            kg_i = knowledge_embeddings[i]
            kg_j = knowledge_embeddings[j]
            
            # 计算相似度
            sim_emb = F.cosine_similarity(emb_i, emb_j, dim=-1)
            sim_kg = F.cosine_similarity(kg_i, kg_j, dim=-1)
            
            # 对比损失
            positive_loss = torch.abs(sim_emb - sim_kg)
            
            # 负采样
            # 这里简化处理，实际应用中需要更复杂的负采样策略
            neg_idx = torch.randint(0, len(entity_embeddings), (1,)).item()
            if neg_idx != i and neg_idx != j:
                neg_sim = F.cosine_similarity(emb_i, entity_embeddings[neg_idx], dim=-1)
                negative_loss = torch.max(torch.tensor(0.0), 0.5 - neg_sim)
                loss += positive_loss + 0.1 * negative_loss
        
        return loss / max(len(entity_pairs), 1)