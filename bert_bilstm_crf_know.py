import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from torchcrf import CRF
from .knowledge_graph import TCMKnowledgeGraph
from .knowledge_fusion import KnowledgeFusionModule, KnowledgeAlignmentLoss

class BertBiLstmCrfKnow(BertPreTrainedModel):
    """BERT+BiLSTM+CRF+KNOW模型"""
    def __init__(self, config, num_labels, knowledge_dim=200):
        super().__init__(config)
        self.num_labels = num_labels
        
        # BERT编码器
        self.bert = BertModel(config)
        
        # 知识图谱
        self.knowledge_graph = TCMKnowledgeGraph(embedding_dim=knowledge_dim)
        
        # 知识融合模块
        self.knowledge_fusion = KnowledgeFusionModule(
            bert_dim=config.hidden_size,
            knowledge_dim=knowledge_dim
        )
        
        # BiLSTM层
        self.bilstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )
        
        # 分类层
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        
        # CRF层
        self.crf = CRF(num_labels, batch_first=True)
        
        # 知识对齐损失
        self.knowledge_loss = KnowledgeAlignmentLoss()
        
        # 损失权重
        self.alpha = 0.7  # CRF损失权重
        self.beta = 0.3   # 知识对齐损失权重
        
        # 初始化权重
        self.init_weights()
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        knowledge_entities=None
    ):
        # BERT编码
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]
        
        # 获取知识嵌入
        batch_size, seq_len = input_ids.shape
        knowledge_embeddings = []
        
        for i in range(batch_size):
            sentence_knowledge = []
            for j in range(seq_len):
                token_id = input_ids[i, j].item()
                # 这里需要tokenizer将id转换为文本
                # 简化处理：使用占位符
                if hasattr(self, 'tokenizer'):
                    token_text = self.tokenizer.decode([token_id])
                else:
                    token_text = f"token_{token_id}"
                
                # 获取知识嵌入
                if knowledge_entities is not None and i < len(knowledge_entities):
                    # 如果有预计算的知识实体
                    kg_emb = self.knowledge_graph.get_entity_embedding(
                        knowledge_entities[i][j] if j < len(knowledge_entities[i]) else ""
                    )
                else:
                    # 动态获取知识
                    kg_emb = self.knowledge_graph.get_token_knowledge(token_text)
                
                sentence_knowledge.append(kg_emb)
            
            knowledge_embeddings.append(torch.cat(sentence_knowledge, dim=0))
        
        knowledge_embeddings = torch.stack(knowledge_embeddings, dim=0)  # [B, L, knowledge_dim]
        
        # 知识融合
        fused_output, attention_weights = self.knowledge_fusion(
            sequence_output,
            knowledge_embeddings,
            attention_mask
        )
        
        # BiLSTM处理
        lstm_output, _ = self.bilstm(fused_output)  # [B, L, hidden_size]
        
        # 分类
        logits = self.classifier(lstm_output)  # [B, L, num_labels]
        
        # 计算损失
        loss = None
        if labels is not None:
            # CRF损失
            crf_loss = -self.crf(logits, labels, mask=attention_mask.bool())
            
            # 知识对齐损失
            if knowledge_entities is not None:
                # 提取实体嵌入
                entity_mask = labels != 0  # 假设0是'O'标签
                entity_embeddings = lstm_output[entity_mask]
                
                # 获取对应的知识嵌入
                kg_entity_embeddings = knowledge_embeddings[entity_mask]
                
                # 计算知识对齐损失（这里需要实体对关系，简化处理）
                know_loss = self.knowledge_loss(
                    entity_embeddings,
                    kg_entity_embeddings,
                    []  # 实际应用中需要提供实体对
                )
                
                # 总损失
                loss = self.alpha * crf_loss + self.beta * know_loss
            else:
                loss = crf_loss
        
        # 预测
        if not self.training:
            predictions = self.crf.decode(logits, mask=attention_mask.bool())
        else:
            predictions = None
        
        output = {
            'loss': loss,
            'logits': logits,
            'predictions': predictions,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
            'knowledge_attention': attention_weights
        }
        
        return output
    
    def set_tokenizer(self, tokenizer):
        """设置tokenizer用于知识获取"""
        self.tokenizer = tokenizer