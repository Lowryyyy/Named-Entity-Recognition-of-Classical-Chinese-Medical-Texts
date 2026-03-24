import torch
import torch.nn as nn
from typing import List, Optional


class CRF(nn.Module):
    def __init__(self, num_tags: int, batch_first: bool = False):
        super(CRF, self).__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.transitions, -0.1, 0.1)
        nn.init.normal_(self.start_transitions, -0.1, 0.1)
        nn.init.normal_(self.end_transitions, -0.1, 0.1)
    
    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)
        
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)
        
        log_likelihood = self._compute_log_likelihood(emissions, tags, mask)
        return -log_likelihood
    
    def _compute_log_likelihood(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        seq_length, batch_size = tags.shape
        
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]
        
        for i in range(1, seq_length):
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]
        
        seq_ends = mask.long().sum(dim=0) - 1
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        score += self.end_transitions[last_tags]
        
        forward_score = self._compute_forward(emissions, mask)
        log_likelihood = score - forward_score
        
        return log_likelihood.sum()
    
    def _compute_forward(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        seq_length, batch_size, _ = emissions.shape
        
        alpha = self.start_transitions + emissions[0]
        
        for i in range(1, seq_length):
            emit_score = emissions[i].unsqueeze(1)
            trans_score = self.transitions.unsqueeze(0)
            next_tag_var = alpha.unsqueeze(2) + trans_score + emit_score
            
            log_sum_exp = torch.logsumexp(next_tag_var, dim=1)
            alpha = log_sum_exp * mask[i].unsqueeze(1) + alpha * (1 - mask[i]).unsqueeze(1)
        
        alpha = alpha + self.end_transitions
        return torch.logsumexp(alpha, dim=1)
    
    def decode(self, emissions: torch.Tensor, mask: Optional[torch.Tensor] = None) -> List[List[int]]:
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)
        
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)
        
        return self._viterbi_decode(emissions, mask)
    
    def _viterbi_decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        seq_length, batch_size, _ = emissions.shape
        
        viterbi = self.start_transitions + emissions[0]
        backpointers = torch.zeros_like(viterbi, dtype=torch.long)
        backpointers_list = [backpointers]
        
        for i in range(1, seq_length):
            viterbi_t = viterbi.unsqueeze(2) + self.transitions.unsqueeze(0)
            viterbi_t, backpointers_t = torch.max(viterbi_t, dim=1)
            viterbi_t = viterbi_t + emissions[i]
            viterbi = viterbi_t * mask[i].unsqueeze(1) + viterbi * (1 - mask[i]).unsqueeze(1)
            backpointers_list.append(backpointers_t)
        
        viterbi = viterbi + self.end_transitions
        best_tags_list = []
        
        for b in range(batch_size):
            seq_end = mask[:, b].long().sum().item() - 1
            _, best_last_tag = torch.max(viterbi[b], dim=0)
            best_last_tag = best_last_tag.item()
            best_tags = [best_last_tag]
            
            for t in range(seq_end, 0, -1):
                best_last_tag = backpointers_list[t][b][best_last_tag].item()
                best_tags.append(best_last_tag)
            
            best_tags.reverse()
            best_tags_list.append(best_tags)
        
        return best_tags_list
