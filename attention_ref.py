"""PyTorch reference for scaled dot-product attention (materializes the full N x N matrix)."""
import math
import torch
 
 
def attention_ref(q, k, v, causal=False):
    # q, k, v: [B, H, N, D]
    scale = 1.0 / math.sqrt(q.shape[-1])
    s = torch.matmul(q, k.transpose(-1, -2)) * scale
    if causal:
        n = q.shape[-2]
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=q.device), diagonal=1)
        s = s.masked_fill(mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.matmul(p, v)
 