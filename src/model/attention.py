import math
import torch
import torch.nn as nn

from src.utils.helpers import clones


def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


def attention(query, key, value, mask=None, dropout=None):
    """Compute Scaled Dot Product Attention."""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # Nulling values outside the mask
        scores = scores.masked_fill(mask == 0, -1e9)

    # Big negative numbers are 0 after sofmax, preventing leftward information flow
    p_attn = scores.softmax(dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size and number of heads"""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equal d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """Corresponds to the MHA schema in the paper"""
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1. Do all the linear projections in batch from d_model => h * d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2. Apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # "Concat" (for the MultiHead part) using a view and apply a final linear
        # Contiguous arranges the tensor in a single contiguous physical memory space
        # -> Transpose or view can scramble physical memory adresses for tensor values
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        del query
        del key
        del value
        return self.linears[-1](x)
