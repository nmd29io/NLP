import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def ScaledDotProductAttention(query, key, value, mask=None,dropout=None):
    """    
    query, key, value: (batch_size x heads x seq_len x d_k )
    mask: ?
    dropout: nn.Dropout module from whichever module using this function
    returns attention output (batch_size x heads x seq_len x d_k)
      and weights
    """
    d_k = query.size(-1)
    # scores: batch_size x heads x seq_len x seq_len
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e5) # big negative

    # softmax to prob. weights
    weights = torch.softmax(scores, dim = -1)
    if dropout is not None:
        weights = dropout(weights)
    output = torch.matmul(weights, value), weights
    return output, weights
