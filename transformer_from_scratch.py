#importing necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import math,copy,re
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
    def forward(self, x):
        out = self.embed(x)
        return out

class PositioanlEmbedding(nn.Module):
    def __init__(self, max_seq_len, embed_model_dim):
        super(PositioanlEmbedding, self).__init__()
        self.embed_dim = embed_model_dim
        
        pe = torch.zeros(max_seq_len, self.embed_dim)

        for pos in range(max_seq_len):
            for i in range(0, self.embed_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.embed_dim)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        def forward(self, x):
            x = x * math.sqrt(self.embed_dim)
            seq_len = x.size(1)
            x = x + torch.autograd.Variable(self.pe[:, :seq_len], requires_grad=False)
            return x
        
class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim = 512, n_heads = 8):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.single_head_dim = int(self.embed_dim / self.n_heads)

        self.query_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.key_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False) 
        self.value_matrix = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)  

        self.out = nn.Linear(self.n_heads * self.single_head_dim, self.embed_dim)

        def forward(self, query, key, value, mask=None): 
            batch_size = key.size(0)
            seq_len = key.size(1)

            seq_len_query = query.size(1)

            #32*10*512 -> 32*10*8*512

            key = key.view(batch_size, seq_len, self.n_heads, self.single_head_dim)

            query = query.view(batch_size, seq_len, self.n_heads, self.single_head_dim)

            value = value.view(batch_size, seq_len, self.n_heads, self.single_head_dim)

            k = self.key_matrix(key)
            q = self.query_matrix(query)
            v = self.value_matrix(value)

            #32*10*8*512 -> 32*8*10*512

            k = k.transpose(1, 2)
            q = q.transpose(1, 2)       
            v = v.transpose(1, 2)

            k_adjusted = k.trasnpose(-1, -2)
            product = torch.matmul(q, k_adjusted)

            if mask is not None:
                product = product.masked_fill(mask == 0, float('-1e9'))
            
            product = product / math.sqrt(self.single_head_dim)

            scores = F.softmax(product, dim=-1)

            scores = torch.matmul(scores, v)

            concat = scores.transpose(1, 2).contiguous().view(batch_size, seq_len_query, self.n_heads * self.single_head_dim)

            #32*8*10*64 -> 32*10*8*64 ->32*10*512

            out = self.out(concat) # 32*10*512

            return out


