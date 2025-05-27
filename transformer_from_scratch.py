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

class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, embed_model_dim):
        super(PositionalEmbedding, self).__init__()
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

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        super(TransformerBlock, self).__init__()

        self.attention = MultiheadAttention(embed_dim, n_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor * embed_dim),
            nn.ReLU(),
            nn.Linear(expansion_factor * embed_dim, embed_dim)
        )

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, key, value, query):
        attention_out = self.attention(query, key, value) # 32*10*512
        attention_residual_out = attention_out + value # 32*10*512

        norm1_out = self.dropout1(self.norm1(attention_residual_out)) # 32*10*512

        feed_fwd_out = self.feed_forward(norm1_out) # 32*10*512

        feed_fwd_residual_out = feed_fwd_out + norm1_out # 32*10*512

        norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out)) # 32*10*512

        return norm2_out

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        super(DecoderBlock, self).__init__()

        self.attention = MultiheadAttention(embed_dim, n_heads=8)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.2)
        self.transformer_block = TransformerBlock(embed_dim, expansion_factor, n_heads)


    def forward(self, key, query, x,mask):

        #we need to pass mask mask only to fst attention
        attention = self.attention(x,x,x,mask=mask) #32x10x512
        value = self.dropout(self.norm(attention + x))

        out = self.transformer_block(key, query, value)


        return out


class TransformerDecoder(nn.Module):
    def __init__(self, target_vocab_size, embed_dim, seq_len, num_layers=2, expansion_factor=4, n_heads=8):
        super(TransformerDecoder, self).__init__()
  
        self.word_embedding = nn.Embedding(target_vocab_size, embed_dim)
        self.position_embedding = PositionalEmbedding(seq_len, embed_dim)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_dim, expansion_factor=4, n_heads=8)
                for _ in range(num_layers)
            ]

        )
        self.fc_out = nn.Linear(embed_dim, target_vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, enc_out, mask):



        x = self.word_embedding(x)  #32x10x512
        x = self.position_embedding(x) #32x10x512
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(enc_out, x, enc_out, mask)

        out = F.softmax(self.fc_out(x))

        return out