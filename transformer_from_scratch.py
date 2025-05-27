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
        