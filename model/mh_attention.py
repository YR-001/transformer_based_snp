from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim=512, heads=8):
        """
        Multi-Head Attention class
        :param embed_dim: the embedding dimension
        :param heads: the number of heads, default equals 8
        """
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim  # 512 by default
        self.heads = heads  # 8 heads by default

        # note: The embedding dimension must be divided by the number of heads
        # We ensure that the dimensions of the model is divisible by the number of heads
        assert embed_dim % heads == 0, 'Embedding dimension must be divisible by the number of heads'

        # d_k is the dimension of each attention head's key, query, and value vectors
        self.d_k = embed_dim // heads # 512 / 8 = 64 by default

        
        # Define the weight matrices for each head: query, value, key: (64x64)
        self.w_query = nn.Linear(self.d_k, self.d_k, bias=False)  # the Query weight metrix
        self.w_value = nn.Linear(self.d_k, self.d_k, bias=False)  # the Value weight metrix
        self.w_key = nn.Linear(self.d_k, self.d_k, bias=False)  # the Key weight metrix

        # Define the weight matrice W0
        # fully connected layer: 8*64x512 or 512x512
        self.w_o = nn.Linear(self.d_k * self.heads, embed_dim)

    def forward(self, key, query, value, mask=None):

        # Input of size: batch_size x sequence length x embedding dims
        batch_size = key.size(0)
        k_len, q_len, v_len = key.size(1), query.size(1), value.size(1)

        # Splitting embeddings (third dimension) into different heads
        # reshape from (batch_size x seq_len x embed_size) -> (batch_size x seq_len x heads x d_k)
        # e.g: from (32x10x512) -> (32x10x8x64)
        key = key.reshape(batch_size, k_len, self.heads, self.d_k)
        query = key.reshape(batch_size, q_len, self.heads, self.d_k)
        value = key.reshape(batch_size, v_len, self.heads, self.d_k)

        key = self.w_key(key)  # (32x10x8x64)
        query = self.w_key(query)  # (32x10x8x64)
        value = self.w_key(value)  # (32x10x8x64)

        ############### query x key ###############

        # query shape: batch_size x q_len, heads, d_k, e.g: (32x10x8x64) -> (seq x embed_dim)
        # key shape: batch_size x k_len, heads, d_k, e.g: (32x10x8x64) 
        # product shape should be: batch_size, heads, q_len, k_len, e.g: (32x8x10x10) -> (seq xseq)
        product = torch.einsum("bqhd,bkhd->bhqk", [query, key])

        # if mask (in decoder)
        if mask is not None:
            product = product.masked_fill(mask == 0, float("-1e20")) # Replace each value where mask is equal to 0 by -1e20

        product = product / sqrt(self.heads)

        attention_scores = F.softmax(product, dim=-1)

        ############### scores x value ###############

        # scores shape: batch_size, heads, q_len, k_len, e.g: (32x8x10x10) -> (seq x seq)
        # value shape: batch_size, v_len, heads, d_k, e.g: (32x10x8x64) -> (seq x embed_dim)
        # output (H): batch_size, q_len, heads * d_k, e.g: (32x10x8x64) -> (seq x embed_dim)
        
        output = torch.einsum("nhql,nlhd->nqhd", [attention_scores, value]).reshape(
            batch_size, q_len, self.heads * self.d_k
        )

        output = self.w_o(output)  # H x W0 = (32x10x8x64) x (32x8x64x512) = (32x10x512)
        
        return output