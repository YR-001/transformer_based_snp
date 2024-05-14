from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):

    def __init__(self, mask, d_model: int, heads: int, dropout: float):
        """
        Multi-Head Attention class
        :param embed_dim: the embedding dimension
        :param heads: the number of heads, default equals 8
        """
        super(MultiHeadAttention, self).__init__()

        self.mask = mask
        self.d_model = d_model # 512 by default
        self.heads = heads #8 by default
        
        # We ensure that the dimensions of the model is divisible by the number of heads
        assert d_model % heads == 0, 'd_model is not divisible by h'
        
        # d_k is the dimension of each attention head's key, query, and value vectors
        self.d_k = d_model // heads # d_k formula, # 512 / 8 = 64 by default
        
        # Defining the weight matrices
        self.w_q = nn.Linear(d_model, d_model, bias=False) # # the Query weight metrix
        self.w_k = nn.Linear(d_model, d_model, bias=False) # W_k
        self.w_v = nn.Linear(d_model, d_model, bias=False) # W_v
        # Define the weight matrice W0
        # fully connected layer: 8*64x512 or 512x512
        self.w_o = nn.Linear(d_model, d_model) # W_o
        
        self.dropout = nn.Dropout(dropout) # Dropout layer to avoid overfitting
    
    @staticmethod
    def attention(query, key, value, dropout: nn.Dropout, mask):# mask => When we want certain words to NOT interact with others, we "hide" them
        
        d_k = query.shape[-1] # The last dimension of query, key, and value
        
        # We calculate the Attention(Q,K,V) as in the formula in the image above 
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k) # @ = Matrix multiplication sign in PyTorch
        # print('Attention score at beggining', attention_scores) #batch_size, heads, q_len, k_len

        # Before applying the softmax, we apply the mask to hide some interactions between words
        if mask is not None: # If a mask IS defined...
            # print('Attention score shape', attention_scores.shape)
            attention_scores.masked_fill_(mask == 0, -1e9) # Replace each value where mask is equal to 0 by -1e9
        attention_scores = attention_scores.softmax(dim = -1) # Applying softmax
        if dropout is not None: # If a dropout IS defined...
            attention_scores = dropout(attention_scores) # We apply dropout to prevent overfitting
            
        return (attention_scores @ value), attention_scores # Multiply the output matrix by the V matrix, as in the formula

    def forward(self, x):

        # Input of size: batch_size x sequence length x embedding dims
        # batch_size, seq_len, d_model = x.shape
        # Projection into query, key, value: (batch, seq_len, d_model)
        query = self.w_q(x) # Q' matrix
        key = self.w_k(x) # K' matrix
        value = self.w_v(x) # V' matrix
        
        # Splitting results into smaller matrices for the different heads
        # Splitting embeddings (third dimension) into h parts
        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.heads, self.d_k).transpose(1,2) # Transpose => bring the head to the second dimension
        key = key.view(key.shape[0], key.shape[1], self.heads, self.d_k).transpose(1,2) # Transpose => bring the head to the second dimension
        value = value.view(value.shape[0], value.shape[1], self.heads, self.d_k).transpose(1,2) # Transpose => bring the head to the second dimension

        # Obtaining the output and the attention scores
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, self.dropout, mask=self.mask)
        
        # Obtaining the H matrix
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.heads * self.d_k)
        
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        output = self.w_o(x)  # H x W0 = (32x10x8x64) x (32x8x64x512) = (32x10x512)
        
        return output
    
