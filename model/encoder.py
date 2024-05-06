import torch.nn as nn
from model.utils import replicate
from model.mh_attention import MultiHeadAttention
from model.embed_layer import Embedding, PositionalEncoding


class TransformerBlock(nn.Module):

    def __init__(self,
                 embed_dim=512,
                 heads=8,
                 expansion_factor=4,
                 dropout=0.2
                 ):
        """
        The Transformer Block used in the encoder and decoder as well

        :param embed_dim: the embedding dimension
        :param heads: the number of heads
        :param expansion_factor: the factor that determines the output dimension of the feed forward layer
        :param dropout: probability dropout (between 0 and 1)
        """
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadAttention(embed_dim, heads)  # the multi-head attention
        self.norm = nn.LayerNorm(embed_dim)  # the normalization layer

        # The fully connected feed-forward layer, 
        # apply two linear transformations with a ReLU activation in between
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor * embed_dim),  # e.g: 512x(4*512) -> (512, 2048)
            nn.ReLU(),  # ReLU activation function
            nn.Linear(embed_dim * expansion_factor, embed_dim),  # e.g: 4*512)x512 -> (2048, 512)
        )

        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, key, query, value, mask=None):
        #################### Multi-Head Attention ####################
        # first, pass the key, query and value through the multi head attention layer
        attention_out = self.attention(key, query, value, mask)  # e.g.: 32x10x512

        ##################### Add & Norm Layer ######################
        # then add the residual connection
        # by adding input to output of attention layer
        attention_out = attention_out + value  # e.g.: 32x10x512

        # after that we normalize and use dropout
        attention_norm = self.dropout(self.norm(attention_out))  # e.g.: 32x10x512
        # print(attention_norm.shape)

        #################### Feed-Forwar Network ####################
        fc_out = self.feed_forward(attention_norm)  # e.g.:32x10x512 -> #32x10x2048 -> 32x10x512

        ##################### Add & Norm Layer ######################
        # Residual connection
        fc_out = fc_out + attention_norm  # e.g.: 32x10x512

        # dropout + normalization
        fc_norm = self.dropout(self.norm(fc_out))  # e.g.: 32x10x512

        return fc_norm


class Encoder(nn.Module):

    def __init__(self,
                 seq_len,
                 vocab_size,
                 embed_dim,
                 num_blocks,
                 expansion_factor,
                 heads,
                 dropout
                 ):
        """
        The Encoder part of the Transformer architecture
        it is a set of stacked encoders on top of each others, in the paper they used stack of 6 encoders

        :param seq_len: the length of the sequence, in other words, the length of the words
        :param vocab_size: the total size of the vocabulary
        :param embed_dim: the embedding dimension
        :param num_blocks: the number of blocks (encoders), 6 by default
        :param expansion_factor: the factor that determines the output dimension of the feed forward layer in each encoder
        :param heads: the number of heads in each encoder
        :param dropout: probability dropout (between 0 and 1)
        """
        super(Encoder, self).__init__()
        # define the embedding: (vocabulary size x embedding dimension)
        self.embedding = Embedding(vocab_size, embed_dim)
        # define the positional encoding: (embedding dimension x sequence length)
        self.positional_encoder = PositionalEncoding(embed_dim, seq_len)

        # define the set of blocks
        # so we will have 'num_blocks' stacked on top of each other
        self.blocks = replicate(TransformerBlock(embed_dim, heads, expansion_factor, dropout), num_blocks)

    def forward(self, x):
        out = self.positional_encoder(self.embedding(x))
        for block in self.blocks:
            out = block(out, out, out)
        
        # print(out.shape) # output shape: batch_size x seq_len x embed_size, e.g.: 32x12x512
        return out