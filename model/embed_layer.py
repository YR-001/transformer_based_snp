from math import sin, cos, sqrt, log
import torch
import torch.nn as nn

class Embedding(nn.Module):

    def __init__(self, vocab_size, embed_dim):
        """
        Embedding class to convert a word into embedding space (numerical representation)
        :param vocab_size: the vocabulary size
        :param embed_dim: the embedding dimension

        example: if we have 1000 vocabulary size and our embedding is 512,
        then the embedding layer will be 1000x512

        suppose we have a batch size of 64 and sequence of 15 words,
        then the output will be 64x15x512 ((samples, seq_length, embed_dim))
        """
        super(Embedding, self).__init__()
        self.embed_dim = embed_dim # Dimension of model
        self.vocab_size = vocab_size # Size of the vocabulary
        self.embed = nn.Embedding(vocab_size, embed_dim) # Pytorch layer that converts integer indices to dense embeddings

    def forward(self, x):
        """
        forward pass
        :param x: the word or sequence of words
        :return: the numerical representation of the input
        """

        # splitted_x = torch.tensor_split(x, [218, 778, 1323, 1871], dim=1)
        # x0 = splitted_x[0]
        # x1 = splitted_x[1]
        # x2 = splitted_x[2]
        # x3 = splitted_x[3]
        # x4 = splitted_x[4]

        # Normalizing the variance of the embeddings
        # output0 = self.embed(x0) * sqrt(self.embed_dim)
        # output1 = self.embed(x1) * sqrt(self.embed_dim)
        # output2 = self.embed(x2) * sqrt(self.embed_dim)
        # output3 = self.embed(x3) * sqrt(self.embed_dim)
        # output4 = self.embed(x4) * sqrt(self.embed_dim)

        # print(f"Embedding shape: {output.shape}") #shape (samples, seq_length, embed_dim)
        # output = torch.cat((output0, output1, output2, output3, output4), 1)
        # print('Output embedding', output.shape)

        output = self.embed(x) * sqrt(self.embed_dim)

        return output
    

class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, max_seq_len=100, dropout=0.1):
        """
        Positional Embedding or Positional Encoding
        The general idea here is to add positional encoding to the input embedding
        before feeding the input vectors to the first encoder/decoder
        The positional embedding must have the same embedding dimension as in the embedding vectors
        For the positional encoding we use sin and cos

        :param embed_dim: the size of the embedding, this must be the same as in embedding vector
        :param max_seq_len: the maximum sequence length (max sequence of words)
        :param dropout: the dropout probability
        """
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim # Dimensionality of model (also named d_model in some codes)
        self.max_seq_len = max_seq_len # Maximum sequence length
        self.dropout = nn.Dropout(dropout) # dropout layer to prevent overfitting

        # Creating a positional encoding matrix of shape (max_seq_len, d_model) filled with zeros
        positional_encoding = torch.zeros(max_seq_len, self.embed_dim)

        # Creating a tensor representing positions (0 to seq_len - 1)
        position = torch.arange(0, max_seq_len).unsqueeze(1) # # Transforming 'position' into a 2D tensor['seq_len, 1']
        
        # Creating the division term for the positional encoding formula
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(log(10000.0) / embed_dim))
        # div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices in pe
        positional_encoding[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to even indices in pe
        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        # Adding an extra dimension at the beginning of pe matrix for batch handling
        pe = positional_encoding.unsqueeze(0)

        # we use register_buffer to save the "pe" parameter to the state_dict
        # Buffer is a tensor not considered as a model parameter
        self.register_buffer('pe', pe)

    def pe_sin(self, position, i):
        return sin(position / (10000 ** (2 * i) / self.embed_dim))

    def pe_cos(self, position, i):
        return cos(position / (10000 ** (2 * i) / self.embed_dim))

    def forward(self, x):
        # print(x.shape)
        # print(self.pe[:, : x.size(1)].shape)

        # splitted_x = torch.tensor_split(x, [218, 778, 1323, 1871], dim=1)
        # x0 = splitted_x[0]
        # x1 = splitted_x[1]
        # x2 = splitted_x[2]
        # x3 = splitted_x[3]
        # x4 = splitted_x[4]

        # # Adding positional encoding to the input tensor X
        # x0 = x0 + self.pe[:, : x0.size(1)].requires_grad_(False)
        # x1 = x1 + self.pe[:, : x1.size(1)].requires_grad_(False) 
        # x2 = x2 + self.pe[:, : x2.size(1)].requires_grad_(False)
        # x3 = x3 + self.pe[:, : x3.size(1)].requires_grad_(False)
        # x4 = x4 + self.pe[:, : x4.size(1)].requires_grad_(False)

        # x = torch.cat((x0, x1, x2, x3, x4), 1)
        # print('Output Positional encoding', x.shape)
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)

        # Dropout for regularization
        return self.dropout(x) 