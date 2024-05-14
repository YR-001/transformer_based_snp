import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from gensim.models import Word2Vec
from math import sin, cos, sqrt, log

from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Sequence, Digits, Whitespace


# ---------------------------------------
# 1. Function for using indexed to embed
# ---------------------------------------
def token_embed(seqs):
    token_to_index = {}  # Initialize an empty dictionary to store token-to-index mappings
    index = 0  # Initialize an index counter
    indexed_kmer_sequences = []  # Initialize an empty list to store indexed k-mer sequences
    # Iterate over each k-mer sequence
    for k_mers in seqs:
        indexed_k_mers = []  # Initialize an empty list to store indexed k-mers for current sequence
        # Assign index to each token
        for k_mer in k_mers:
            if k_mer not in token_to_index:
                token_to_index[k_mer] = index
                index += 1
            indexed_k_mers.append(token_to_index[k_mer])
        indexed_kmer_sequences.append(indexed_k_mers)
    return indexed_kmer_sequences

# indexed_kmer_sequences, token_to_index = token_embed(seqs)

# i.e. # print(mapped_sequences) # [[23, 34, 21, ...], [...], ...]
# print("K-mer Sequences:", indexed_kmer_sequences)

# i.e. dictionary of 64: {'CGA': 0, 'GAC': 1, 'ACA': 2, 'CAG': 3,...}
# print("Token to Index Mapping:", token_to_index)


# ---------------------------------------
# 2. Function for using Word2vec to embed
# ---------------------------------------
def Word2vec_embed(seqs):
    
    # Train Word2Vec model
    word2vec_model = Word2Vec(sentences=seqs, vector_size=100, window=5, min_count=1)
    # word2vec_model.build_vocab(gen_word)
    word2vec_model.train(seqs, total_examples=len(seqs), epochs=1)
    
    # Get embedding for each amino acid
    # embeddings = {word: word2vec_model.wv[word] for word in word2vec_model.wv.index_to_key} # write in shor way
    vocab = word2vec_model.wv.index_to_key  # assuming index_to_key provides vocabulary words
                                            # vocab: ['CAC', 'GGT', 'AGT', ..., 'TTT', 'AAC']
    embeddings = dict.fromkeys(vocab, None)  # create dictionary with all words as keys and None as default values
                                             # {'CAC': None, 'GGT': None, ..., 'TTT': None, 'AAC': None}
    # Fill the dictionary with actual embeddings
    for word in vocab:
        embeddings[word] = word2vec_model.wv[word]

    # Initialize an empty list to store the word embeddings
    sequence_embeddings = []
    # Loop through each word in the sentence
    for sequence in seqs:
        current_seq = []
        for word in sequence:
            current_seq.append(embeddings[word])
        sequence_embeddings.append(current_seq)

    return sequence_embeddings

# embeddings, sequence_embeddings = Word2vec_embed(seq_overlap)

# i.e. dictionary of 64, each array is 100: {'CGA': [1.2 , -7.2, ..], 'GAC': [..], 'ACA': [..], ..}
# print("Token to Word2vec Mapping:", embeddings)

# Output the embedding for a specific token (e.g., 'AGT')
# print("Embedding for k-mer 'AGT':", embeddings['AGT'].shape)
# print(sequence_embeddings)

# ---------------------------------------
# 3. Function for using BPE to embed
# input seq using seq (not in k_mer)
# ---------------------------------------
def prepare_tokenizer_trainer(vocab_size):
    tokenizer = Tokenizer(BPE())
    trainer = BpeTrainer(special_tokens = ['[PAD]'], vocab_size=vocab_size)
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer, trainer

def train_tokenizer(iterator, vocab_size):
    """
    Takes the files and trains the tokenizer.
    """
    tokenizer, trainer = prepare_tokenizer_trainer(vocab_size)
    tokenizer.train_from_iterator(iterator, trainer) # training the tokenzier
    return tokenizer    

def batch_iterator(dataset):
    batch_size = 500
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]

def BPE_embed(seqs, chr_index, vocab_size=2048):
    
    tokenizer_path = os.path.join('./', "tokenizer_BPE" + str(chr_index) +".json")
    if not os.path.exists(tokenizer_path):
        print('Tokenizer BPE {} is not exited, create BPE tokenize'.format(chr_index))
        # Create (training) new tokenizer from the dataset
        # print(batch_iterator(seqs))
        tokenizer = train_tokenizer(batch_iterator(seqs), vocab_size)
        # tokenizer.enable_padding(length=max_length) # padding to max_len

        # Saving trained tokenizer to the file path
        tokenizer.save(tokenizer_path) # dictionary of {"G": 1, "A": 2, ....}
        # print(tokenizer.get_vocab_size()) # get vocab_size
    else:
        print('Tokenizer BPE {} is already created, just load it'.format(chr_index))
        tokenizer = Tokenizer.from_file(tokenizer_path) # If the tokenizer already exist, we load it
    
    return tokenizer # Returns the loaded tokenizer or the trained tokenizer

# Check and choose the max_len of sequence in config (i.e. if reasult =180 --> max_len = 200) 
def choose_max_length(X, tokenizer):
    max_len_src = 0

    for x in X:
        src_ids = tokenizer.encode(x).ids
        max_len_src = max(max_len_src, len(src_ids))
    return print(f'Max length of sample in SNP dataset: {max_len_src}')


# Tokenizing data
# by assign idices to each token
def encode(X, tokenizer, max_length):
    result = []
    tokenizer.enable_padding(length=max_length)
    # loop each sample in data(i.e. X_train ['SK GE EL FT G.', '...',...])
    for x in X:

        # assign idices to each token[13, 29, 5, 52, 18] + padding
        ids = tokenizer.encode(x).ids 

        if len(ids) > max_length:
            ids = ids[:max_length] # trunct sequences if len(sample)>max_len
        # else:
        #     tokenizer.enable_padding(length=max_length)

        result.append(ids)

    return result
# seqs = encode(seqs)
# return seqs



# X = BPE_embed(join_data, vocab_size, results_path, max_length)
# print(X[0])


# ---------------------------------------
# Function to embed sequence
# ---------------------------------------