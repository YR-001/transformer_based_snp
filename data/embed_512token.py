import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from math import sin, cos, sqrt, log

# Load file
X_train_csv, y_train_csv = pd.read_csv('x_train_onehot.csv'), pd.read_csv('y_train_onehot.csv')
X_train, y_train = X_train_csv.iloc[:,1:].to_numpy(), y_train_csv.iloc[:,1].to_numpy()
# print(X_train[:3])
sequence_train = [''.join(seq) for seq in X_train]
# print(join_data)

dna = ["ACGTCGTACGTACGTACGTATATGAGCTGCTACGATCA", "ATCGATCGATTGCAGCTAGCTAGCGGTACGTACGTATATG"]
# Define transformer parameters
max_sequence_length = 4
embedding_dim = 128
num_segments = 20  # Adjust this value based on the sequence length and max_sequence_length

# Step 1: Split DNA sequence into non-overlapping k-mers
def seqs2kmer(seqs, kmer):
    seq_kmers = []
    for s in seqs:
        sequence_kmers = []  
        # Iterate over the sequence to generate k-mers
        for i in range(0, len(s) - kmer + 1):
            kmer_sequence = s[i:i+kmer]
            sequence_kmers.append(kmer_sequence)
            
        seq_kmers.append(sequence_kmers)
    return seq_kmers

seq = seqs2kmer(dna, 3)
print(seq)

def tokenize_kmers(kmers, max_sequence_length=max_sequence_length):
    tokens_list = []
    for kmer in kmers:
        tokens = ["[CLS]"] + kmers[:max_sequence_length-2] + ["[SEP]"]
    return tokens

tokens = tokenize_kmers(seq, max_sequence_length)
print('tokenize_kmers', tokens)
exit(1)

# Step 2: Split DNA sequence into segments using sliding window
# Add special token [CLS] and [SEP] -> [CLS] seq [SEP]
def split_into_segments(sequences, segment_length):
  """
  This function splits a list of DNA sequences into segments of a specified length.

  Args:
      sequences: A list of DNA sequences as strings.
      segment_length: The desired length of each segment as an integer.

  Returns:
      A list containing sub-lists of DNA segments for each original sequence.
  """

  segmented_sequences = []
  for sequence in sequences:
    segments = []
    for i in range(0, len(sequence), segment_length):
      segment = sequence[i:i+segment_length]
      segment_with_tokens = f"[CLS]{segment}[SEP]"
      segments.append(segment_with_tokens)
    segmented_sequences.append(segments)
  return segmented_sequences

segments = split_into_segments(dna, max_sequence_length)
print(segments)

# Step 2: 


