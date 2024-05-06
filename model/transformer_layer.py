# PyTorch
import pandas as pd
import numpy as np
import sklearn
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

# Math
import math
from scipy import stats
import time
import os

# HuggingFace libraries 
from datasets import load_dataset, load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from model.embed_layer import *
from model.mh_attention import *
from model.encoder import *
from model.transformer import *

# ---------------------------------------
# 1. Class Dataset
class DatasetSNP(Dataset):
    def __init__(self, encodings, labels):
        super().__init__

        self.encodings = encodings
        self.labels = labels 

    def __getitem__(self, idx):
        return self.encodings[idx], self.labels[idx]

    def __len__(self):
        return len(self.encodings)

# ---------------------------------------
# Transform dataset to tensor
# Transform to tensor (list of tensor, each tensor is each sample)

def dataset_tensor(X_train, y_train):
    X_train = [torch.tensor(item) for item in X_train]
    y_train = [torch.tensor(item) for item in y_train]
    train_dataset = DatasetSNP(X_train, y_train)
    return train_dataset

# ---------------------------------------
# 2. Define Transformer model



# Example usage
# src_vocab_size = 30000
# target_vocab_size = 11
# embed_dim = 10  # Dimensionality of embeddings for each word/ nucleotide
# num_blocks = 6
# seq_len = 12

# # let 0 be sos token and 1 be eos token
# src = torch.tensor([[0, 2, 5, 6, 4, 3, 9, 5, 2, 9, 10, 1],
#                     [0, 2, 8, 7, 3, 4, 5, 6, 7, 2, 10, 1]])
# target = torch.tensor([[0, 1, 7, 4, 3, 5, 9, 2, 8, 10, 9, 1],
                    #    [0, 1, 5, 6, 2, 4, 7, 6, 2, 8, 10, 1]])

# print(src.shape, target.shape) # Shape(samples, seq_len)

# -----------------------------------------------------
# Testing Embedding block
# -----------------------------------------------------

def Test_Embedding_Positional(src, src_vocab_size, seq_len,embed_dim: int = 512):

    # Create embed model
    embedding_model = Embedding(src_vocab_size, embed_dim)
    # Create positionel embedding
    position_model = PositionalEncoding(embed_dim, seq_len)
    # # Embed DNA sequences
    embeddings = embedding_model(src)
    position_embeddings = position_model(embeddings)

    return position_embeddings

def Test_encoderblock(src,src_vocab_size, seq_len, embed_dim: int=512):
    encoder = Encoder(seq_len,
                      vocab_size=src_vocab_size,
                      embed_dim=embed_dim,
                      num_blocks=6,
                      expansion_factor=4,
                      heads=8,
                      dropout=0.2)
    out = encoder(src)
    return out

def Test_transformerblock(src, src_vocab_size, seq_len, embed_dim: int=512):
    model = TransformerSNP(src_vocab_size, 
                           seq_len,
                           embed_dim,
                           num_blocks=6,
                           expansion_factor=4,
                           heads=8,
                           dropout=0.2)
    out = model(src)
    return out
                 

# ---------------------------------------
# Training model
# Function to validate the model for one epoch
def validate_one_epoch(model, val_loader, loss_function):

    # arrays for tracking eval results
    avg_loss = 0.0
    arr_val_losses = []

    # evaluate the trained model
    model.eval()
    with torch.no_grad():
        # Iterate through the validation loader
        for i, (inputs, targets) in enumerate(val_loader):
            # inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            arr_val_losses.append(loss.item())
    
    # calculate average validation loss
    avg_loss = np.average(arr_val_losses)
    return avg_loss

def predict(model, val_loader):
    model.eval()
    predictions = None
    with torch.no_grad():
        # iterate through the validation loader
        for i, (inputs, targets) in enumerate(val_loader):
            # inputs, targets = inputs.to(device), targets.to(device)
            # inputs  = inputs.float()
            outputs = model(inputs)
            # concatenate the predictions
            predictions = torch.clone(outputs) if predictions is None else torch.cat((predictions, outputs))
    ret_output = predictions.detach().numpy()
    # convert predictions to numpy array based on device
    # if device == torch.device('cpu'):
    #     ret_output = predictions.detach().numpy()
    # else:
    #     ret_output = predictions.cpu().detach().numpy()
    
    return ret_output

def train_model(model, train_loader, val_loader, epochs):

    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.00002)
    loss = 0.0

    list_of_rows = []
    for epoch in range(epochs):
        # Training
        # start_time = time.time()
        model.train()
        for idx, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            # unsqueeze y to match shape of ouput
            # y = y.unsqueeze(1) # torch.Size([8]) --> torch.Size([8, 1])
            # print('shape of y', y.size())
            outputs = model(x) # torch.Size([8, 1])
            # print('shape of output', outputs.size())
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()


        # val_loss, spearman_val_corr, spearman_val_p = calc_metrics_regression(model, valid_loader)
        # test_loss, spearman_test_corr, spearman_test_p = calc_metrics_regression(model, test_loader)
        
        # logger.info(f'epoch = {epoch}, val_loss = {val_loss}, spearman_val_corr = {spearman_val_corr}, spearman_val_p = {spearman_val_p}, test_loss = {test_loss}, spearman_test_corr = {spearman_test_corr}, spearman_test_p = {spearman_test_p}')
        # list_of_rows.append({'epoch': epoch, 'val_loss': val_loss, 'spearman_val_corr': spearman_val_corr, 'spearman_val_p': spearman_val_p, 'test_loss': test_loss, 'spearman_test_corr': spearman_test_corr, 'spearman_test_p': spearman_test_p})
        
        torch.save(model.state_dict(), os.path.join('./', f"checkpoint_{epoch}.pt"))
    return model

    
