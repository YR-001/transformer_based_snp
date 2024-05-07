# https://github.com/idotan286/BiologicalTokenizers/blob/main/train_tokenizer_bert.py#L110

import torch.nn as nn
from transformers import BertModel, BertConfig, BertTokenizerFast
from transformers import OpenAIGPTConfig, OpenAIGPTModel
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.trainers import BpeTrainer, WordLevelTrainer, \
                                WordPieceTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Sequence, Digits, Whitespace
from torch.optim import Adam
import sys
import os
import time
from transformers import PreTrainedTokenizerFast
import pickle
import subprocess as sp
import os
import logging
import random
from sklearn.metrics import matthews_corrcoef, accuracy_score
import numpy as np
import argparse
from scipy import stats

logger=logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

TOKEZNIER_BPE = "BPE"
TOKEZNIER_WPC = "WPC"
TOKEZNIER_UNI = "UNI"
TOKEZNIER_WORDS = "WORDS"
TOKEZNIER_PAIRS = "PAIRS"

UNK_TOKEN = "<UNK>"  # token for unknown words
SPL_TOKENS = [UNK_TOKEN]  # special tokens

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels 

    def __getitem__(self, idx):
        return self.encodings[idx], self.labels[idx]

    def __len__(self):
        return len(self.encodings)

# ----------------------------------
# Function for tokenizer
# ----------------------------------
def prepare_tokenizer_trainer(alg, voc_size):
    """
    Prepares the tokenizer and trainer with unknown & special tokens.
    """
    if alg == TOKEZNIER_WORDS:
        tokenizer = Tokenizer(WordLevel(unk_token = UNK_TOKEN))
        trainer = WordLevelTrainer(special_tokens = SPL_TOKENS)
    elif alg == TOKEZNIER_PAIRS:
        tokenizer = Tokenizer(WordLevel(unk_token = UNK_TOKEN))
        trainer = WordLevelTrainer(special_tokens = SPL_TOKENS)
    elif alg == TOKEZNIER_BPE:
        tokenizer = Tokenizer(BPE(unk_token = UNK_TOKEN))
        trainer = BpeTrainer(special_tokens = SPL_TOKENS, vocab_size=voc_size)
    elif alg == TOKEZNIER_UNI:
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(unk_token= UNK_TOKEN, special_tokens = SPL_TOKENS, vocab_size=voc_size)
    elif alg == TOKEZNIER_WPC:
        tokenizer = Tokenizer(WordPiece(unk_token = UNK_TOKEN, max_input_chars_per_word=10000))
        trainer = WordPieceTrainer(special_tokens = SPL_TOKENS, vocab_size=voc_size)

    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer, trainer

def train_tokenizer(iterator, alg, vocab_size):
    """
    Takes the files and trains the tokenizer.
    """
    tokenizer, trainer = prepare_tokenizer_trainer(alg, vocab_size)
    tokenizer.train_from_iterator(iterator, trainer) # training the tokenzier
    return tokenizer    

def batch_iterator(dataset):
    batch_size = 500
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]
    
    
def train_biological_tokenizer(tokenizer_type, vocab_size, results_path, max_length):
    """
    Reads the data from folder, trains the tokenizer, encode the sequences and returns list of data for BERT training
    """
    df_train = pd.read_csv('./data_mini/train.csv')
    df_valid = pd.read_csv('./data_mini/valid.csv')
    df_test = pd.read_csv('./data_mini/test.csv')

    # Get array y
    logger.info(f'starting a REGRESSION task!')
    y_train = df_train['label'].astype(float).tolist() # list of labels [1.32, 1.42, ...]
    y_valid = df_valid['label'].astype(float).tolist()
    y_test = df_test['label'].astype(float).tolist()
        
    num_of_classes = 1 # For Task regression

    # Get array X
    X_train = df_train['seq'].astype(str).tolist() #list of all samples ['SKGEELFTG', '...',...]
    X_valid = df_valid['seq'].astype(str).tolist()
    X_test = df_test['seq'].astype(str).tolist()

    if 'WORDS' == tokenizer_type:

        
        X_train = [' '.join([*aminos]) for aminos in X_train] #['S K G E E L F T G', '...',...]
                                                              # joins all the elements into a single string using the separator specified before the dot.
                                                              # asterisk (*) before 'aminos' :the unpacking operator -> the iterable (aminos) into individual elements
        print(X_train[:3])
        X_valid = [' '.join([*aminos]) for aminos in X_valid]
        X_test = [' '.join([*aminos]) for aminos in X_test]

        
    elif 'PAIRS' == tokenizer_type:
        def create_pairs(sequences):
            results = []
            for amino in sequences:
                amino_spaces = [*amino]
                if len(amino_spaces[::2]) == len(amino_spaces[1::2]):
                    pairs = [i+j for i,j in zip(amino_spaces[::2], amino_spaces[1::2])]
                elif len(amino_spaces[::2]) < len(amino_spaces[1::2]):
                    lst = amino_spaces[::2].copy()
                    lst.append('')
                    pairs = [i+j for i,j in zip(lst, amino_spaces[1::2])] #add an element to the first list
                else:
                    lst = amino_spaces[1::2].copy()
                    lst.append('')
                    pairs = [i+j for i,j in zip(amino_spaces[::2], lst)] #add an element to the second list
                results.append(' '.join(pairs))
            return results.copy()
        X_train = create_pairs(X_train) #['SK GE EL FT G.', '...',...]
        X_valid = create_pairs(X_valid)
        X_test = create_pairs(X_test)

    # Create (training) new tokenizer from the dataset
    logger.info(f'starting to train {tokenizer_type} tokenizer...')
    tokenizer = train_tokenizer(batch_iterator(X_train), tokenizer_type, vocab_size)
    tokenizer.enable_padding(length=max_length) #padding to max_len

    # Saving trained tokenizer to the file path
    logger.info(f'saving tokenizer to {results_path}...')
    tokenizer.save(os.path.join(results_path, "tokenizer.json")) # dictionary of {"G": 1, "L": 2, ....}
    print(tokenizer.get_vocab_size()) # get vocab_size

    # Tokenizing data
    # by assign idices to each token
    def encode(X):
        result = []
        for x in X: #loop each sample in data(i.e. X_train ['SK GE EL FT G.', '...',...])
            ids = tokenizer.encode(x).ids #assign idices to each token[13, 29, 5, 52, 18]  + padding
            if len(ids) > max_length:
                ids = ids[:max_length] # trunct sequences if len(sample)>max_len
            result.append(ids)
        return result
    X_train_ids = encode(X_train)
    X_valid_ids = encode(X_valid)
    X_test_ids = encode(X_test)

    # Transform to tensor (list of tensor, each tensor is each sample)
    X_train_ids = [torch.tensor(item) for item in X_train_ids]
    y_train = [torch.tensor(item) for item in y_train]
    logger.info(f'loaded train data to device')
    train_dataset = Dataset(X_train_ids, y_train)

    X_valid_ids = [torch.tensor(item) for item in X_valid_ids]
    y_valid = [torch.tensor(item) for item in y_valid]
    logger.info(f'loaded valid data to device')
    valid_dataset = Dataset(X_valid_ids, y_valid)

    X_test_ids = [torch.tensor(item) for item in X_test_ids]
    y_test = [torch.tensor(item) for item in y_test]
    logger.info(f'loaded test data to device')
    test_dataset = Dataset(X_test_ids, y_test)
    
    return num_of_classes, train_dataset, valid_dataset, test_dataset

# -----------------------------------------
# Create Transformer model
# -----------------------------------------   
class BioBERTModel(nn.Module):
    def __init__(self, hidden_size, num_layers, num_attention_heads, num_classes):
        super(BioBERTModel, self).__init__()
        configuration = BertConfig(hidden_size=hidden_size, num_hidden_layers=num_layers, num_attention_heads=num_attention_heads)
        self.transformer = BertModel(configuration)
        
        # additional layers for the classification / regression task
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes) #regression task: num_classes=1
        )
    
    def forward(self, ids, mask=None, token_type_ids=None):
        sequence_output, pooled_output = self.transformer(
           ids, 
           attention_mask=mask,
           token_type_ids=token_type_ids,
           return_dict=False
        )
        # print(sequence_output.size()) #torch.Size([8, 300, 128]) (batch, max_len, hidden_size)

        sequence_output = torch.mean(sequence_output, dim=1) #torch.Size([batch, hidden_size])
        result = self.head(sequence_output) #torch.Size([batch, 1]) 

        return result

# -----------------------------------------
# Define training loop
# -----------------------------------------
def train_model(model, train_generator, valid_generator, test_generator, epochs, print_training_logs, results_path):

    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.00002)

    def calc_metrics_regression(model, generator):
        loss = 0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for x,y in generator:
                outputs = model(x)  
                outputs = outputs.to(torch.float)
                # print('shape of output', outputs.size()) #torch.Size([1, 1])
                y_pred.append(outputs[0].item())
                y = y.to(torch.float).unsqueeze(1) #torch.Size([1]) --> torch.Size([1, 1])
                # print('shape of y', y.size())
                loss += loss_fn(outputs, y).item()
                y_true.append(y.item())
            loss = loss / len(generator)
            spearman = stats.spearmanr(y_pred, y_true)
        return loss, spearman[0], spearman[1]
    
    list_of_rows = []
    for epoch in range(1, epochs + 1):
        logger.info(f'----- starting epoch = {epoch} -----')
        epoch_loss = 0.0
        running_loss = 0.0
        # Training
        start_time = time.time()
        model.train()
        for idx, (x, y) in enumerate(train_generator):
            optimizer.zero_grad()
            # unsqueeze y to match shape of ouput
            y = y.unsqueeze(1) # torch.Size([8]) --> torch.Size([8, 1])
            # print('shape of y', y.size())
            outputs = model(x) # torch.Size([8, 1])
            # print('shape of output', outputs.size())
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            if idx % print_training_logs == print_training_logs - 1:
                end_time = time.time()
                logger.info('[%d, %5d] time: %.3f loss: %.3f' %
                      (epoch, idx + 1, end_time - start_time, running_loss / print_training_logs))
                running_loss = 0.0
                start_time = time.time()

        model.eval()

        val_loss, spearman_val_corr, spearman_val_p = calc_metrics_regression(model, valid_generator)
        test_loss, spearman_test_corr, spearman_test_p = calc_metrics_regression(model, test_generator)
        
        logger.info(f'epoch = {epoch}, val_loss = {val_loss}, spearman_val_corr = {spearman_val_corr}, spearman_val_p = {spearman_val_p}, test_loss = {test_loss}, spearman_test_corr = {spearman_test_corr}, spearman_test_p = {spearman_test_p}')
        list_of_rows.append({'epoch': epoch, 'val_loss': val_loss, 'spearman_val_corr': spearman_val_corr, 'spearman_val_p': spearman_val_p, 'test_loss': test_loss, 'spearman_test_corr': spearman_test_corr, 'spearman_test_p': spearman_test_p})
        
        torch.save(model.state_dict(), os.path.join(results_path, f"checkpoint_{epoch}.pt"))
    
    df_loss = pd.DataFrame(list_of_rows)
    df_loss.to_csv(os.path.join(results_path, f"results.csv"), index=False)

# -----------------------------------------
# Config
# -----------------------------------------
tokenizer_type = TOKEZNIER_BPE
vocab_size = 21 # from tokenizer.get_vocab_size()
results_path = './'
max_length = 300
hidden_size = 128
layers_num = 2
attention_heads_num = 2
epochs = 2
print_training_loss = 100
# -----------------------------------------
# Run model
# -----------------------------------------
num_classes, train_dataset, valid_dataset, test_dataset =train_biological_tokenizer(tokenizer_type, vocab_size, results_path, max_length)
model = BioBERTModel(hidden_size, layers_num, attention_heads_num, num_classes)

total_params = sum(p.numel() for p in model.parameters())
logger.info(f'num of paramters = {total_params}')

g = torch.Generator()
g.manual_seed(0)
train_generator = torch.utils.data.DataLoader(train_dataset, shuffle=True, num_workers=0, batch_size=8, generator=g)
valid_generator = torch.utils.data.DataLoader(valid_dataset, shuffle=True, num_workers=0, batch_size=1, generator=g)
test_generator = torch.utils.data.DataLoader(test_dataset, shuffle=True, num_workers=0, batch_size=1, generator=g)

train_model(model, train_generator, valid_generator, test_generator, epochs, print_training_loss, results_path)