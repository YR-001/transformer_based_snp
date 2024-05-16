import json
import copy
import torch
import optuna
import sklearn
import random

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold

from torch import optim
from torch.nn import Sequential, MaxPool1d, Flatten, LeakyReLU, BatchNorm1d, Dropout, Linear, ReLU, Tanh

from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, Dataset

from model import embed_layer, mh_attention, encoder, transformer, utils


# ==============================================================
# Utils/Help function
# ==============================================================
def get_activation_func(name):
    if name == 'ReLU':
        act_func = ReLU()
    elif name == 'LeakyReLU':
        act_func = LeakyReLU()
    else:
        act_func = Tanh()
    return act_func

# Create a custom dataset class to handle list of tensors
class ListTensorDataset(Dataset):
    def __init__(self, X_list, y_train):
        self.X_list = X_list
        self.y_train = y_train

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        # Return a list containing all tensors in the batch and the corresponding label
        return [X[idx] for X in self.X_list], self.y_train[idx]

# set seeds for reproducibility
def set_seeds(seed: int=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # if gpu cuda available
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)

# Create mask to ignore Padding token
def create_mask_each_chr(X):
    mask_all_chr = []
    for xi in X:
        mask = (xi != 0).astype(np.int32) # Shape: (batch_size, sequence_length)
        # Reshape the mask to the desired shape: (batch_size, 1, 1, sequence_length)
        final_mask = mask[:, np.newaxis, np.newaxis, :]  # Shape: (batch_size, 1, 1, sequence_length)
        mask_tensor = torch.LongTensor(final_mask)
        mask_all_chr.append(mask_tensor)
    return mask_all_chr

# def create_mask_each_chr(X):
#     mask_all_chr = []
#     for xi in X:
#         mask = (xi != 0).int()  # Shape: (batch_size, sequence_length)
#         # Reshape the mask to the desired shape: (batch_size, 1, 1, sequence_length)
#         final_mask = mask.unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, sequence_length)
#         mask_all_chr.append(final_mask)
#     return mask_all_chr

# Average transformer output along sequence-dim to process with linear head regression
class Pooling_Transformer_output(torch.nn.Module):
    def __init__(self):
        super(Pooling_Transformer_output, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=1)

# Concatenate each chromosome (output of Transformer) into 1 input for Regression block
def Concatenate_chr(chr1, chr2, chr3, chr4, chr5):
    output = torch.cat((chr1, chr2, chr3, chr4, chr5), 1)
    return output

# ==============================================================
# Define Transformer Model
# ==============================================================

class TransformerSNP(torch.nn.Module):
    def __init__(self, src_vocab_size, seq_len, tuning_params):
        super(TransformerSNP, self).__init__()
        print("[TransformerSNP-init]: inside TransformerSNP ")
        embedding_dimension = int(tuning_params['n_heads'] * tuning_params['d_k'])

        self.embedding = embed_layer.Embedding(src_vocab_size, embedding_dimension)
        self.positional_encoding = embed_layer.PositionalEncoding(embedding_dimension, seq_len, tuning_params['dropout'])
        self.encoder_blocks = torch.nn.ModuleList([
            encoder.Encoder(embedding_dimension, tuning_params['n_heads'], tuning_params['mlp_factor'], tuning_params['dropout'])
            for _ in range(tuning_params['n_blocks'])
        ])

    def forward(self, x, mask):
        x = self.embedding(x)
        print('[TransformerSNP] in forward: passed emdedding')
        x = self.positional_encoding(x)
        print('[TransformerSNP] in forward: passed positional_encoding')
        i = 0
        for block in self.encoder_blocks:
            x = block(x, mask)
            print('[TransformerSNP] in forward: passed encoder_blocks {}'.format(i))
            i = i + 1
        return x
    
class RegressionBlock(torch.nn.Module):
    def __init__(self, tuning_params):
        super(RegressionBlock, self).__init__()

        print("[RegressionBlock-init]: inside RegressionBlock")

        embedding_dimension = int(tuning_params['n_heads'] * tuning_params['d_k'])
        self.pooling_layer = Pooling_Transformer_output()
        print("[RegressionBlock-init]: passed Pooling_Transformer_output")

        self.dropout = Dropout(tuning_params['dropout'])
        print("[RegressionBlock-init]: passed Dropout")

        self.linear = Linear(in_features=embedding_dimension, out_features=1)
        print("[RegressionBlock-init]: passed Linear")

    def forward(self, x):
        x = self.pooling_layer(x)
        print("[RegressionBlock-forward]: passed pooling_layer")
        x = self.dropout(x)
        print("[RegressionBlock-forward]: passed dropout")
        x = self.linear(x)
        print("[RegressionBlock-forward]: passed linear")
        return x

class EnsembledModel(torch.nn.Module):
    def __init__(self, src_vocab_size, max_seq_lens, tuning_params):
        super(EnsembledModel, self).__init__()
        print("[EnsembledModel-init]: inside EnsembledModel ")
        self.src_vocab_size = src_vocab_size
        self.tuning_params = tuning_params
        self.transformer_chr1 = TransformerSNP(src_vocab_size, max_seq_lens[0], tuning_params)
        print("[EnsembledModel-init]: passed init TransformerSNP")
        self.transformer_chr2 = TransformerSNP(src_vocab_size, max_seq_lens[1], tuning_params)
        self.transformer_chr3 = TransformerSNP(src_vocab_size, max_seq_lens[2], tuning_params)
        self.transformer_chr4 = TransformerSNP(src_vocab_size, max_seq_lens[3], tuning_params)
        self.transformer_chr5 = TransformerSNP(src_vocab_size, max_seq_lens[4], tuning_params)
        self.regression_block = RegressionBlock(tuning_params)
    
    # Create mask to ignore Padding token
    def create_mask_list(self, list_X):
        mask_all_chr = []
        for xi in list_X:
            mask = (xi != 0).int()
            # final_mask = mask.unsqueeze(1).unsqueeze(2)
            final_mask = mask[:, np.newaxis, np.newaxis, :]
            # mask_tensor = torch.LongTensor(final_mask)
            mask_all_chr.append(final_mask)

        return mask_all_chr
    
    # Concatenate each chromosome (output of Transformer) into 1 input for Regression block
    def concatenate_chr(self, args):
        output = torch.cat(args, 1)
        return output
    
    def forward(self, list_X):
        print('[EnsembledModel] in forward')
        mask_X = self.create_mask_list(list_X)
        for i in range(len(mask_X)):
            print('  + Masking Chr{}: list_X[{}]={} mask_X[{}]={}'.format(i, i, list_X[i].shape, i, mask_X[i].shape))

        transformer_output1 = self.transformer_chr1(list_X[0], mask_X[0])
        transformer_output2 = self.transformer_chr1(list_X[1], mask_X[1])
        transformer_output3 = self.transformer_chr1(list_X[2], mask_X[2])
        transformer_output4 = self.transformer_chr1(list_X[3], mask_X[3])
        transformer_output5 = self.transformer_chr1(list_X[4], mask_X[4])

        concatenated_output = self.concatenate_chr([transformer_output1, transformer_output2, transformer_output3, transformer_output4, transformer_output5])
        concatenated_output = concatenated_output.detach_()
        print('[EnsembledModel] in forward: passed concatenate_chr')

        regression_output = self.regression_block(concatenated_output)
        print('[EnsembledModel] in forward: passed regression_block')
        output = regression_output.squeeze(1)

        return output
    
# ==============================================================
# Define training and validation loop
# ==============================================================
# Function to train the model for one epoch
def train_one_epoch(model, train_loader, loss_function, optimizer, device):
    
    model.train()
    
    # iterate through the train loader
    for i, (inputs, targets) in enumerate(train_loader):
        # inputs, targets = inputs.to(device), targets.to(device)
        # forward pass 
        print('[train_one_epoch] start training one epoch')

        pred_outputs = model(inputs)
        print('[train_one_epoch] iter {} - passed pred_outputs: {}'.format(i, pred_outputs))

        # calculate training loss
        loss_training = loss_function(pred_outputs, targets)
        print('[train_one_epoch] iter {} - passed loss_function'.format(i))

        # backward pass and optimization
        optimizer.zero_grad()
        print('[train_one_epoch] iter {} - passed optimizer.zero_grad'.format(i))

        loss_training.backward()
        print('[train_one_epoch] iter {} - passed backward'.format(i))

        optimizer.step()
        print('[train_one_epoch] iter {} - passed optimizer.step'.format(i))

# Function to validate the model for one epoch
def validate_one_epoch(model, val_loader, loss_function, device):

    # arrays for tracking eval results
    avg_loss = 0.0
    arr_val_losses = []

    # evaluate the trained model
    model.eval()
    with torch.no_grad():
        # Iterate through the validation loader
        for i, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            arr_val_losses.append(loss.item())
    
    # calculate average validation loss
    avg_loss = np.average(arr_val_losses)
    return avg_loss

# Function to make predictions using the given model
def predict(model, val_loader, device):
    model.eval()
    predictions = None
    with torch.no_grad():
        # iterate through the validation loader
        for i, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            # inputs  = inputs.float()
            outputs = model(inputs)
            # concatenate the predictions
            predictions = torch.clone(outputs) if predictions is None else torch.cat((predictions, outputs))

    # convert predictions to numpy array based on device
    if device == torch.device('cpu'):
        ret_output = predictions.detach().numpy()
    else:
        ret_output = predictions.cpu().detach().numpy()
    
    return ret_output

# Function to train model on train loader and evaluate on validation loader
# also return early_stopping_point
def train_val_loop(model, training_params, tuning_params, X_train_list, y_train, X_val_list, y_val, device):
    
    print('Going to train_val_loop')

    # transform data to tensor format
    list_tensor_X_train = [torch.from_numpy(item).long() for item in X_train_list]
    tensor_y_train = torch.Tensor(y_train)
   
    list_tensor_X_val = [torch.from_numpy(item).long() for item in X_val_list]
    tensor_y_val = torch.Tensor(y_val)
    
    # squeeze y to get suitable y dims for training Transformer
    tensor_y_train, tensor_y_val = tensor_y_train.view(len(y_train),1), tensor_y_val.view(len(y_val),1)

    # create the list dataset
    train_dataset = ListTensorDataset(list_tensor_X_train, y_train)
    val_dataset   = ListTensorDataset(list_tensor_X_val, y_train)
    
    # define data loaders for training and validation data
    # train_loader = DataLoader(dataset=list(zip(X_train, tensor_y_train)), batch_size=training_params['batch_size'], shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=training_params['batch_size'], shuffle=True)

    # val_loader = DataLoader(dataset=list(zip(X_val, tensor_y_val)), batch_size=training_params['batch_size'], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=training_params['batch_size'], shuffle=False)

    # define loss function and optimizer
    loss_function = torch.nn.MSELoss()
    print('[train_val_loop] Start optim.Adam')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=tuning_params['learning_rate'], weight_decay=tuning_params['weight_decay'])
    print('[train_val_loop] Passed optim.Adam')
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data) # weight, bias

    # track the best loss value and best model
    best_model = copy.deepcopy(model)
    best_loss  = None

    # track the epoch with best values
    epochs_no_improvement = 0
    early_stopping_point = None

    # training loop over epochs
    num_epochs = training_params['num_epochs']
    early_stop_patience = training_params['early_stop']
    for epoch in range(num_epochs):
        print('Start going to train_one_epch')
        train_one_epoch(model, train_loader, loss_function, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, loss_function, device)
        
        # check if the current validation loss is the best observed so far
        # if current val loss is not the best, increase the count of epochs with no improvement
        if best_loss == None or val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model)
            epochs_no_improvement = 0
        else:
            epochs_no_improvement += 1
        
        print('Epoch {}/{}: current_loss={:.5f} | best_loss={:.5f}'.format(epoch, num_epochs, val_loss, best_loss))

        # check if early stopping criteria are met
        # if the current epoch is greater than or equal to 20 
        # and epochs with no improvement surpass early stopping patience
        if epoch >= 20 and epochs_no_improvement >= early_stop_patience:
            # set the early stopping point
            early_stopping_point = epoch - early_stop_patience
            print("Stopped at epoch " + str(epoch) + "| " + "Early stopping point = " + str(early_stopping_point))
            # predict using the best model 
            model = best_model
            y_pred = predict(model, val_loader, device)
            return y_pred, early_stopping_point
    
    # return the best predicted values
    y_pred = predict(best_model, val_loader, device)
    return y_pred, early_stopping_point

# ==============================================================
# Define objective function for tuning hyperparameters
# ==============================================================
def objective(trial, list_X_train, src_vocab_size, y, data_variants, training_params_dict, avg_stop_epochs, device):
    
    # for tuning parameters
    tuning_params_dict = {
        'learning_rate': trial.suggest_categorical('learning_rate', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]), 
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2),
        'n_blocks': trial.suggest_int("n_blocks", 2, 6,step=2),
        'n_heads': trial.suggest_int("n_heads", 2, 6,step=2),
        'd_k': trial.suggest_categorical('d_k', [16, 32, 64]),
        'mlp_factor': trial.suggest_int("mlp_factor", 2, 4,step=1),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
    }

    # log early stopping point at each fold
    early_stopping_points = []
    
    # iterate for training and tuning
    print('\n----------------------------------------------')
    print("Params for Trial " + str(trial.number))
    print(trial.params)
    print('----------------------------------------------')

    # tracking the results
    first_obj_values = []
    second_obj_values = []

    # forl cross-validation kfolds, default = 5 folds
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
    # Split the data into train and validation sets for all chromosomes at once
    all_folds = [kfold.split(X_chr) for X_chr in list_X_train]

    # Pipeline-like structure
    for fold, fold_indices in enumerate(zip(*all_folds), start=1):
        X_train_list = []
        X_val_list = []

        print("Preprocessing Fold No.{}: ".format(fold))
        for i, (train_ids, val_ids) in enumerate(fold_indices, start=1):
            
            X_chr = list_X_train[i-1]
            X_train, X_val = X_chr[train_ids], X_chr[val_ids]
            y_train, y_val = y[train_ids], y[val_ids]
            X_train_list.append(X_train)
            X_val_list.append(X_val)

        # get the max sequence length of each chrobosome
        max_seq_lens = []
        for i in range(len(X_train_list)):
            max_seq_lens.append(X_train_list[i].shape[1])
        print("Max sequence lengths of each chrobosome: {} ".format(max_seq_lens))

        try:
            # model = RegressionBlock(tuning_params=tuning_params_dict).to(device)
            model = EnsembledModel(src_vocab_size, max_seq_lens, tuning_params_dict).to(device)
            print("Passed create EnsembledModel")

        except Exception as err:
            print('Trial failed. Error in model creation, {}'.format(err))
            raise optuna.exceptions.TrialPruned()

        # call training model over each fold
        try:
            y_pred, stopping_point = train_val_loop(model, training_params_dict, tuning_params_dict,
                                     X_train_list, y_train, X_val_list, y_val, device)
            # record the early-stopping points
            if stopping_point is not None:
                early_stopping_points.append(stopping_point)
            else:
                early_stopping_points.append(training_params_dict['num_epochs'])
            # calculate objective value
            obj_value1 = sklearn.metrics.mean_squared_error(y_true=y_val, y_pred=y_pred)
            obj_value2 = sklearn.metrics.explained_variance_score(y_true=y_val, y_pred=y_pred)
            print('      explained_var={:.5f} | mse_loss={:.5f}'.format(obj_value2, obj_value1))

            # report pruned values
            trial.report(value=obj_value2, step=fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            # accumulate the obj val losses
            first_obj_values.append(obj_value1)
            second_obj_values.append(obj_value2)

        # for pruning the tuning process
        except (RuntimeError, TypeError, ValueError) as exc:
            print(exc)
            if 'out of memory' in str(exc):
                print('Out of memory')
            else:
                print('Trial failed. Error in optim loop.')
            raise optuna.exceptions.TrialPruned()
    
    # return the average val loss
    current_val_loss = float(np.mean(first_obj_values))
    current_val_expv = float(np.mean(second_obj_values))

    # Average value of early stopping points of all innerfolds for refitting of final model
    early_stopping_point = int(np.mean(early_stopping_points))
    print('----------------------------------------------')
    print("Average early_stopping_point: {}| avg_exp_var={:.5f}| avg_loss={:.5f}".format(early_stopping_point, current_val_expv, current_val_loss))
    print('----------------------------------------------\n')

    # try to return avg stop epochs for a specific trial 
    avg_stop_epochs[trial.number] = early_stopping_point

    # return current_val_expv
    return current_val_loss

# ==============================================================
# Call tuning function
# ==============================================================
def tuning_Transformer(datapath, list_X_train, src_vocab_size, y, data_variants, training_params_dict, device):

    # set seeds for reproducibility
    set_seeds()

    # for tracking the tuning information
    # minmax = '_minmax' if data_variants[0] == True else ''
    # standard = '_standard' if data_variants[1] == True else ''
    # pcafitting = '_pca' if data_variants[2] == True else ''
    # pheno = str(data_variants[3])

    # create a list to keep track all stopping epochs during a hyperparameter search
    # because in some cases, the trial is pruned so the stopping epochs default = num_epochs
    avg_stopping_epochs = [training_params_dict['num_epochs']] * training_params_dict['num_trials']

    # create an optuna tuning object, num trials default = 100
    num_trials = training_params_dict['num_trials']
    study = optuna.create_study(
        study_name='transformer'+'mseloss_'+'data',
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=training_params_dict['optunaseed']),
        pruner=optuna.pruners.PercentilePruner(percentile=training_params_dict['percentile'], n_min_trials=training_params_dict['min_trials'])
    )
    
    # searching loop with objective tuning
    study.optimize(lambda trial: objective(trial, list_X_train, src_vocab_size, y, data_variants, training_params_dict, avg_stopping_epochs, device), n_trials=num_trials)

    # get early stopping of the best trial
    num_avg_stop_epochs = avg_stopping_epochs[study.best_trial.number]
    
    # print statistics after tuning
    print("Optuna study finished, study statistics:")
    print("  Finished trials: ", len(study.trials))
    print("  Pruned trials: ", len(study.get_trials(states=(optuna.trial.TrialState.PRUNED,))))
    print("  Completed trials: ", len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))))
    print("  Best Trial: ", study.best_trial.number)
    print("  Value: ", study.best_trial.value)
    print("  AVG stopping: ", num_avg_stop_epochs)
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print("    {}: {}".format(key, value))
    print('----------------------------------------------\n')

    best_params = study.best_trial.params
    best_params['avg_epochs'] = num_avg_stop_epochs
    print('Check best params: {}'.format(best_params))

    # record best parameters to file
    with open(f"./tuned_transformer_" + "pheno" + ".json", 'w') as fp:
        json.dump(best_params, fp)

    return best_params

# ==============================================================
# Evaluation the performance on test set
# ==============================================================
def evaluate_result_Transformer(datapath, list_X_train, src_vocab_size, y_train, list_X_test, y_test, best_params, data_variants, device):

    # set seeds for reproducibility
    set_seeds()
    
    # for tracking the tuning information
    # minmax = '_minmax' if data_variants[0] == True else ''
    # standard = '_standard' if data_variants[1] == True else ''
    # pcafitting = '_pca' if data_variants[2] == True else ''
    # pheno = str(data_variants[3])

    # extract preprocessed data variants for tuning
    # minmax_scaler_mode = data_variants[0]
    # standard_scaler_mode = data_variants[1]
    # pca_fitting_mode = data_variants[2]

    # preprocessing data
    # if minmax_scaler_mode == 1: # minmax scaler
    #     y_train, y_test = preprocess_mimax_scaler(y_train, y_test)
    # if standard_scaler_mode == 1: # standard scaler
    #     X_train, X_test = preprocess_standard_scaler(X_train, X_test)
    # if pca_fitting_mode == 1: # pca fitting
    #     X_train, X_test = decomposition_PCA(X_train, X_test, best_params['pca'])

    # extract training and tuned parameters
    batch_size = 32
    num_epochs = best_params['avg_epochs']
    learning_rate = best_params['learning_rate']
    momentum = best_params['weight_decay']

    list_X_mask = create_mask_each_chr(list_X_train)

    chr1_model = TransformerSNP(list_X_mask[0], src_vocab_size, seq_len=list_X_train[0].shape[1], tuning_params=best_params).to(device)
    chr2_model = TransformerSNP(list_X_mask[1], src_vocab_size, seq_len=list_X_train[1].shape[1], tuning_params=best_params).to(device)
    chr3_model = TransformerSNP(list_X_mask[2], src_vocab_size, seq_len=list_X_train[2].shape[1], tuning_params=best_params).to(device)
    chr4_model = TransformerSNP(list_X_mask[3], src_vocab_size, seq_len=list_X_train[3].shape[1], tuning_params=best_params).to(device)
    chr5_model = TransformerSNP(list_X_mask[4], src_vocab_size, seq_len=list_X_train[4].shape[1], tuning_params=best_params).to(device)

    chr1_output, chr2_output, chr3_output, chr4_output, chr5_output = chr1_model(torch.LongTensor(list_X_train[0])), chr2_model(torch.LongTensor(list_X_train[1])), chr3_model(torch.LongTensor(list_X_train[2])), chr4_model(torch.LongTensor(list_X_train[3])), chr5_model(torch.LongTensor(list_X_train[4]))
    
    X_train = Concatenate_chr(chr1_output, chr2_output, chr3_output, chr4_output, chr5_output)
    X_train = X_train.detach_()

    list_X_mask_test = create_mask_each_chr(list_X_test)

    chr1_test_model = TransformerSNP(list_X_mask_test[0], src_vocab_size, seq_len=list_X_test[0].shape[1], tuning_params=best_params).to(device)
    chr2_test_model = TransformerSNP(list_X_mask_test[1], src_vocab_size, seq_len=list_X_test[1].shape[1], tuning_params=best_params).to(device)
    chr3_test_model = TransformerSNP(list_X_mask_test[2], src_vocab_size, seq_len=list_X_test[2].shape[1], tuning_params=best_params).to(device)
    chr4_test_model = TransformerSNP(list_X_mask_test[3], src_vocab_size, seq_len=list_X_test[3].shape[1], tuning_params=best_params).to(device)
    chr5_test_model = TransformerSNP(list_X_mask_test[4], src_vocab_size, seq_len=list_X_test[4].shape[1], tuning_params=best_params).to(device)

    chr1_test_output, chr2_test_output, chr3_test_output, chr4_test_output, chr5_test_output = chr1_test_model(torch.LongTensor(list_X_test[0])), chr2_test_model(torch.LongTensor(list_X_test[1])), chr3_test_model(torch.LongTensor(list_X_test[2])), chr4_test_model(torch.LongTensor(list_X_test[3])), chr5_test_model(torch.LongTensor(list_X_test[4]))
    
    X_test = Concatenate_chr(chr1_test_output, chr2_test_output, chr3_test_output, chr4_test_output, chr5_test_output)
    X_test = X_test.detach_()

    # Sequence length
    seq_len = X_train.shape[1]

    # create model
    model = RegressionBlock(tuning_params=best_params).to(device)

    # transform data to tensor format
    # tensor_X_train = torch.LongTensor(X_train)
    tensor_y_train = torch.Tensor(y_train)
    # tensor_X_test = torch.LongTensor(X_test)
    tensor_y_test = torch.Tensor(y_test)

    # squeeze y for training Transformer to tensor
    tensor_y_train, tensor_y_test = tensor_y_train.view(len(y_train),1), tensor_y_test.view(len(y_test),1)

    # define data loaders for training and testing data
    train_loader = DataLoader(dataset=list(zip(X_train, tensor_y_train)), batch_size=batch_size, shuffle=True)
    test_loader   = DataLoader(dataset=list(zip(X_test, tensor_y_test)), batch_size=batch_size, shuffle=False)

    # define loss function and optimizer
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=momentum)

    # training loop over epochs
    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, loss_function, optimizer, device)
    
    # predict result test 
    y_pred = predict(model, test_loader, device)

    # collect mse, r2, explained variance
    test_mse = sklearn.metrics.mean_squared_error(y_true=y_test, y_pred=y_pred)
    test_exp_variance = sklearn.metrics.explained_variance_score(y_true=y_test, y_pred=y_pred)
    test_r2 = sklearn.metrics.r2_score(y_true=y_test, y_pred=y_pred)
    test_mae = sklearn.metrics.mean_absolute_error(y_true=y_test, y_pred=y_pred)

    print('--------------------------------------------------------------')
    print('Test Transformer results: avg_loss={:.4f}, avg_expvar={:.4f}, avg_r2score={:.4f}, avg_mae={:.4f}'.format(test_mse, test_exp_variance, test_r2, test_mae))
    print('--------------------------------------------------------------')


    return test_exp_variance


