import pandas as pd
import numpy as np

import torch

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


# -------------------------------------------------------------
#  1. Read data + Split into train, test
# -------------------------------------------------------------
def read_data_pheno(datapath, type):
    """
    Read the raw data or impured data by pandas,
        + in: temporarily the data path is fixed
        + out: the returned format is pandas dataframe and write to .csv files
    """

    df_genotypes  = pd.read_csv(datapath + '/data/raw_geneotype_dataset.csv')
    df_phenotypes = pd.read_csv(datapath + '/data/phenotype_data.csv')

    # delete nan values from phenotye data
    df_pheno = df_phenotypes.dropna(subset=['sample_ids', 'pheno'+str(type)]) #only delte missing value in pheno1 column
    # print('Number of Corresponding phenotype values:\t%d' %df_pheno.shape[0])

    # select the samples id that we have in y_matrix.csv
    unique_ids_ymatrix = df_pheno['sample_ids'].unique()

    # filter the sample ids in x_matrix.csv that fits with the ids in y_matrix.csv
    df_genotypes = df_genotypes[df_genotypes['sample_ids'].isin(unique_ids_ymatrix)]

    # get the list of common ids between two datasets
    common_sample_ids = df_genotypes['sample_ids'].unique()

    # fileter again the sample ids in y_matrix.csv
    df_pheno = df_pheno[df_pheno['sample_ids'].isin(common_sample_ids)]

    # then map the continuous_values of y to x
    phenotype_dict_arr = df_pheno.set_index('sample_ids').to_dict()['pheno'+str(type)]
    trans_phenotype_dict = {key: float(value) for key, value in phenotype_dict_arr.items()}
    df_genotypes['pheno'+str(type)] = df_genotypes['sample_ids'].map(trans_phenotype_dict) # add label column to genotypes data

    # create new X1, y1
    X = df_genotypes.iloc[:,1:df_genotypes.shape[1]-1]
    y = df_genotypes[['sample_ids', 'pheno'+str(type)]]

    # convert new dataset to csv
    X.to_csv(datapath + '/data/pheno' + str(type) + '/x_matrix.csv')
    y.to_csv(datapath + '/data/pheno' + str(type) + '/y_matrix.csv')
    # print('------------------------------------------------------------------\n')

def split_train_test_data(datapath, type):
    """
    Read the preprocessed data after matching input features and labels,
        + in: path to X and y
        + out: the X and y as type numpy array
    """
    
    X = pd.read_csv(datapath + '/data/pheno' + str(type) + '/x_matrix.csv')
    y = pd.read_csv(datapath + '/data/pheno' + str(type) + '/y_matrix.csv')

    X_nparray = X.iloc[:,2:]
    y_nparray = y.iloc[:,2]

    X_train, X_test, y_train, y_test = train_test_split(X_nparray, y_nparray, train_size=0.9, random_state=42, shuffle=True)
    
    X_train.to_csv(datapath + '/data/pheno' + str(type) + '/x_train.csv')
    y_train.to_csv(datapath + '/data/pheno' + str(type) + '/y_train.csv')
    X_test.to_csv(datapath + '/data/pheno' + str(type) + '/x_test.csv')
    y_test.to_csv(datapath + '/data/pheno' + str(type) + '/y_test.csv')

# -------------------------------------------------------------
#  2. Split data into each chromosome
# -------------------------------------------------------------

def split_into_chromosome_train(datapath, type):

    X_train = pd.read_csv(datapath + '/data/pheno' + str(type) + '/x_train.csv')

    # Load data into a DataFrame object:
    df_train = pd.DataFrame(X_train)

    # Sorting the Column name. Format: chromosome_position
    # DataFrame.columns: return the column labels
    # for each column name: we'll provide a tuple (chromosome, position) to the sorting function. 
    # As tuples are sorted by comparing them field by field, this will effectively sort column name by chromosome, then by position.
    sorted_columns_train = sorted(df_train.columns[1:], key=lambda x: (int(x.split("_")[0]), int(x.split("_")[1])))

    # Sort the DataFrame based on the sorted columns
    sorted_df_train = df_train[["Unnamed: 0"] + sorted_columns_train]

    # Filter groups of chromosomes
    chr1_columns = [column for column in sorted_df_train.columns if column.startswith('1_')]
    df_chr1 = sorted_df_train[['Unnamed: 0'] + chr1_columns]

    chr2_columns = [column for column in sorted_df_train.columns if column.startswith('2_')]
    df_chr2 = sorted_df_train[['Unnamed: 0'] + chr2_columns]

    chr3_columns = [column for column in sorted_df_train.columns if column.startswith('3_')]
    df_chr3 = sorted_df_train[['Unnamed: 0'] + chr3_columns]

    chr4_columns = [column for column in sorted_df_train.columns if column.startswith('4_')]
    df_chr4 = sorted_df_train[['Unnamed: 0'] + chr4_columns]

    chr5_columns = [column for column in sorted_df_train.columns if column.startswith('5_')]
    df_chr5 = sorted_df_train[['Unnamed: 0'] + chr5_columns]

    # Convert into numpy array
    X_chr1, X_chr2, X_chr3, X_chr4, X_chr5 = df_chr1.iloc[:,1:].to_numpy(), df_chr2.iloc[:,1:].to_numpy(), df_chr3.iloc[:,1:].to_numpy(), df_chr4.iloc[:,1:].to_numpy(), df_chr5.iloc[:,1:].to_numpy()

    # Create a list of sequences 
    # from [['C' 'G' 'A' ... 'A' 'G' 'G'], [..], ..] --> ['CGA..AGG', '...',...]
    X_chr1_train = [''.join(seq) for seq in X_chr1]
    X_chr2_train = [''.join(seq) for seq in X_chr2]
    X_chr3_train = [''.join(seq) for seq in X_chr3]
    X_chr4_train = [''.join(seq) for seq in X_chr4]
    X_chr5_train = [''.join(seq) for seq in X_chr5]

    return X_chr1_train, X_chr2_train, X_chr3_train, X_chr4_train, X_chr5_train

def split_into_chromosome_test(datapath, type):

    X_test = pd.read_csv(datapath + '/data/pheno' + str(type) + '/x_test.csv')

    # Load data into a DataFrame object:
    df_train = pd.DataFrame(X_test)

    # Sorting the Column name. Format: chromosome_position
    # DataFrame.columns: return the column labels
    # for each column name: we'll provide a tuple (chromosome, position) to the sorting function. 
    # As tuples are sorted by comparing them field by field, this will effectively sort column name by chromosome, then by position.
    sorted_columns_train = sorted(df_train.columns[1:], key=lambda x: (int(x.split("_")[0]), int(x.split("_")[1])))

    # Sort the DataFrame based on the sorted columns
    sorted_df_train = df_train[["Unnamed: 0"] + sorted_columns_train]

    # Filter groups of chromosomes
    chr1_columns = [column for column in sorted_df_train.columns if column.startswith('1_')]
    df_chr1 = sorted_df_train[['Unnamed: 0'] + chr1_columns]

    chr2_columns = [column for column in sorted_df_train.columns if column.startswith('2_')]
    df_chr2 = sorted_df_train[['Unnamed: 0'] + chr2_columns]

    chr3_columns = [column for column in sorted_df_train.columns if column.startswith('3_')]
    df_chr3 = sorted_df_train[['Unnamed: 0'] + chr3_columns]

    chr4_columns = [column for column in sorted_df_train.columns if column.startswith('4_')]
    df_chr4 = sorted_df_train[['Unnamed: 0'] + chr4_columns]

    chr5_columns = [column for column in sorted_df_train.columns if column.startswith('5_')]
    df_chr5 = sorted_df_train[['Unnamed: 0'] + chr5_columns]

    # Convert into numpy array
    X_chr1, X_chr2, X_chr3, X_chr4, X_chr5 = df_chr1.iloc[:,1:].to_numpy(), df_chr2.iloc[:,1:].to_numpy(), df_chr3.iloc[:,1:].to_numpy(), df_chr4.iloc[:,1:].to_numpy(), df_chr5.iloc[:,1:].to_numpy()

    # Create a list of sequences 
    # from [['C' 'G' 'A' ... 'A' 'G' 'G'], [..], ..] --> ['CGA..AGG', '...',...]
    X_chr1_test = [''.join(seq) for seq in X_chr1]
    X_chr2_test = [''.join(seq) for seq in X_chr2]
    X_chr3_test = [''.join(seq) for seq in X_chr3]
    X_chr4_test = [''.join(seq) for seq in X_chr4]
    X_chr5_test = [''.join(seq) for seq in X_chr5]

    return X_chr1_test, X_chr2_test, X_chr3_test, X_chr4_test, X_chr5_test

# -------------------------------------------------------------
#  3. Load data
# -------------------------------------------------------------

def load_split_data(datapath, type):

    y_train = pd.read_csv(datapath + '/data/pheno' + str(type) + '/y_train.csv')
    y_test = pd.read_csv(datapath + '/data/pheno' + str(type) + '/y_test.csv')

    y_train_nparray = y_train.iloc[:,1].to_numpy()
    y_test_nparray = y_test.iloc[:,1].to_numpy()


    return y_train_nparray, y_test_nparray