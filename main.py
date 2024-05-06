import argparse

from preprocess.snp_splitting import *
from preprocess.snp_tokenize import *
from preprocess.snp_embed import *
from model.transformer_layer import *


if __name__ == '__main__':
    """
    Run the main.py file to start the program:
        + Process the input arguments
        + Read data
        + Preprocess data
        + Train models
        + Prediction
    """

    # ----------------------------------------------------
    # Process the arguments
    # ----------------------------------------------------
    parser = argparse.ArgumentParser()

    parser.add_argument("-ddi", "--data_dir", type=str,
                        default='/Users/nghihuynh/Documents/MscTUM_BioTech/thesis/code/transformer_SNP',
                        help="Path to the data folder")
    
    parser.add_argument("-mod", "--model", type=str,
                        default='Transformer',
                        help="NNet model for training the phenotype prediction")
    
    parser.add_argument("-kme", "--kmer", type=int,
                        default=3,
                        help="The number of kmer to tokenize sequence X")
    
    parser.add_argument("-tok", "--tokenize_type", type=str,
                        default='overlap',
                        help="Type of tokenizing methods: overlap, non_overlap, nuc")
    
    parser.add_argument("-min", "--minmax", type=int,
                        default=1,
                        help="Nomalizing y with min-max scaler")
    
    parser.add_argument("-sta", "--standa", type=int,
                        default=0,
                        help="Nomalizing X with min-max scaler")
    
    parser.add_argument("-pca", "--pcafit", type=int,
                        default=0,
                        help="Reducing and fitting X with PCA")
    
    parser.add_argument("-dat", "--dataset", type=int,
                        default=1,
                        help="The set of data using for training")
    
    parser.add_argument("-gpu", "--gpucuda", type=int,
                        default=0,
                        help="Training the model on GPU")

    args = vars(parser.parse_args())

    # ----------------------------------------------------
    # Check available GPUs, if not, run on CPUs
    # ----------------------------------------------------
    dev = "cpu"
    if args["gpucuda"] >= 1 and torch.cuda.is_available(): 
        print("GPU CUDA available, using GPU for training the models.")
        dev = "cuda:" + str(args["gpucuda"]-1) # to get the idx of gpu device
    else:
        print("GPU CUDA not available, using CPU instead.")
    device = torch.device(dev)

    # ----------------------------------------------------
    # Parsing the input arguments
    # ----------------------------------------------------
    datapath = args["data_dir"]
    model = args["model"]
    kmer = args["kmer"]
    tokenize_type = args["tokenize_type"]
    minmax_scale = args["minmax"]
    standa_scale = args["standa"]
    pca_fitting  = args["pcafit"]
    dataset = args["dataset"]
    gpucuda = args["gpucuda"]

    # print('-----------------------------------------------')
    # print('Input arguments: ')
    # print('   + data_dir: {}'.format(datapath))
    # print('   + model: {}'.format(model))
    # print('   + minmax_scale: {}'.format(minmax_scale))
    # print('   + standa_scale: {}'.format(standa_scale))
    # print('   + pca_fitting: {}'.format(pca_fitting))
    # print('   + dataset: pheno_{}'.format(dataset))
    # print('   + gpucuda: {}'.format(gpucuda))

    # data_variants = [minmax_scale, standa_scale, pca_fitting, dataset]
    # print('   + data_variants: {}'.format(data_variants))
    # print('-----------------------------------------------\n')

    # ----------------------------------------------------
    # Read data and preprocess
    # ----------------------------------------------------

    # One_hot Encoding
    # read_data_pheno(datapath, 1)
    # split_train_test_data(datapath, 1)

    # ----------------------------------------------------
    # Tune and evaluate the model performance
    # ----------------------------------------------------
    # set up parameters for tuning
    training_params_dict = {
        'num_trials': 100,
        'min_trials': 20,
        'percentile': 65,
        'optunaseed': 42,
        'num_epochs': 80,
        'early_stop': 20,
        'batch_size': 32
    }


    print('---------------------------------------------------------')
    # print('Tuning MLP with dataset pheno-{}, minmax={}, standard={}, pcafit={}'.format(dataset, minmax_scale, standa_scale, pca_fitting))
    print('---------------------------------------------------------\n')
    X_chr1_train, X_chr2_train, X_chr3_train, X_chr4_train, X_chr5_train = split_into_chromosome(datapath, dataset)
    y_train, X_test, y_test= load_split_data(datapath, dataset)

    # Tokenize data
    tokenized_chr1_seq = snp_tokenizer(tokenize_type, seqs=X_chr1_train, kmer=kmer)
    tokenized_chr2_seq = snp_tokenizer(tokenize_type, seqs=X_chr2_train, kmer=kmer)
    tokenized_chr3_seq = snp_tokenizer(tokenize_type, seqs=X_chr3_train, kmer=kmer)
    tokenized_chr4_seq = snp_tokenizer(tokenize_type, seqs=X_chr4_train, kmer=kmer)
    tokenized_chr5_seq = snp_tokenizer(tokenize_type, seqs=X_chr5_train, kmer=kmer)

    # Embed data
    # a. Embed token using index
    # embedded_seq = token_embed(tokenized_chr1_seq)
    # b. Embed token using Word2vec
    # embedded_seq = Word2vec_embed(tokenized_chr1_seq)
    # c. Embed token using BPE
    X_chr1_tokenizer = BPE_embed(X_chr1_train)
    # print(X_chr1_tokenizer.get_vocab_size())

    # choose_max_length(X_chr1_train, X_chr1_tokenizer) #218 # set max_len 220

    embedded_X_chr1 = encode(X_chr1_train, X_chr1_tokenizer) # assign idices to each token[13, 29, 5, 52, 18, ...]
    # print(embedded_X_chr1[0])
    
    # transform to tensor
    tensor_y = torch.Tensor(y_train).view(len(y_train),1) #torch.Size([450, 1])
    embedded_X_chr1 = torch.LongTensor(embedded_X_chr1)
    # print(embedded_X_chr1.shape)    # torch.Size([450, 220])
    
    seq_len = embedded_X_chr1.shape[1]
    src_vocab_size = X_chr1_tokenizer.get_vocab_size()

    # test = Test_Embedding_Positional(embedded_X_chr1, X_chr1_tokenizer.get_vocab_size(), seq_len)
    # print(test)

    # test_ecocer = Test_encoderblock(embedded_X_chr1, X_chr1_tokenizer.get_vocab_size(), seq_len)
    # print(test_ecocer.shape) # torch.Size([450, 220, 512]) (batch, max_len, embed_dim)
  
    # test_transformer = Test_transformerblock(embedded_X_chr1, src_vocab_size, seq_len)
    # print('Shape',test_transformer.shape) #torch.Size([450, 1])

    # train-test split for evaluation of the model
    X_train_chr1, X_val_chr1, y_train, y_val = train_test_split(embedded_X_chr1, tensor_y, train_size=0.7, shuffle=True)

    # Dataloader
    train_loader = DataLoader(dataset=list(zip(X_train_chr1, y_train)), batch_size=8, shuffle=True)
    val_loader = DataLoader(dataset=list(zip(X_val_chr1, y_val)), batch_size=8, shuffle=True)

    model = TransformerSNP(src_vocab_size, 
                           seq_len,
                           embed_dim=512,
                           num_blocks=6,
                           expansion_factor=4,
                           heads=8,
                           dropout=0.2)
    
    trained_model = train_model(model, train_loader, val_loader, epochs=2)

    y_pred = predict(model, val_loader)
    # collect mse, r2, explained variance
    test_mse = sklearn.metrics.mean_squared_error(y_true=y_val, y_pred=y_pred)
    test_exp_variance = sklearn.metrics.explained_variance_score(y_true=y_val, y_pred=y_pred)
    test_r2 = sklearn.metrics.r2_score(y_true=y_val, y_pred=y_pred)
    test_mae = sklearn.metrics.mean_absolute_error(y_true=y_val, y_pred=y_pred)

    print('--------------------------------------------------------------')
    print('Test MLP results: avg_loss={:.4f}, avg_expvar={:.4f}, avg_r2score={:.4f}, avg_mae={:.4f}'.format(test_mse, test_exp_variance, test_r2, test_mae))
    print('--------------------------------------------------------------')

    # train_dataset = dataset_tensor(embedded_X_chr1, y_train)
    exit(1)
    # best_params = tuning_MLP(datapath, X_train, y_train, data_variants, training_params_dict, device)
    # evaluate_result_MLP(datapath, X_train, y_train, X_test, y_test, best_params, data_variants, device)




    