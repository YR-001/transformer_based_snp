import argparse

from preprocess.snp_splitting import *
from preprocess.snp_tokenize import *
from preprocess.snp_embed import *
from model.transformer_layer import *
from model.model_transformer_snp import *


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

    data_variants = [minmax_scale, standa_scale, pca_fitting, dataset]
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
        'num_trials': 1,
        'min_trials': 20,
        'percentile': 65,
        'optunaseed': 42,
        'num_epochs': 1,
        'early_stop': 20,
        'batch_size': 32
    }


    print('---------------------------------------------------------')
    # print('Tuning MLP with dataset pheno-{}, minmax={}, standard={}, pcafit={}'.format(dataset, minmax_scale, standa_scale, pca_fitting))
    print('---------------------------------------------------------\n')
    # len each chromosome_train: 2401, 1696, 2015, 1727, 2161
    X_chr1_train, X_chr2_train, X_chr3_train, X_chr4_train, X_chr5_train = split_into_chromosome_train(datapath, dataset)
    X_chr1_test, X_chr2_test, X_chr3_test, X_chr4_test, X_chr5_test = split_into_chromosome_test(datapath, dataset)
    y_train, y_test= load_split_data(datapath, dataset)

    # Tokenize data
    # tokenized_chr1_seq = snp_tokenizer(tokenize_type, seqs=X_chr1_train, kmer=kmer)
    # tokenized_chr2_seq = snp_tokenizer(tokenize_type, seqs=X_chr2_train, kmer=kmer)
    # tokenized_chr3_seq = snp_tokenizer(tokenize_type, seqs=X_chr3_train, kmer=kmer)
    # tokenized_chr4_seq = snp_tokenizer(tokenize_type, seqs=X_chr4_train, kmer=kmer)
    # tokenized_chr5_seq = snp_tokenizer(tokenize_type, seqs=X_chr5_train, kmer=kmer)

    ##################################################################################
    # Embed data
    # a. Embed token using index
    # embedded_seq = token_embed(tokenized_chr1_seq)
    # b. Embed token using Word2vec
    # embedded_seq = Word2vec_embed(tokenized_chr1_seq)
    # print(X_chr1_train)
    X_chr1_kmer = seqs2kmer_nonoverlap(X_chr1_train, kmer)
    X_chr2_kmer = seqs2kmer_nonoverlap(X_chr2_train, kmer)
    X_chr3_kmer = seqs2kmer_nonoverlap(X_chr3_train, kmer)
    X_chr4_kmer = seqs2kmer_nonoverlap(X_chr4_train, kmer)
    X_chr5_kmer = seqs2kmer_nonoverlap(X_chr5_train, kmer)

    X_chr1_tokenizer = kmer_embed(X_chr1_kmer, 1)

    X_test_chr1_kmer = seqs2kmer_nonoverlap(X_chr1_train, kmer)
    X_test_chr2_kmer = seqs2kmer_nonoverlap(X_chr2_train, kmer)
    X_test_chr3_kmer = seqs2kmer_nonoverlap(X_chr3_train, kmer)
    X_test_chr4_kmer = seqs2kmer_nonoverlap(X_chr4_train, kmer)
    X_test_chr5_kmer = seqs2kmer_nonoverlap(X_chr5_train, kmer)

    # x1 = choose_max_length(X_chr1_kmer, X_chr1_tokenizer) #801
    # x2 = choose_max_length(X_chr2_kmer, X_chr1_tokenizer) #566
    # x3 = choose_max_length(X_chr3_kmer, X_chr1_tokenizer) #672
    # x4 = choose_max_length(X_chr4_kmer, X_chr1_tokenizer) #576
    # x5 = choose_max_length(X_chr5_kmer, X_chr1_tokenizer) #721

    embedded_X_chr1 = np.array(encode(X_chr1_kmer, X_chr1_tokenizer, 801))
    embedded_X_chr2 = np.array(encode(X_chr2_kmer, X_chr1_tokenizer, 566))
    embedded_X_chr3 = np.array(encode(X_chr3_kmer, X_chr1_tokenizer, 672))
    embedded_X_chr4 = np.array(encode(X_chr4_kmer, X_chr1_tokenizer, 576))
    embedded_X_chr5 = np.array(encode(X_chr5_kmer, X_chr1_tokenizer, 721))

    list_X_train = [embedded_X_chr1, embedded_X_chr2, embedded_X_chr3, embedded_X_chr4, embedded_X_chr5]

    embedded_X_test_chr1 = np.array(encode(X_test_chr1_kmer, X_chr1_tokenizer, 801)) # assign idices to each token[13, 29, 5, 52, 18, ...]
    embedded_X_test_chr2 = np.array(encode(X_test_chr2_kmer, X_chr1_tokenizer, 566))
    embedded_X_test_chr3 = np.array(encode(X_test_chr3_kmer, X_chr1_tokenizer, 672))
    embedded_X_test_chr4 = np.array(encode(X_test_chr4_kmer, X_chr1_tokenizer, 576))
    embedded_X_test_chr5 = np.array(encode(X_test_chr5_kmer, X_chr1_tokenizer, 721))
    
    list_X_test = [embedded_X_test_chr1, embedded_X_test_chr2, embedded_X_test_chr3, embedded_X_test_chr4, embedded_X_test_chr5]

    src_vocab_size = X_chr1_tokenizer.get_vocab_size()

    best_params = tuning_Transformer(datapath, list_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
    evaluate_result_Transformer(datapath, list_X_train, src_vocab_size, y_train, list_X_test, y_test, best_params, data_variants, device)

    
    exit(1)
    ##################################################################################
    # c. Embed token using BPE
    X_chr1_tokenizer = BPE_embed(X_chr1_train, 1)
    X_chr2_tokenizer = BPE_embed(X_chr2_train, 2)
    X_chr3_tokenizer = BPE_embed(X_chr3_train, 3)
    X_chr4_tokenizer = BPE_embed(X_chr4_train, 4)
    X_chr5_tokenizer = BPE_embed(X_chr5_train, 5)

    # pad_token = torch.tensor([X_chr1_tokenizer.token_to_id("[PAD]")], dtype=torch.int64)

    # X_test_chr1_tokenizer = BPE_embed(X_chr1_test, 6)
    # X_test_chr2_tokenizer = BPE_embed(X_chr2_test, 7)
    # X_test_chr3_tokenizer = BPE_embed(X_chr3_test, 8)
    # X_test_chr4_tokenizer = BPE_embed(X_chr4_test, 9)
    # X_test_chr5_tokenizer = BPE_embed(X_chr5_test, 10)
    
    
    #Vocab_size is the same, set vocab_size=2048
    # x1_vocab = print(X_chr1_tokenizer.get_vocab_size())
    # x2_vocab = print(X_chr2_tokenizer.get_vocab_size())
    # x3_vocab = print(X_chr3_tokenizer.get_vocab_size())
    # x4_vocab = print(X_chr4_tokenizer.get_vocab_size())
    # x5_vocab = print(X_chr5_tokenizer.get_vocab_size())

    # Check max_len for each chrmosome in train dataset
    # x1 = choose_max_length(X_chr1_train, X_chr1_tokenizer)  # max_len = 460 
    # x2 =choose_max_length(X_chr1_train, X_chr2_tokenizer)   # max_len = 650 
    # x3= choose_max_length(X_chr1_train, X_chr3_tokenizer)   # max_len = 635 
    # x4 =choose_max_length(X_chr1_train, X_chr4_tokenizer)   # max_len = 643 
    # x5 =choose_max_length(X_chr1_train, X_chr5_tokenizer)   # max_len = 621 

    # Check max_len for each chrmosome in test dataset
    # x1 = choose_max_length(X_chr1_test, X_test_chr1_tokenizer)  # max_len = 464 
    # x2 =choose_max_length(X_chr1_test, X_test_chr2_tokenizer)   # max_len = 659 
    # x3= choose_max_length(X_chr1_test, X_test_chr3_tokenizer)   # max_len = 633 
    # x4 =choose_max_length(X_chr1_test, X_test_chr4_tokenizer)   # max_len = 641 
    # x5 =choose_max_length(X_chr1_test, X_test_chr5_tokenizer)   # max_len = 620 

    # x1 = choose_max_length(X_chr1_test, X_chr1_tokenizer)  # max_len = 450 
    # x2 =choose_max_length(X_chr1_test, X_chr2_tokenizer)   # max_len = 642 
    # x3= choose_max_length(X_chr1_test, X_chr3_tokenizer)   # max_len = 631 
    # x4 =choose_max_length(X_chr1_test, X_chr4_tokenizer)   # max_len = 636 
    # x5 =choose_max_length(X_chr1_test, X_chr5_tokenizer)   # max_len = 616 



    embedded_X_chr1 = np.array(encode(X_chr1_train, X_chr1_tokenizer, 460)) # assign idices to each token[13, 29, 5, 52, 18, ...]
    embedded_X_chr2 = np.array(encode(X_chr2_train, X_chr2_tokenizer, 650))
    embedded_X_chr3 = np.array(encode(X_chr3_train, X_chr3_tokenizer, 635))
    embedded_X_chr4 = np.array(encode(X_chr4_train, X_chr4_tokenizer, 643))
    embedded_X_chr5 = np.array(encode(X_chr5_train, X_chr5_tokenizer, 621))

    list_X_train = [embedded_X_chr1, embedded_X_chr2, embedded_X_chr3, embedded_X_chr4, embedded_X_chr5]

    embedded_X_test_chr1 = np.array(encode(X_chr1_test, X_chr1_tokenizer, 450)) # assign idices to each token[13, 29, 5, 52, 18, ...]
    embedded_X_test_chr2 = np.array(encode(X_chr2_test, X_chr2_tokenizer, 642))
    embedded_X_test_chr3 = np.array(encode(X_chr3_test, X_chr3_tokenizer, 631))
    embedded_X_test_chr4 = np.array(encode(X_chr4_test, X_chr4_tokenizer, 636))
    embedded_X_test_chr5 = np.array(encode(X_chr5_test, X_chr5_tokenizer, 616))
    
    list_X_test = [embedded_X_test_chr1, embedded_X_test_chr2, embedded_X_test_chr3, embedded_X_test_chr4, embedded_X_test_chr5]

    # X_train = np.concatenate((embedded_X_chr1, embedded_X_chr2, embedded_X_chr3, embedded_X_chr4, embedded_X_chr5), axis=1)
    # X_test = np.concatenate((embedded_X_test_chr1, embedded_X_test_chr2, embedded_X_test_chr3, embedded_X_test_chr4, embedded_X_test_chr5), axis=1)

    # transform to tensor
    # tensor_y = torch.Tensor(y_train).view(len(y_train), 1)
    # tensor_X = torch.LongTensor(X_train)

    # seq_len = tensor_X.shape[1]
    src_vocab_size = X_chr1_tokenizer.get_vocab_size()

    best_params = tuning_Transformer(datapath, list_X_train, src_vocab_size, y_train, data_variants, training_params_dict, device)
    evaluate_result_Transformer(datapath, list_X_train, src_vocab_size, y_train, list_X_test, y_test, best_params, data_variants, device)

    exit(1)
    # test = Test_Embedding_Positional(embedded_X_chr1, X_chr1_tokenizer.get_vocab_size(), seq_len)
    # print(test)

    # test_ecocer = Test_encoderblock(embedded_X_chr1, X_chr1_tokenizer.get_vocab_size(), seq_len)
    # print(test_ecocer.shape) # torch.Size([450, 220, 512]) (batch, max_len, embed_dim)
  
    # test_transformer = Test_transformerblock(embedded_X_chr1, src_vocab_size, seq_len)
    # print('Shape',test_transformer.shape) #torch.Size([450, 1])

    # train-test split for evaluation of the model
    # X_train, X_val, y_train, y_val = train_test_split(tensor_X, tensor_y, train_size=0.7, shuffle=True)

    # Dataloader
    train_loader = DataLoader(dataset=list(zip(X_train, y_train)), batch_size=45, shuffle=True)
    val_loader = DataLoader(dataset=list(zip(X_val, y_val)), batch_size=45, shuffle=True)
    
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


    # best_params = tuning_MLP(datapath, X_train, y_train, data_variants, training_params_dict, device)
    # evaluate_result_MLP(datapath, X_train, y_train, X_test, y_test, best_params, data_variants, device)

