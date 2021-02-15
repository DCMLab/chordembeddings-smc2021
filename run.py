import pandas as pd
import numpy as np
from load_data import load_chord_data_sentences, load_key_data, all_composers, load_train_test_df
from gensim.models import Word2Vec
from lstm_window import LSTMOneHot, df_string_emb_window 
import torch
import torch.nn as nn
import torch.optim as optim
major_file = 'data/chord_vocab_major.txt'
minor_file = 'data/chord_vocab_minor.txt' 
full_vocab = 'data/chord_vocab_full.txt'




def run_minor(test_composer, num_train, df_all, min_count=50, size=5, window=2, sg=1, num_epochs=2) :
    '''For a given test composer, runs the training process of the LSTM model 'num_train' times, and creates a dataframe 
    that contains as column the run number, as a line the name of the composer as well as each chord and as values all the accuracies for a given composer, a given chord. All of this for the MINOR chords
    
    test_composer : in the format ['composer'], e.g test_composer = ['Gesualdo']
    num_train : number of times we repeat the training process
    df_all : dataframe obtained with this function for an other test composer 
    
    returns df : dataframe that concatenat the results for the test_composer, and the results already obtained in df_all
    '''
    #call the split_train_test function, to create dataframes for training, and dataframes for testing composers
    df_train, df_test, composers_without_test = load_train_test_df(all_composers, test_composer, key_mode='MINOR')
    
    #we call load_chord_data_sentences to have the list of list of chords for the training composers, as input to the W2Vec model
    data_without_test = load_chord_data_sentences(composers_without_test, key_mode='MINOR')
    
    ####We repeat the training/testing process####
    df = pd.DataFrame() #initialize a dataframe
    for i in range(num_train):
        #word2vec model
        word2vec_model = Word2Vec(data_without_test, min_count=min_count, size=size, window=window, sg=sg, iter = 500)
    
    
        #### Create dataframe containing context of a chord ####
        window_size = 2 #window to look for to define the context

        #call the df_string_emb_window funtion 
        df_embedding_train, df_string_train = df_string_emb_window(word2vec_model, dataframe = df_train, window_size = window_size)
        df_embedding_test, df_string_test = df_string_emb_window(word2vec_model, dataframe = df_test, window_size =window_size)
    
    
        ###Load the vocab###
        #vocab = pd.read_csv(minor_file, header = None) #doesn't seem to work
        vocab = pd.read_csv(full_vocab, header = None)
        
        
        ####LSTM structure####
        #create the model :
        hidden_dim = 15
        output_size = vocab.shape[0]

        #we use the window technique (input = context of the chord), we specify it by improving the input shape of the model
        lstm_window = LSTMOneHot(2 * window_size * size, hidden_dim, output_size)
    
        #define loss function and optimizer
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(lstm_window.parameters(), lr=0.001)
        print('Run '+str(i))
        #training and testing process :
        lstm_window.learn(df_embedding_train, df_string_train, loss_function, optimizer, num_epochs, vocab)
        dict_accuracy, global_accuracy = lstm_window.test(df_string_test, df_embedding_test, word2vec_model, vocab)
        
        #create the dataframe
        keys = list(dict_accuracy.keys()) #name of the chords
        keys.append('global') #we add the global accuracy

        accuracies = list(dict_accuracy.values()) #accuracies for each chord
        accuracies.append(global_accuracy) #we add the global accuracy

        arrays = [np.array([test_composer[0].replace('.csv', '') for i in range(len(keys))]), np.array(keys)] #to create a multi-index dataframe

        df_i = pd.DataFrame(data = accuracies,index=arrays, columns=['run'+str(i+1)]) #create a dataframe for the run i
        
        df = pd.concat([df, df_i], axis=1) #concatenate dataframe for each run
        
    df = pd.concat([df, df_all], axis = 0)
        
    return df



def run_major(test_composer, num_train, df_all, min_count=50, size=5, window=2, sg=1, num_epochs=2) :
    '''For a given test composer, runs the training process of the LSTM model 'num_train' times, and creates a dataframe 
    that contains as column the run number, as a line the name of the composer as well as each chord and as values all the accuracies for a given composer, a given chord. All of this for the MAJOR chords
    
    test_composer : in the format ['composer'], e.g test_composer = ['Gesualdo']
    num_train : number of times we repeat the training process
    df_all : dataframe obtained with this function for an other test composer 
    
    returns df : dataframe that concatenat the results for the test_composer, and the results already obtained in df_all
    '''
    
    #call the split_train_test function, to create dataframes for training, and dataframes for testing composers
    df_train, df_test, composers_without_test = load_train_test_df(all_composers, test_composer, key_mode='MAJOR')
    
    #we call load_chord_data_sentences to have the list of list of chords for the training composers, as input to the W2Vec model
    data_without_test = load_chord_data_sentences(composers_without_test, key_mode='MAJOR')

    ####We repeat the training/testing process####
    df = pd.DataFrame() #initialize a dataframe
    for i in range(num_train):
        #word2vec model
        word2vec_model = Word2Vec(data_without_test, min_count=min_count, size=size, window=window, sg=sg, iter = 500)
    
    
        #### Create dataframe containing context of a chord ####
        window_size = 2 #window to look for to define the context

        #call the df_string_emb_window funtion 
        df_embedding_train, df_string_train = df_string_emb_window(word2vec_model, dataframe = df_train, window_size = window_size)
        df_embedding_test, df_string_test = df_string_emb_window(word2vec_model, dataframe = df_test, window_size =window_size)
    
    
        ###Load the vocab###
        #vocab = pd.read_csv(major_file, header = None) #doesn't seem to work
        vocab = pd.read_csv(full_vocab, header = None)
        
        
        ####LSTM structure####
        #create the model :
        hidden_dim = 15
        output_size = vocab.shape[0]

        #we use the window technique (input = context of the chord), we specify it by improving the input shape of the model
        lstm_window = LSTMOneHot(2 * window_size * size, hidden_dim, output_size)

        #define loss function and optimizer
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(lstm_window.parameters(), lr=0.001)
    
        print('Run '+str(i))
        #training and testing process :
        lstm_window.learn(df_embedding_train, df_string_train, loss_function, optimizer, num_epochs, vocab)
        dict_accuracy, global_accuracy = lstm_window.test(df_string_test, df_embedding_test, word2vec_model, vocab)
        
        #create the dataframe
        keys = list(dict_accuracy.keys()) #name of the chords
        keys.append('global') #we add the global accuracy

        accuracies = list(dict_accuracy.values()) #accuracies for each chord
        accuracies.append(global_accuracy) #we add the global accuracy

        arrays = [np.array([test_composer[0].replace('.csv', '') for i in range(len(keys))]), np.array(keys)] #to create a multi-index dataframe

        df_i = pd.DataFrame(data = accuracies,index=arrays, columns=['run'+str(i+1)]) #create a dataframe for the run i
        
        df = pd.concat([df, df_i], axis=1) #concatenate dataframe for each run
        
    df = pd.concat([df, df_all], axis = 0)
        
    return df

#df_minor = pd.DataFrame() 
#df_major = pd.DataFrame() 
#for composer in all_composers:
#    df_minor = run_minor(test_composer = [composer], num_train = 10, df_all = df_minor,
#                                      min_count=50, size=5, window=2, sg=1, num_epochs=2)
#    df_major = run_major(test_composer = [composer], num_train = 10, df_all = df_major,
#                                      min_count=50, size=5, window=2, sg=1, num_epochs=2)
#    
#df_minor.to_csv('results_minor.csv')
#df_major.to_csv('results_major.csv')
        