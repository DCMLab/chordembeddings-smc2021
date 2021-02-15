import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from lstm_window import df_string_emb

def entropy_before(word2vec_model, dataframe_test):
    '''Computes the entropy per chord for the testing composer. In this function we compute the proportion of uses per chord, before a given chord. Example : before the I:MAJ, we have in 30% of cases a V:MAJ,...
    Then, based on this proportions that can be viewed as probabilities, we compute the entropy for each chord, this represents the surprise of the use of a chord. More this entropy is high, more te probability for a chord appearing before is ditributed on all chords.
    Input : - word2vec_model : word2vec model
            - dataframe_test : dataframe that contains the sentences for the test composer
    Output : - combinations : dataframe in which the (i,j) element represent the probability that chord j appear before chord i
             - entropy : dataframe : entropy for each chord
    '''
    #load the dataframe of curerentchord-next chord : this dataframe has 2 features : X (input) and Y (label)
    df_embedding_test, dataframe_string = df_string_emb(word2vec_model = word2vec_model, dataframe = dataframe_test)
    
    #table of probability that, given a chord, what will be the chord before
    combinations = pd.DataFrame()
    for chord_Y in dataframe_string['Y'].unique(): #for all given chord
        dict_chord = {} #contains Y values
        df_chord = dataframe_string[dataframe_string['Y'] == chord_Y] #filter the dataframe with lines that contains chord Y
        for ind, row in df_chord.iterrows():
            chord_X = row['X']
            #we count the number of chords before for each chord
            if chord_X not in dict_chord:
                dict_chord[chord_X] = 1
            else :
                dict_chord[chord_X] += 1
            combinations.loc[chord_Y, chord_X] = dict_chord[chord_X]/df_chord.shape[0] #proportion of previous chord occuring
            
    #Entropy : we apply the formula
    entropy = pd.DataFrame()
    for chord in combinations.index:
        entropy.loc[chord, 'entropy'] = -((combinations.loc[chord].apply(lambda x : np.log(x))) * (combinations.loc[chord])).sum()
   
    return combinations, entropy

def entropy_after(word2vec_model, dataframe_test):
    '''Computes the entropy per chord for the testing composer. In this function we compute the proportion of uses per chord, after a given chord. Example : after the I:MAJ, we have in 30% of cases a V:MAJ,...
    Then, based on this proportions that can be viewed as probabilities, we compute the entropy for each chord, this represents the surprise of the use of a chord. More this entropy is high, more te probability for a chord appearing after is ditributed on all chords. Same than last function, but with chord_X and chord_Y inversed
    Input : - word2vec_model : word2vec model
            - dataframe_test : dataframe that contains the sentences for the test composer
    Output : - combinations : dataframe in which the (i,j) element represent the probability that chord j appear after chord i
             - entropy : dataframe : entropy for each chord
    '''
    #load the dataframe of curerntchord-next chord :
    df_embedding_test, dataframe_string = df_string_emb(word2vec_model = word2vec_model, dataframe = dataframe_test)
    
    combinations = pd.DataFrame()
    for chord_X in dataframe_string['X'].unique():
        dict_chord = {} #contains Y values
        df_chord = dataframe_string[dataframe_string['X'] == chord_X]
        for ind, row in df_chord.iterrows():
            chord_Y = row['Y']
            if chord_X not in dict_chord:
                dict_chord[chord_Y] = 1
            else :
                dict_chord[chord_Y] += 1
            combinations.loc[chord_X, chord_Y] = dict_chord[chord_Y]/df_chord.shape[0] #proportion of next chord occuring

    #Entropy
    entropy = pd.DataFrame()
    for chord in combinations.index:
        entropy.loc[chord, 'entropy'] = -((combinations.loc[chord].apply(lambda x : np.log(x))) * (combinations.loc[chord])).sum()
   
    return combinations, entropy

def vizualize(dict_accuracy, dataframe_test, word2vec_model, test_composer, dict_df) :
    '''Plots Accuracy vs Entropy for all composers contained in dict_df, and add the test composer in it
    Input : - dict_accuracy : the dictionnary returned by the testing process, contains accuracy per chord for the test composer
            - dataframe_test : dataframe containing sentences for the test composer
            - word2vec_model : word2vec model
            - test_composer : test compsoer, e.g ['Bach']
            
    '''
    #compute accuracy :
    combinations, entropy = entropy_before(word2vec_model, dataframe_test)
    
    #create dataframe for the accuracies
    df_accuracy = pd.DataFrame()
    df_accuracy['accuracy'] = dict_accuracy.values()
    df_accuracy.index = dict_accuracy.keys()
    
    #join the 2 to have a dataframe containing both accuracy and entropy
    data = entropy.join(df_accuracy)
    
    #add this dataframe to the dictionnary dict_df to save results for future plots
    dict_df[(test_composer[0])] = data
    
    #plot
    fig, ax3 = plt.subplots( figsize = (10,8))
    plt.title('Accuracy and entropy in the case of MAJOR chords')
    plt.xlabel('Entropy')
    plt.ylabel('Accuracy')
    
    for composer in dict_df.keys():
        plt.scatter(x= dict_df[composer]['entropy'], y = dict_df[composer]['accuracy'], label = composer)
        
    plt.legend(loc = 'upper left')
    plt.grid(axis = 'x')
    
    return dict_df
    