'''Definition of the LSTM network, training and testing process'''
import torch
import torch.nn as nn
import pandas as pd
import numpy as np


def df_string_emb(word2vec_model, dataframe):
    '''This function creates the dataframes containing the input and the output of the LSTM model. In this function the input
    is the previous chord and the label is the current chord. 
    
    Input : -word2vec_model : The Word2Vec model trained, to access the embedding coordinates
            - dataframe : the dataframe containing the sentences
            
    Output : - df_embedding : dataframe containing the embedding coordinates for both X the input, and Y the label
             - df_string : dataframe containing the chords as strings for both X the input, and Y the label
             
    '''
    #create 2 dataframes : 1 containing the coordinates in the embedding space, one containing the strings
    emb_X = [] #contains coordinates of the current word in the embedding space
    emb_Y = [] #contains coordinates of the next word in the embedding space

    chord_X = [] #contains the current chord
    chord_Y = [] #contains the next chord

    for index, row in dataframe.iterrows(): #for each sentence
        sentence = row['sentence'] #sentence
        for i in range(1,len(sentence)): #for each element of that sentence (we start from 1, to use 'i-1' and not 'i', but this is arbitrary)
            current_chord = sentence[i-1] 
            next_chord = sentence[i] 
            
            if current_chord in word2vec_model.wv.vocab.keys() and next_chord in word2vec_model.wv.vocab.keys(): #if both words are in the vocabulary
                #we add the embedding coordinates in the lists create at the beginning
                emb_X.append(word2vec_model.wv[current_chord]) 
                emb_Y.append(word2vec_model.wv[next_chord])
                
                #we add the strings
                chord_X.append(current_chord) 
                chord_Y.append(next_chord)

    #create the dataframe containing the embedding coordinates, and the dataframe containing the strings
    df_embedding = pd.DataFrame(list(zip(emb_X, emb_Y)), columns=['X', 'Y']) 
    df_string = pd.DataFrame(list(zip(chord_X, chord_Y)), columns=['X', 'Y'])
    
    return df_embedding, df_string



def df_string_emb_window(word2vec_model, dataframe, window_size):
    '''This function creates the dataframes containing the input and the output of the LSTM model. In this function the input
    is the chords that appear within a window_size of the current chord, and the label is the current chord. 
    
    Input : -word2vec_model : The Word2Vec model trained, to access the embedding coordinates
            - dataframe : the dataframe containing the sentences
            - window_size : window to consider (if window_size = 2, we take the 2 previous chords AND the 2 next chords as input)
            
    Output : - df_embedding : dataframe containing the embedding coordinates for both X the input, and Y the label
             - df_string : dataframe containing the chords as strings for both X the input, and Y the label
             
    '''
    #create 2 dataframes : 1 containing the coordinates in the embedding space, one containing the strings
    emb_X = [] #contains coordinates of the current word in the embedding space
    emb_Y = [] #contains coordinates of the next word in the embedding space

    chord_X = [] #contains the current chord
    chord_Y = [] #contains the next chord

    for index, row in dataframe.iterrows():
        sentence = row['sentence']
        for i in range(window_size, len(sentence) - window_size):
            focus_chord = sentence[i]
            chords_before = sentence[i-window_size : i]
            chords_after = sentence[i+1: i + 1 + window_size]
            #we gather the chords before and the chords after in an entity named 'context'
            context = chords_before + chords_after 
            
            #next we check if all chords in the context is in the model vocabulary :
            context_vocab = True #initialize a boolean
            for chord in context :
                if chord not in word2vec_model.wv.vocab.keys() :
                    context_vocab = False #the boolean becomes False if any of the chord in the context is not in the vocabulary
                    
            #we fill the list of embedding coordinates of the context : 
            emb_context = np.array([], dtype = 'float32')
            if context_vocab == True : #If all words in the context are in the vocabulary of the wor2vec model
                for chord in context :
                    emb_context = np.concatenate((emb_context, word2vec_model.wv[chord])) #we concatenate coordinates
                
            #If all words in the context + the focus chord are in the vocabulary of the wor2vec model, we add the embedding coordinates and the strings in the lists
            if context_vocab == True and focus_chord in word2vec_model.wv.vocab.keys() : 
                #Add the embedding coordinates
                emb_X.append(emb_context) #Add the embedding coordinates of the words in the context
                emb_Y.append(word2vec_model.wv[focus_chord])
                
                #Add the strings
                chord_X.append(context)
                chord_Y.append(focus_chord)

    #create the dataframe containing the embedding coordinates, and the dataframe containing the strings
    df_embedding = pd.DataFrame(list(zip(emb_X, emb_Y)), columns=['X', 'Y'])
    df_string = pd.DataFrame(list(zip(chord_X, chord_Y)), columns=['X', 'Y'])
    
    return df_embedding, df_string


class LSTMOneHot(nn.Module):
    '''
    Class implementing an LSTM-based NN that predicts a chord given some input. This input can either be the current chord, or the context of this chord. This choice is made in the call of this class, in the main file or notebook.The output of this model is a one-hot vector of the same size as the vocabulary, with a one at the place of the chord, and zeros anywhere else.
    '''

    def __init__(self, input_size, hidden_dim, output_size ):
        super(LSTMOneHot, self).__init__()

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_size, hidden_dim)
        
        #Linear maaping between the hidden layer and the output layer (output layer : vector of the same size as the vocabulary size)
        self.linear = nn.Linear(hidden_dim, output_size)
        
        #Softmax to transform values into probabilities
        self.soft = nn.Softmax(dim = 0)
        

    def forward(self, coordinates):
        '''Forward pass
        Input : - coordinates : coordinates in the embedding space of the chord (or chordS in case of the window technique)
        '''
        out, _ = self.lstm(coordinates.view(1, 1, -1)) #apply the lstm layer
        out = out.squeeze() #squeeze the vector to remove 1-D dimensions for later operations
        out = self.linear(out) #apply the linear mapping
        out = self.soft(out) #apply the softmax operation
        return out
    
    
    def learn(self,df_embedding_train, df_string_train, loss_function, optimizer, num_epochs, vocab):
        '''Training process : learn the model
    Input : - df_embedding_train : dataframe containing the embedding coordinates of the chords for the Input X and the Output Y, for the training composers
            - df_string_train : dataframe containing the strings of the chords for the input X and the output Y, for the training composers
            - model : model to learn : LSTM in our case
            - loss_function : loss function to use
            - optimizer : optimizer to use
            - num_epochs : number of epochs
            - vocab : list of the vocabulary      
        '''
        #define the input and output tensors :
        tensor_X = torch.tensor(list(df_embedding_train['X'].values))
        tensor_Y = torch.tensor(list(df_embedding_train['Y'].values))

        #strings
        next_strings = list(df_string_train['Y'].values)


        #training : for each element of the input tensor 'tensor_X', compare the output of the model and the real chord, and learn
        for i in range(num_epochs):

            running_loss = 0.0 #Loss on the current slice of iterations
            for ind, chord in enumerate(tensor_X): #for each element of the input tensor
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                self.zero_grad()

                # Step 2 - compute model predictions and loss

                real_chord = next_strings[ind] 
                #real one-hot vector, with a one at the place of the chord, and zeros anywhere else
                one_hot_prob = torch.tensor(np.where(vocab[0].values == real_chord, 1.0, 0.0), dtype = torch.float)

                #predicted one-hot vector
                output = self(chord)

                #loss
                loss = loss_function(output.squeeze(), one_hot_prob.squeeze()) 

                # Step 3 - do a backward pass and a gradient update step
                optimizer.zero_grad()  
                loss.backward()
                optimizer.step()

                running_loss += loss.item() #add the loss to the running loss
                if ind % 10000 == 9999: #print every 10000 mini_batches
                    print('Epoch : {}/{}, Iteration : {}/{} , Loss : {}'.format(i + 1, num_epochs, ind +1, len(tensor_X), running_loss/10000))
                    running_loss = 0.0 #reset the running loss, so that the next print is only for the slice of 5000 iterations
                
                
    def test(self,df_string_test, df_embedding_test, word2vecmodel, vocab):
        '''
        Testing process : evaluate the model
        Input : - df_embedding_test : dataframe containing the embedding coordinates of the chords for the Input X and the Output Y, for the testing composer
            - df_string_train : dataframe containing the strings of the chords for the input X and the output Y, for the testing composer
            - model : model to learn : LSTM in our case
            - word2vecmodel : word2vec model trained, to compute similarities
            - vocab : list of the vocabulary
            
    '''
        similarities = [] #list that will contain the similarity between the real next chord and the predicted next chord
        correct_pred = 0 #counter on the global number of correct predictions
        dict_occurence = {} #dictionnary that will contain the number of occurence per chord
        dict_correct_pred = {} #dictionnary that will contain the number of correct predictions for each chord
        dict_accuracy = {} #dictionnary that will contain the accuracy per chord : accuracy = (nb of occurence)/(nb of correct predictions)

        for ind in range(df_string_test.shape[0]): #for all indexes in the dataframe
            #current chords
            current_chord_emb = torch.from_numpy(df_embedding_test['X'][ind]) #coordinates in the embedding space

            #prediction :
            output_model = self(current_chord_emb)
            output_model = output_model.detach().numpy() #better form for later operations

            max_index = np.argmax(output_model) #take the index of the maximum element of this output

            predicted_chord = vocab.loc[max_index][0] #take the predicted chord from the vocabulary

            #real chord
            real_next_chord = df_string_test['Y'][ind]

            #Add the chord to the dictionnary containing the number of occurence per chord
            if real_next_chord not in dict_occurence: #if it's not in the dictionnary yet, we initialize the value to 1
                dict_occurence[real_next_chord] = 1
            else:
                dict_occurence[real_next_chord] += 1

            #Also add it to the dictionnary containing the number of correct prediction per chord, if it's not already the case
            if real_next_chord not in dict_correct_pred:
                    dict_correct_pred[real_next_chord] = 0
            
            #If the prediction is good : increment the dictionnary of correct prediction for that chord, and the global number of correct predictions :
            if real_next_chord == predicted_chord :
                dict_correct_pred[real_next_chord] += 1
                correct_pred +=1


            #Compute the similarity between the real chord and the predicted chord, according to the word2vec model
            similarity = word2vecmodel.wv.similarity(real_next_chord, predicted_chord)
            similarities.append(similarity)
            
            #uncomment the following if you want to see the chords for which the similarity is low
            #if similarity<0.50 : 
                #print('Similarity : {:.2f}, \nreal chord : {}, \npredicted chord : {}\n'.format(similarity, real_next_chord, predicted_chord))



        #sort the dictionnary of number of occurence :
        dict_occurence = {chord : nb for chord, nb in sorted(dict_occurence.items(), key=lambda item: item[1], reverse = True)}

        #print the results of global accuracies and mean of similarities between the real chord and the predicted chord
        global_accuracy = correct_pred/df_string_test.shape[0]
        #print('\nGlobal Accuracy : {:.2%}'.format(global_accuracy))
        #print('Mean of similarities : {}'.format(np.mean(similarities)))

        
        #For each chord in the dictionnary, add the accuracy per chord by filling the dictionnary of accuracies, and print accuracies per chord
        for chord in dict_occurence :
            dict_accuracy[chord] = dict_correct_pred[chord]/dict_occurence[chord]
            #print('Number of {} : {}, Prediction accuracy : {:.2%}'.format(chord, dict_occurence[chord], dict_accuracy[chord]))
            
        #We return this dictionnary plus the global accuracy
        return dict_accuracy, global_accuracy