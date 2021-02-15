import torch
import torch.nn as nn


class LSTMPredictor(nn.Module):
    '''
    Class implementing an LSTM-based NN that predicts a chord given the past chords.
    The context is given as a certain number of chords surrounding the target chord, expressed in embedding coordinates.
    The prediction is given as a discrete distribution over the chords in the vocabulary.
    '''

    def __init__(self, w2v_model, hidden_dim):
        super(LSTMPredictor, self).__init__()

        self.wv = w2v_model.wv
        self.embed_dim = self.wv.vectors.shape[1]
        self.hidden_dim = hidden_dim
        
        # The LSTM takes word embeddings as inputs, and outputs hidden states with dimensionality hidden_dim.
        self.lstm = nn.LSTM(self.embed_dim, hidden_dim)
        
        # Map hidden states to logits
        self.linear = nn.Linear(hidden_dim, len(self.wv.vocab.keys()))
        

    def forward(self, chord, prev_state):
        embed = torch.tensor(self.wv[chord])
        hidden, next_state = self.lstm(embed.view(1, 1, -1))
        logits = self.linear(hidden)[0][0] # hidden is 3-D
        return logits, next_state
    
    
    def get_zero_state(self):
        '''
        Returns the initial state of the LSTM layer (to be called at the beginning of every sentence).
        '''
        return torch.zeros(1, 1, self.hidden_dim), torch.zeros(1, 1, self.hidden_dim)
    
    
    def clean_sentences(self, sentences):
        '''
        Rids the sentences of rare words that don't appear in the Word2Vec dictionary.
        '''
        return [[chord for chord in sentence if chord in self.wv.vocab] for sentence in sentences]
    
    
    def learn(self, train_sentences, optimiser, num_epochs, iters_to_log=5000):
        '''
        Training of the NN.
        '''
        
        train_sentences = self.clean_sentences(train_sentences)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            print('Starting epoch', epoch)
            
            n_iters = 0
            cumulative_loss = 0
            for sentence in train_sentences:
                # Restart the state at every sentence
                state_h, state_c = self.get_zero_state()
                for index, curr_chord in enumerate(sentence):
                    # Use curr_chord as new information to predict next_chord
                    if index+1 == len(sentence):
                        break
                    next_chord = sentence[index+1]
                    
                    logits, (state_h, state_c) = self(curr_chord, (state_h, state_c))
                    # CrossEntropyLoss wants minibatches
                    logits_2d = logits.view(1, -1)
                    target_1d = torch.tensor([self.wv.vocab[next_chord].index])
                    loss = criterion(logits_2d, target_1d)
                    cumulative_loss += loss.item()
                    
                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()
                    
                    n_iters += 1
                    if n_iters % iters_to_log == 0:
                        print('Iteration', n_iters, ': average loss =', cumulative_loss/iters_to_log)
                        cumulative_loss = 0
                        
            print('Closing epoch', epoch, '\n')
            
        return
    

    def test(self, test_sentences):
        '''
        Testing of the NN. Returns the accuracy, both global and detailed by chord, beside the occurrences detailed by chord.
        '''
        test_sentences = self.clean_sentences(test_sentences)

        occurrences_total = 0
        correct_preds_total = 0
        occurrences_by_chord = {}
        correct_preds_by_chord = {}

        # Compute statistics
        for sentence in test_sentences:
            # Restart the state at every sentence
            state_h, state_c = self.get_zero_state()
            for index, curr_chord in enumerate(sentence):
                # Use curr_chord as new information to predict next_chord
                if index+1 == len(sentence):
                    break
                next_chord = sentence[index+1]

                logits, (state_h, state_c) = self(curr_chord, (state_h, state_c))
                pred_chord = self.wv.index2word[logits.argmax()]

                if next_chord not in occurrences_by_chord:
                    occurrences_by_chord[next_chord] = 0
                    correct_preds_by_chord[next_chord] = 0
                occurrences_by_chord[next_chord] += 1
                occurrences_total += 1
                if next_chord == pred_chord:
                    correct_preds_total +=1
                    correct_preds_by_chord[next_chord] += 1

        # Scale to get accuracies
        accuracy_total = correct_preds_total / occurrences_total
        accuracy_by_chord = {chord : correct_preds_by_chord[chord]/occurrences_by_chord[chord] for chord in occurrences_by_chord}

        return accuracy_total, accuracy_by_chord, occurrences_by_chord