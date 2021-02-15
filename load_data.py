import pandas as pd


# Data folders
data_folder = 'data/'
chord_folder = 'data/chord/'
key_folder = 'data/key/'

# File names
noisy_composers = ['Bach', 'Debussy' ,'Gesualdo', 'Grieg', 'Medtner', 'Schütz', 'Sweelinck', 'Wagner']
non_noisy_composers = ['Beethoven', 'Chopin', 'Corelli', 'Couperin', 'Dvorak', 'Kozeluh', 'Liszt', 'Mendelssohn', 'Monteverdi', 
                       'Mozart', 'Pleyel', 'Ravel', 'Schubert', 'Schumann', 'Tchaikovsky', 'WFBach']
all_composers = noisy_composers + non_noisy_composers


def load_key_data(composers, drop_one_worders=True):
    '''
    Read the key datasets for a list of composers, and return them in a list containing all the sentences.
    Each sentence is a list of words. Each word is a string.
    If specified by the argument, we drop the sentences with just one word.
    
    - Input.  composers: iterable of file names (example: ['Bach.csv']).
              drop_one_worders: flag to control whether to include sentences with one word.
    - Output. sentences: list of words (strings).
    '''
    
    # Dataframe that will contain all sentences
    frame = pd.DataFrame()
    
    for composer in composers:
        # We use sep='\n' to be able to first separate each line, but we'll have just one column
        data = pd.read_csv(key_folder + composer, sep='\n', names=['sentence'], header=None, error_bad_lines=False)
        # We replace the sentence column with a list of split words
        data['sentence'] = data['sentence'].str.split(',', expand=False)
        
        # If we want to remove sentences with only one word: drop all lines for which the length of the list is 1
        if drop_one_worders:
            data = data[data['sentence'].apply(lambda sent : len(sent)>1)]
            
        # Concatenate to global frame
        frame = pd.concat([frame, data], ignore_index=True)
        
    # Final list of sentences
    sentences = frame['sentence'].tolist()
    
    return sentences


def load_chord_data_sentences(composers, key_mode):
    '''
    Read the chord datasets for a list of composers, and return them in a list containing all the sentences.
    Each sentence is a list of words. Each word is a string.
    The 'key_mode' argument specifies whether we want just the sections in a major or minor key, or if we want them all.
    
    - Input.  composers: iterable of file names (example: ['Bach.csv']).
              key_mode: string to control whether to include sentences in major/minor mode, or both.
    - Output. sentences: list of words (strings).
    '''
    
    # Dataframe that will contain all sentences
    frame = load_chord_data_df(composers, key_mode)
    
    # List of sentences to return
    sentences = get_chord_sentences(frame)
    
    return sentences


def get_chord_sentences(chord_df):
    '''
    Transforms a dataframe of sentences into a list of sentences (each sentence is a list of strings).
    
    Input.  chord_df: the dataframe of sentences.
    Output. chord_sentences: the list of sentences.
    '''
    
    chord_sentences = chord_df['sentence'].tolist()
    return chord_sentences


def get_augmented_sentence(key_mode, sentence):
    '''
    Modifies each word in the sentence, by pre-pending the mode to it.
    
    - Input.  sentence: list of words (strings).
              key_mode: a word.
    - Output. aug: list of augmented words (strings).
    '''
    
    aug = []
    for word in sentence:
        aug.append(key_mode + ';' + word)
        
    return aug


def load_chord_data_df(composers, key_mode):
    '''
    Read the chord datasets for a list of composers, and return them in a dataframe containing all the sentences.
    The 'key_mode' argument specifies whether we want just the sections in a major or minor key, or if we want them all.
    
    - Input.  composers: iterable of file names (example: ['Bach.csv']).
              key_mode: string to control whether to include sentences in major/minor mode, or both.
    - Output. sentences: list of words (strings).
    '''
    
    # Add the '.csv' if not done in the input
    for index, composer in enumerate(composers):
        if '.csv' not in composer:
            composers[index] += '.csv'
    
    # Dataframe that will contain all sentences
    frame = pd.DataFrame()
    
    for composer in composers:
        # We use sep=';' to split into a 'mode' column and a 'sentence' column
        data = pd.read_csv(chord_folder + composer, sep=';', names=['key_mode', 'sentence'], header=None, error_bad_lines=False)
        # We replace the sentence column with a list of split words
        data['sentence'] = data['sentence'].str.split(',', expand=False)
            
        # Concatenate to global frame
        frame = pd.concat([frame, data], ignore_index=True)
        
    # If both modes are required, we have to modify words so as to include the mode. Else, we have to drop irrelevant sentences
    if key_mode == 'both':
        frame['sentence'] = frame.apply(lambda row : get_augmented_sentence(row['key_mode'], row['sentence']), axis=1)
    else:
        frame = frame[frame['key_mode'] == key_mode.upper()]
        
    return frame


def load_train_test_df(composers, test_composers, key_mode):
    '''
    Remove the composers specified by 'test_composers' from the list of composers, and returns a train and a test dataframe.
    
    Input.  composers: list of all composers
            test_composers: composer(s) to drop and use for testing (e.g. ['Bach'])
    Output. train_df: dataframe for all composers except the test composers
            test_df: dataframe only for the test composers
            train_composers: the list of all composers without the test composers
    
    '''
    
    # Create the list of composers for training:
    train_composers = [comp for comp in composers if comp not in test_composers]
            
    # Create the training and the tsting dataframes
    train_df = load_chord_data_df(train_composers, key_mode)
    test_df = load_chord_data_df(test_composers, key_mode)
    
    return train_df, test_df, train_composers


def load_train_test_sentences(composers, test_composers, key_mode):
    '''
    Remove the composers specified by 'test_composers' from the list of composers, and returns a train and a test 
    list of sentences.
    
    Input.  composers: list of all composers
            test_composers: composer(s) to drop and use for testing (e.g. ['Bach'])
    Output. train_sentences: list of sentences for all composers except the test composers
            test_sentences: list of sentences only for the test composers
            train_composers: the list of all composers without the test composers
    
    '''
    
    train_df, test_df, train_composers = load_train_test_df(composers, test_composers, key_mode)
    
    return get_chord_sentences(train_df), get_chord_sentences(test_df), train_composers