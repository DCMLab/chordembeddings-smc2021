import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from load_data import load_chord_data_sentences, load_key_data, all_composers, load_chord_data_df



#We define the style of the composers :
ren_baroque = ['Gesualdo.csv', 'Sweelinck.csv', 'Monteverdi.csv', 'Schütz.csv', 'Corelli.csv', 'Couperin.csv',
              'Bach.csv']

classical = ['WFBach.csv', 'Mozart.csv', 'Kozeluh.csv', 'Beethoven.csv', 'Schubert.csv', 'Pleyel.csv']

romantic = ['Mendelssohn.csv', 'Chopin.csv', 'Schumann.csv', 'Wagner.csv', 'Liszt.csv', 'Tchaikovsky.csv', 'Dvorak.csv',
            'Grieg.csv']

#We create a datframe containing the year of birth and year of death information
from load_data import all_composers

for index, composer in enumerate(all_composers):
    all_composers[index] = composer.replace('.csv', '')
    
print(all_composers)
Dict_dates = {'composer' : all_composers,
              'date_of_birth' : [1685, 1862, 1566, 1843, 1880, 1585, 1562, 1813, 1770, 1810, 1653, 1668, 1841, 1747, 1811, 1809,
                                 1567, 1756, 1757, 1875, 1797, 1810, 1840, 1710],
              'date_of_death' : [1750, 1918, 1613, 1907, 1951, 1672, 1621, 1883, 1827, 1849, 1713, 1733, 1904, 1818, 1886, 1847,
                                 1643, 1791, 1831, 1937, 1828, 1856, 1893, 1784]}

dates_df = pd.DataFrame(data = Dict_dates).sort_values(by='date_of_death', ascending = True)



def nb_chords(all_composers, ren_baroque, classical, romantic):
    '''This function computes the number of chords used by composers, the number of chords per style and the number of
    MAJOR chords.
    Input : - all_composers : list of all composers
            - ren_baroque : list of all composers that have a renaissance or a baroque style
            - classical : list of all composers that have a classical style
            - romantic : list of all composers that have a romantic style
    Output : - df_nb : dataframe that contains the composer, the number of chords and the number of major chords
             - df : dataframe that contains
    '''
    #initialization of the dataframe and the dictionnaries
    df = pd.DataFrame(index = ['ren_baroque', 'classical', 'romantic'])
    dict_comp = {} #will contain the number of chords for each composer
    dict_maj = {} #will contain the number of major chords for each composer
    
    #initialize the count of renaissance or baroque composers : useful for the assignment in the dataframe per style
    count_ren_baroque = 0
    
    #for each composer
    for composer in all_composers:
        composer = [composer] #string of the composer
        df_composer = load_chord_data_df(composer, key_mode = 'both') #load the dataframe of sentences for this composer
        
        composer_name = composer[0].replace('.csv','') #remove the '.csv'
        
        #initialization of the count
        nb_chords = 0
        nb_of_chords_in_maj = 0
        
        for index, row in df_composer.iterrows(): #for each sentence
            sentence = row['sentence'] #sentence
            if row['key_mode'] == 'MAJOR': #if the sentence is from a MAJOR sentence
                nb_of_chords_in_maj += len(sentence) #nb of MAJOR chords += nb of chords from this MAJOR sentence
            nb_chords+=len(sentence) #nb of chords += nb of chords from this sentence
            
        #add it to the dictionnaries
        dict_maj[composer_name] = nb_of_chords_in_maj
        dict_comp[composer_name] = nb_chords
        
        #add the number of chords for each style, we save the number of composers from classical, to insert at the correct position the classical
        if composer[0] in classical :
            df.insert(count_ren_baroque, composer[0], [0, nb_chords, 0]) #add the corresponding column
            
        if composer[0] in ren_baroque :
            count_ren_baroque +=1
            df.insert(0, composer[0], [nb_chords, 0, 0]) #add the corresponding column
            
        if composer[0] in romantic :
            df.insert(df.shape[1], composer[0], [0, 0, nb_chords]) #add the corresponding column
    
    #define the dataframe that contains the number of chords per composer, and the number of MAJOR chords per composer
    df_nb = pd.DataFrame(data = dict_comp.items(), columns=['composer', 'nb_chords']).set_index('composer')
    df_maj = pd.DataFrame(data = dict_maj.items(), columns=['composer', 'nb_chords_maj']).set_index('composer')
    #join these 2 dataframes
    df_nb = df_nb.join(df_maj).rename(columns={"index": "composer"}).reset_index()
    
    #remove '.csv' from the column names of the dataframe per style
    df.columns = df.columns.str.replace('.csv', '')
    
    #join the year_of_death and the year_of_birth features
    df_nb = df_nb.merge(dates_df, on='composer').sort_values(by = 'date_of_death',ascending = True)
    
    return df_nb, df


def list_all_chords(all_composers) :
    '''returns the list of all different chords used for all composers'''
    df_all_composers = load_chord_data_df(all_composers, key_mode = 'both') #load the data
    chords=[] #list that will contain the chords
    for index, row in df_all_composers.iterrows(): #for each sentence
        sentence = row['sentence'] #sentence
        for i in range(len(sentence)):
            chords.append(sentence[i]) #we add the chord

    df_all_chords = pd.DataFrame(chords) #dataframe containing all different chords
    list_of_chords = df_all_chords[0].unique() #we select the unique ones
    return list_of_chords


def nb_occurence_chord(all_composers):
    '''Find the number of occurence of each chord in the data'''
    df_all_composers = load_chord_data_df(all_composers, key_mode = 'both') #load the data for all composers
    list_of_chords = list_all_chords(all_composers) #find the list of different chords
    dict_chords={} #will contain the number of occurence for each chord
    
    for index, row in df_all_composers.iterrows():
        sentence = row['sentence'] #sentence
        for chord in sentence:
            if chord not in dict_chords : #if the chord is not in the dictionnary yet : add it
                dict_chords[chord] = 1
            else :
                dict_chords[chord] += 1
                
    #define the dataframe, sort the dataframe in a decreasing order
    df_occurence = pd.DataFrame(dict_chords.items(), columns=['chord', 'number_of_occurences']).sort_values(
        by = 'number_of_occurences',ascending = False, ignore_index = True) 
    return df_occurence


def df_heatmap(composers, key_mode = 'both', treshold = 500) :
    '''Construct the dataframe for plotting the heatmap
    Input : - composers : list of composers we want to study
            - key_mode : key mode to study : 'MAJOR' or 'MINOR'
            - treshold : number of minimum appearance of a chord in the entire dataset, to put in the dataframe
    Output : - df_comp_chord : dataframe that has as rows the chords, as columns the composers and as values the number of occurence for a given chord and a given composer
    '''
    list_of_chords = list_all_chords(composers) #find the list of different chords
    list_comp = []
    for composer in composers:
        composer = [composer]
        df_composer = load_chord_data_df(composer, key_mode = 'both') #load the data for this composer
        df_composer = df_composer[df_composer['key_mode']==key_mode] #filter with the key mode
        composer_name = composer[0].replace('.csv','') #to not have '.csv' in the name of the composer
        
        #We create a dictionnary and intialize all its value to 0 for each chord
        dict_comp={}
        for chord in list_of_chords:
            dict_comp[chord] = 0
        #Add the number of occurence of chords in the dictionnary, and then add this dictionnary in a list
        for index, row in df_composer.iterrows():
            sentence = row['sentence'] #sentence
            for chord in sentence:
                dict_comp[chord] += 1
        list_comp.append(dict_comp)
        
    #define the dataframe
    df_comp_chord = pd.DataFrame(list_comp, index = composers).transpose() #create the dataframe
    df_comp_chord.columns = df_comp_chord.columns.str.replace('.csv','') #remove the '.csv'

    #we only use chords that are used more than treshold times in total for all composers
    df_comp_chord = df_comp_chord.loc[(df_comp_chord.sum(axis = 1)>treshold).values]
    df_comp_chord = df_comp_chord / df_comp_chord.sum()
    
    return df_comp_chord