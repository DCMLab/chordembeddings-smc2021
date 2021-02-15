import numpy as np


def dimred_keyed_vec(wv, reduce_dim, comps_to_keep=[0,1]):
    '''
    Applies the provided dimensionality-reduction function to the provided keyed vector, word-wise.
    
    Input.  wv: a KeyedVector.
            reduce_dim: a function for dimensionality reduction, taking as input a (n_samples * n_features) array-like.
            comps_to_keep: the components to keep.
    Output. wv_red: a dictionary mapping the same words as wv to dimensionality-reduced vectors.
    '''
    
    # Build (n_samples * n_features) matrix
    X = wv_to_coordinates(wv) 
    
    # Reduce dimensionality
    X_red = reduce_dim(X)
    
    # Reconstruct a dictionary out of X_red (the order is preserved)
    wv_red = {}
    for i, word in enumerate(wv.vocab.keys()):
        wv_red[word] = X_red[i][comps_to_keep]
    
    return wv_red


def cluster_keyed_vec(wv, cluster, relabel=True):
    '''
    Applies the provided clustering method to the provided keyed vector, word-wise. 
    If specified, it re-labels the clusters so that their cardinalities are decreasing in the index.
    It also returns a dictionary mapping labels to clusters (lists of chords).
    
    Input.  wv: a KeyedVector.
            cluster: a clustering object.
            relabel: a flag signalling whether or not we should relabel.
    Output. wv_clus: a dictionary mapping the same words as wv to cluster identifiers.
            relabelling: a dictionary mapping old labels to new labels.
            clusters: dictionary mapping labels to clusters (lists of chords).
    '''
    
    # Build (n_samples * n_features) matrix
    X = wv_to_coordinates(wv) 
    
    # Form clusters
    cluster.fit(X)
    
    # Reconstruct a dictionary (the order is preserved)
    wv_clus = {}
    labels = cluster.labels_
    relabelling = None
    if relabel:
        labels, relabelling = relabel_decreasing(cluster.labels_)
    for i, word in enumerate(wv.vocab.keys()):
        wv_clus[word] = labels[i]
    
    # Dictionary of clusters
    clusters = {}
    for word in wv_clus:
        label = wv_clus[word]
        if label in clusters:
            clusters[label].append(word)
        else:
            clusters[label] = [word]
    
    if relabel:
        return wv_clus, clusters, relabelling
    return wv_clus, clusters


def wv_to_coordinates(wv):
    '''
    Function that takes a KeyedVector, and returns a matrix of coordinates, 
    containing for each word its coordinates in the embedding space
    
    Input.  model created by word2vec, size of the embedding space
    Output. Array of size (nb of vocab words in word2vec) x (size of the embedding space), containing the coordinates
            of each word in this embedding space
    '''
    vocab = list(wv.vocab.keys()) #chords
    size = wv[vocab[0]].shape[0]
    coordinates = np.ones((len(vocab), size)) #initialisation of the matrix containing coordinates in the embedding space
    for indx, chord in enumerate(vocab): #For each chord
        coordinates[indx,:] = wv[chord] #we add the coordinates
    return coordinates


def relabel_decreasing(labels):
    '''
    Takes an array of labels, and returns a relabelling such that the number of occurrences of 0 is at least that of 1, and so on.
    
    Input.  labels: the array of labels.
    Output. new_labels: the array of new labels.
            relabelling: the dictionary mapping old labels to new labels.
    '''
    
    # Dictionary counting the occurrences
    occur = {}
    for label in labels:
        if label in occur:
            occur[label] += 1
        else:
            occur[label] = 1
    
    #print(occur)
    relabelling = {old_label : pos for pos, (old_label, occ) in enumerate(sorted(occur.items(), reverse=True, 
                                                                                key=lambda pair : pair[1]))}
    new_labels = [relabelling[label] for label in labels]
    
    return new_labels, relabelling