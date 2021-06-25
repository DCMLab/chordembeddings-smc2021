# A Historical Analysis of Harmonic Progressions Using Chord Embeddings

This repository contains the data and code for our upcoming 2021 SMC paper, "Chord Embeddings of Harmonic Progressions Reveal Historical Trends".

If you use it, please cite it:

```
@inproceedings{Anzuoni:21,
  title={Chord Embeddings of Harmonic Progressions Reveal Historical Trends},
  author={Anzuoni, Elia and Ayhan, Sinan and Dutto, Federico and McLeod, Andrew and Moss, Fabian C. and Rohrmeier, Martin},
  booktitle={Sound and Music Computing Conference {(SMC)}},
  year={2021},
  pages={284--291}
}
```

## Abstract

This study focuses on the exploration of the possibilities arising from the application of an NLP word-embedding method (Word2Vec) to a large corpus of musical chord sequences, spanning multiple musical periods. First, we analyse the clustering of the embedded vectors produced by Word2Vec in order to probe its ability to learn common musical patterns. We then implement an LSTM-based neural network which takes these vectors as input with the goal of predicting a chord given its surrounding context in a chord sequence. We use the variability in prediction accuracy to quantify the stylistic differences among various composers in order to detect idiomatic uses of some chords by some composers. The historical breadth of the corpus used allows us to draw some conclusions about broader patterns of changing chord usage across musical periods from Renaissance to Modernity.


# Description of the Github repository

The code is contained in the top level.  
The folder [data](data) contains the datasets for this project.   
The folder [papers](papers) contains some reference academic papers.  
The folder [extra_results](extra_results) contains additional pictures and documents supporting the results outlined in the paper.  

Following is the list of dependencies, and description of the code and the datasets.

## Dependencies

The project depends on the following libraries:

[NumPy](https://numpy.org/): version 1.19.2  
[Matplotlib](https://matplotlib.org/): version 3.3.2  
[SciPy](https://www.scipy.org/): version 1.5.2  
[seaborn](https://seaborn.pydata.org/): version 0.11.0  
[pandas](https://pandas.pydata.org/): version 1.1.3  
[PyTorch](https://pytorch.org/): version 1.7.0  
[Gensim](https://pypi.org/project/gensim/): version 3.8.0  
[scikit-learn](https://scikit-learn.org/stable/): version 0.23.2  

## [Results](results)

This folder contains the results described in the paper.

### Dendrograms
There are two dendrograms, one for major sections (also shown in the paper), and another for minor sections. They both show how clustering can isolate some well-known tonal relationships between chords.

### Accuracies
There are two csv files for the accuracies detailed by composer and chord, one for major and one for minor sections. We ran the algorithm 10 times, to be able to compute the mean and standard deviation of accuracies, per composer and per chord.

## [Data](data)

The [data](data/) folder contains two distinct dataset: a *key* dataset (containing the sequence of keys traversed by each piece of each composer) and a *chord* dataset (containing the sequence of chords traversed by each section of each composer). In this project, we only used the chord dataset.  

### [Top level](data)

The 6 files at the top level are almost unused. They provide the vocabulary of distinct words contained in a dataset: for example, [chord_vocab_minor.txt](data/chord_vocab_minor.txt) contains the list of all unique chords used by composers in minor sections.

### [Chord dataset](data/chord)

This dataset consists of 24 csv files, one for each composer. Each file consists of several lines, each representing a musical section. Each section consists of a key mode (e.g. MINOR), followed by a semicolon and by a list of chords in relative notation. Files contain both major and minor sections, so it is up to us to separate them.

### [Key dataset](data/key)

This dataset is similar to the chord dataset, but is not used in this project.

## [Code](.)

The code is organised in notebooks and Python files. Most of the functions are implemented in the Python files, so that notebooks can be kept at a relatively high level.

### [EDA.ipynb](EDA.ipynb)

This notebook implements the exploratory data analysis of the dataset.  
It plots the total number of (non-unique) chords - possibly split between major and minor sections - used by each composer - possibly grouped by artistic current.  
It also produces a barplot indicating the most popular chords - possibly split between major and minor sections - for all composers.
Finally, it plots a heatmap depicting, for a pair (`composer`, `chord`), the proportion of occurrences of `chord` in the whole corpus of `composer`.

### [Clustering.ipynb](Clustering.ipynb)

This notebook implements the clustering task.  
After loading the data (all the sections in the same key mode), it applies Word2Vec on it with suitable parameters.  
Two approaches are then attempted at clustering: K-means and hierarchical clustering.  
K-means was discarded for the reasons outlined in the paper; the notebook also shows why it was hard to pick a specific value of K.  
Hierarchical clustering is applied with `affinity='cosine'` (the metric used in the space) and `linkage='complete'` (the policy to compute the distance between two clusters, corresponding to the "max rule" detailed in the paper); setting `distance_threshold=0` allows to plot the dendrogram afterwards.  
Finally, it prints, for each chord in the vocabulary, its closest chords in the embedding space (according to the cosine distance).

### [Prediction_continual.ipynb](Prediction_continual.ipynb)

This notebook implements the chord prediction task with architecture A of the predictor network.  
The data loading and Word2Vec training is the same as before, except the sections of a given *test composer* are left out to test the predictor.  
It then trains the predictor network on the same training dataset as Word2Vec, it tests it on the left-out composer, and details the accuracy results (both global and by chord).

### [Prediction_window.ipynb](Prediction_window.ipynb)

This notebook is very similar to the previous one, except it uses architecture B of the predictor network.  
At the bottom, it also performs the unsuccessful task (briefly cited in the paper) of correlation between context entropy and prediction accuracy.

### [dimred_cluster.py](dimred_cluster.py)

This file provides helper functions for the clustering and the (unreported) dimensionality reduction task (with primitives like PCA or tSNE).  
The functions adapt the existing primitives to get, for example, a dictionary mapping chords to cluster indices, or a list of clusters (each being a list of chords).

### [eda.py](eda.py)

This file provides functions used in [EDA.ipynb](EDA.ipynb).  
These are functions that, for example, group chords occurrences by composer, or help to draw a heatmap.

### [entropy_study.py](entropy_study.py)

This file contains functions that compute the context entropy, for various definitions of context, and one to visualise in a scatterplot the correlation with the prediction accuracy.

### [load_data.py](load_data.py)

This file contains functions to load the chord dataset, both as a dataframe and as a list of lists (i.e. list of sections, each being a list of chords).

### [lstm_continual.py](lstm_continual.py)

This file contains the definition of the predictor network with architecture A.
Training and testing are methods of the predictor class.  
The input of the network is a chord, which is internally mapped to embedding coordinates via the Word2Vec model (given in the constructor); the state of the LSTM layer is used to keep memory, and is reset at the beginning of each sentence; the output is a vector of logits, and the loss is the CrossEntropyLoss.

### [lstm_window.py](lstm_window.py)

This file contains the definition of the predictor network with architecture B.
Training and testing are methods of the predictor class.  
The input of the network is directly the juxtaposition of the embedding coordinates of the chords constituting the context of the focus chord; the state of the LSTM layer is not used; the output is a vector of probabilites, and the loss is the MSELoss.

### [visual_clusters.py](visual_clusters.py)

This file contains functions helping to visualise the clusters, for example by plotting the dendrogram, or by plotting the dimensionality-reduced chord embeddings with a different colour for each cluster.

This file contains functions helping to visualise the clusters, for example by plotting the dendrogram, or by plotting the dimensionality-reduced chord embeddings with a different colour for each cluster.

### [run.py](run.py)

This file runs the algorithm a given number of times for all composers and output a dataframe that contains the accuracies for each composer and each run
