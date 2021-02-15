# Data

This folder contains two distinct dataset: a *key* dataset (containing the sequence of keys traversed by each piece of each composer) and a *chord* dataset (containing the sequence of chords traversed by each section of each composer). In this project, we only used the chord dataset.  

### [Top level](.)

The 6 files at the top level are almost unused. They provide the vocabulary of distinct words contained in a dataset: for example, [chord_vocab_minor.txt](data/chord_vocab_minor.txt) contains the list of all unique chords used by composers in minor sections.

### [Chord dataset](./chord)

This dataset consists of 24 csv files, one for each composer. Each file consists of several lines, each representing a musical section. Each section consists of a key mode (e.g. MINOR), followed by a semicolon and by a list of chords in relative notation. Files contain both major and minor sections, so it is up to us to separate them.

### [Key dataset](./key)

This dataset is similar to the chord dataset, but is not used in this project.

