# Extra results

This folder contains additional figures and documents supporting the results outlined in the paper.

### Dendrograms
There are two dendrograms, one for major sections (also shown in the paper), and another for minor sections. They both show how clustering can isolate some well-known tonal relationships between chords.

### Accuracies
There are two xlsx files for the accuracies detailed by composer and chord, one for major and one for minor sections.  
The last column of each row shows the average of that row, i.e. the average prediction accuracy for that chord. We want to detect outliers, i.e. (`composer`, `chord`) cells such that the accuracy in the cell is significantly lower or higher than the accuracies for the same `chord` (i.e. in the same row): for this reason, the average is unweighted.  
Each cell is coloured either green (positive outlier), yellow (regular point), or red (negative outlier): an outlier is a cell whose accuracy is off by at least 15% from the corresponding row average.

