# Drug-Activity-Prediction
Binary Classification model that can determine, whether a particular Drug is active (1) or not (0).

## The objective of this assignment are the following:
* Use/implement a feature selection/reduction technique.   
* Experiment with various classification models.
* Think about dealing with imbalanced data.
* F1 Scoring Metric

## Data Description:
The training dataset consists of 800 records and the test dataset consists of 350 records. Training class labels are provided and the test labels are held out. The attributes are binary and are presented in a sparse matrix format within train.dat and test.dat.
* train.dat: Training set (a sparse binary matrix, patterns in lines, features in columns, with class label 1 or 0 in the first column).
* test.dat: Testing set (a sparse binary matrix, patterns in lines, features in columns, no class label provided).

The dataset has an imbalanced distribution i.e., within the training set there are only 78 actives (+1) and 722 inactives (0). No information is provided for the test set regarding the distribution. Since the dataset is imbalanced F1-score is used instead of Accuracy as performance metric.

## High level implementation flow:
Ensemble Dimensionality reducer is used to check which dimensionality reduction technique or which combination of dimensionality reduction techniques gives the best performance. Ensemble classifier on the other end, combines the predicted values from different classifiers to give one classified output. 

<p align="center">
  <img src="https://github.com/nivedithabhandary/Drug-Activity-Prediction/blob/master/Flow_diagram.png">
</p>

## Implementation files included:
* classifier.py: Main program which does preprocessing of datasets and also calls different ensemble reduction and ensemble classifier functions
* EnsembleDimensionalityReduction.py: Helper class to test different DR techniques
* EnsembleClassifier.py: Helper methods to check different classifiers

<p align="center">
  <img src="https://github.com/nivedithabhandary/Drug-Activity-Prediction/blob/master/Results.png">
</p>

From the experiments it was found that, dimensionality reduction techniques have significant impact on the performance. Sparse PCA technique gave better results when compared to PCA and LDA, probably because of the sparsity of data. Classifiers like KNN and Decision Tree performed better compared to Naïve Bayes and SVM. SMOTE and Ensemble of Classifiers did not improve the performance on this sparse dataset.
