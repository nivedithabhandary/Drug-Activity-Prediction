import re
import numpy as np
import pickle
from sklearn.lda import LDA
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn import cross_validation, linear_model
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

from EnsembleDimensionalityReduction import *

train_file = "train.dat"
test_file = "test.dat"
edr_output = "edr_output.dat"
edr_test_output = "edr_test_output.dat"
print_debug = True
feature_vector_size = 100001
num_pca_components = 1000
k_folds = 10

edr_types = ["sparsePCA"]
edr_n_components = [100]
dump_test_results = True

def preprocess(filename, mode):
    if print_debug is True:
        print 'Processing file'

    with open(filename, "r") as fh:
        lines = fh.readlines()

    if mode == "train":
        labels = [int(l[0]) for l in lines]
        for index, item in enumerate(labels):
            if (item == 0):
                labels[index] = -1
        docs = [re.sub(r'[^\w]', ' ',l[1:]).split() for l in lines]

    else:
        labels = []
        docs = [re.sub(r'[^\w]', ' ',l).split() for l in lines]

    features = []

    for doc in docs:
        line = [0]*feature_vector_size
        for index, val in enumerate(doc):
            line[int(val)] = 1
        features.append(line)

    return features, labels

if __name__ == '__main__':

    print 'Experiments for HW4'

    # Extract faetures
    features, labels = preprocess(train_file, "train")

    # Reduce dimensions
    edr = EnsembleDimensionalityReduction(edr_types, edr_n_components)
    edr.enableDebugMessages()
    reduced_features = edr.fit_transform(features, labels)

    # Storing reduced features to pickle
    with open(edr_output, "wb") as output:
        pickle.dump(reduced_features, output, pickle.HIGHEST_PROTOCOL)

    # Resample using SMOTE
    #sm = SMOTE(kind='regular')
    #features_resampled, labels_resampled = sm.fit_sample(reduced_features, labels)

    # Different classifiers - KNN, Decision Tree, SVM, Naive Bayes, AdaBoost M1, Random Forest

    names = ["KNN", "Decision Tree", "SVM", "Naive Bayes", "LogisticRegression", "AdaBoost", "RandomForest"]
    classifiers = [KNeighborsClassifier(n_neighbors=10),
    DecisionTreeClassifier(random_state=0),
    SVC(kernel="linear", C=0.025),
    GaussianNB(), LogisticRegression(random_state=1),
    AdaBoostClassifier(n_estimators=100),
    RandomForestClassifier(n_estimators=10)
    ]

    test_features, test_labels = preprocess(test_file, "test")
    test_reduced_features = edr.transform(test_features)
    with open(edr_test_output, "wb") as output:
        pickle.dump(test_reduced_features, output, pickle.HIGHEST_PROTOCOL)

    if print_debug is True:
        print 'Cross validation'

    with open(edr_output, "rb") as output:
        reduced_features = pickle.load(output)

    for name, clf in zip(names, classifiers):
        print 'Metric for ' + name
        cv_predicted = cross_val_predict(clf, reduced_features, labels, cv=k_folds)
        cv_predicted_file = 'hw4_cv_predicted_'+name+'.dat'
        print 'Dumping Cross validation predicted results to ', cv_predicted_file
        target = open(cv_predicted_file, 'w')
        for t in cv_predicted:
            target.write(str(t))
            target.write("\n")
        target.close()
        print metrics.classification_report(labels, cv_predicted)

        scores = cross_validation.cross_val_score(clf, reduced_features, labels)
        print '\nCross validation scores: '
        print scores.mean()

        # Actual training
        clf.fit(reduced_features, labels)

        # Predict test labels
        with open(edr_test_output, "rb") as output:
            test_reduced_features = pickle.load(output)
        test_predicted = clf.predict(test_reduced_features)
        print 'Test predicted for ' + name
        print test_predicted
        print sum(test_predicted)

        if dump_test_results is True:
            result_file = 'hw4_'+name+'_'.join(edr_types)+'.dat'
        print 'Dumping test results to ', result_file
        target = open(result_file, 'w')
        for t in test_predicted:
            if int(t) == -1:
                t = 0
            target.write(str(t))
            target.write("\n")
        target.close()

print 'Done!'
print 'Multiple results file generated. One for each classifier.'
