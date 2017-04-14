import re
import numpy as np
import random
from sklearn import metrics

clf1_filename = "hw4_cv_predicted_KNN.dat"
clf2_filename = "hw4_cv_predicted_Decision Tree.dat"
#clf3_filename = "hw4_cv_predicted_SVM.dat"
clf3_filename = "hw4_cv_predicted_LogisticRegression.dat"

test_file1 = "hw4_KNN.dat"
test_file2 = "hw4_Decision Tree.dat"
test_file3 = "hw4_LogisticRegression.dat"

# KNN, Decision Tree, SVM
def get_weights_ensemble_classifier(labels):

    with open(clf1_filename, "r") as fh:
        lines = fh.readlines()
    clf1_predicted_labels = [int(l[0:2]) for l in lines]
    with open(clf2_filename, "r") as fh:
        lines = fh.readlines()
    clf2_predicted_labels = [int(l[0:2]) for l in lines]
    with open(clf3_filename, "r") as fh:
        lines = fh.readlines()
    clf3_predicted_labels = [int(l[0:2]) for l in lines]

    f = open('weights_new.csv','w')
    for i in range(0,1000):
        weights = []
        # Get random weights in range [0-1)
        w1 = np.random.randn()
        w2 = np.random.randn()
        w3 = np.random.randn()
        weights.extend((w1,w2,w3))

        # Get ensemble predicted labels
        ensemble_predicted_labels = [0]*len(labels)
        for i in range(0,len(labels)):
            total = w1*clf1_predicted_labels[i] + w2*clf2_predicted_labels[i] + w3*clf3_predicted_labels[i]
            if total>=0:
                ensemble_predicted_labels[i] = 1
            else:
                ensemble_predicted_labels[i] = -1

        f.write("Weights: ")
        f.write(" ".join(str(x) for x in weights))
        f.write(" ")
        #f.write(metrics.classification_report(labels, ensemble_predicted_labels))
        f.write(str(metrics.f1_score(labels, ensemble_predicted_labels, average='weighted')))
        f.write(" ")
        f.write(str(metrics.f1_score(labels, ensemble_predicted_labels, average='binary')))
        f.write("\n")
    f.close()

    # Compare ensemble predicted labels with actual labels

def ensemble_classifier():

    w1 = 0.35390857
    w2 = 0.496915507
    w3 = -0.591769508

    with open(test_file1, "r") as fh:
        lines = fh.readlines()
    test1_labels = [int(l[0:2]) for l in lines]
    with open(test_file2, "r") as fh:
        lines = fh.readlines()
    test2_labels = [int(l[0:2]) for l in lines]
    with open(test_file3, "r") as fh:
        lines = fh.readlines()
    test3_labels = [int(l[0:2]) for l in lines]

    ensemble_predicted_test_labels = [0]*len(test1_labels)
    for i in range(0,len(test1_labels)):
        total = w1*test1_labels[i] + w2*test2_labels[i] + w3*test3_labels[i]
        if total>=0:
            ensemble_predicted_test_labels[i] = 1
        else:
            ensemble_predicted_test_labels[i] = 0

    result_file = 'hw4_ensemble_classifier_3.dat'
    target = open(result_file, 'w')
    for t in ensemble_predicted_test_labels:
        target.write(str(t))
        target.write("\n")
    target.close()

    print "Done!"

'''
if __name__ == '__main__':
    get_weights_ensemble_classifier(labels)
'''
