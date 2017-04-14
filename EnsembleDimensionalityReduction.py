from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.lda import LDA
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
import numpy as np

class EnsembleDimensionalityReduction(BaseEstimator, TransformerMixin):
    """
    Class to create ensemble of Dimensionality Reduction techniques
    """

    def __init__(self, types, n_components):
        """
        Initializing the class with different techniques and their parameters

        Parameters
        ----------
        types : List of Strings.
                Different type of DR techniques to be included in the ensemble
        n_components : List of int.
                       Number of components to keep for each type

        """
        self.pca = None
        self.sparsePCA = None
        self.lda = None
        self.__isFit = False
        self.ensemble_reduced_X = None
        self.__debug = False

        for index, item in enumerate(types):
            if "pca" in item:
                self.pca = PCA(n_components[index])
            if "sparsePCA" in item:
                self.sparsePCA = SparsePCA(n_components[index])
            if "lda" in item:
                self.lda = LDA()

    def enableDebugMessages(self):
        """
        Enable debug messages
        """
        self.__debug = True

    def disableDebugMessage(self):
        """
        Disable debug messages
        """
        self.__debug = False

    def __printDebugMessages(self, message):
        """
        Helper for debug messageses
        """
        if self.__debug is True:
            print 'DEBUG: '+str(message)


    def fit(self, X, y=None):
        """
        Method to Fit the model with X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.
        y : list of training labels

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        if self.pca is not None:
            self.__printDebugMessages("fit() PCA")
            self.pca.fit(X)
        if self.sparsePCA is not None:
            self.__printDebugMessages("fit() sparsePCA")
            self.sparsePCA.fit(X)
        if self.lda is not None:
            self.__printDebugMessages("fit() lda")
            self.lda.fit(X,y)

        self.__isFit = True

    def __update_ensemble_reduced_X(self, reduced_X):
        if self.ensemble_reduced_X is None:
            self.ensemble_reduced_X = reduced_X
        else:
            self.ensemble_reduced_X = np.hstack((self.ensemble_reduced_X, reduced_X))
        self.__printDebugMessages("__update_ensemble_reduced_X() {} {}".format(np.shape(self.ensemble_reduced_X), np.shape(reduced_X)))

        return self.ensemble_reduced_X


    def transform(self, X):
        """Apply ensemble of dimensionality reduction on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        ensemble_reduced_X : array-like, shape (n_samples, n_components)
                             Training data, where n_samples is the number of
                             samples and n_components is the ensemble of number
                             of features from different DR techniques
        """

        if self.__isFit is False:
            raise Exception("Call fit before transform!")

        self.ensemble_reduced_X = None

        if self.pca is not None:
            self.__printDebugMessages("transform() PCA")
            self.__update_ensemble_reduced_X(np.array(self.pca.transform(X)))
        if self.sparsePCA is not None:
            self.__printDebugMessages("transform() SparsePCA")
            self.__update_ensemble_reduced_X(np.array(self.sparsePCA.transform(X)))
        if self.lda is not None:
            self.__printDebugMessages("transform() lda")
            self.__update_ensemble_reduced_X(np.array(self.lda.transform(X)))

        return self.ensemble_reduced_X


    def fit_transform(self, X, y=None):
        """Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : list of training labels

        Returns
        -------
        ensemble_reduced_X : array-like, shape (n_samples, n_components)
                             Training data, where n_samples is the number of
                             samples and n_components is the ensemble of number
                             of features from different DR techniques
        """
        self.fit(X,y)
        return self.transform(X)

'''
if __name__ == '__main__':
    X = [[1, 2, 3], [3, 4, 5], [-1, 4, 1], [-1, -5, 3], [-3, 4, 0] ,[3, -5, -4]]
    y = [0, 1, 2, 1, 0, 2]
    X_= [[-1, -2, 3], [3, 4, -5]]

    edr = EnsembleDimensionalityReduction(["pca"], [2])
    edr.enableDebugMessages()
    edr.fit(X,y)
    print edr.transform(X)
    print edr.transform(X_)
    edr.disableDebugMessage()
    print "\n\n"

    edr = EnsembleDimensionalityReduction(["pca", "lda"], [2, 2])
    edr.enableDebugMessages()
    print edr.fit_transform(X,y)
'''
