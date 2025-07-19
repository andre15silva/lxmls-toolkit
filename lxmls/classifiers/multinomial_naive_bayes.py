import numpy as np
import scipy as scipy

import lxmls.classifiers.linear_classifier as lc


class MultinomialNaiveBayes(lc.LinearClassifier):
    def __init__(self, xtype="gaussian"):
        lc.LinearClassifier.__init__(self)
        self.trained = False
        self.likelihood = 0
        self.prior = 0
        self.smooth = False
        self.smooth_param = 1

    def train(self, x, y):
        # n_docs = no. of documents
        # n_words = no. of unique words
        n_docs, n_words = x.shape

        # classes = a list of possible classes
        classes = np.unique(y)
        # n_classes = no. of classes
        n_classes = np.unique(y).shape[0]

        # initialization of the prior and likelihood variables
        prior = np.zeros(n_classes)
        likelihood = np.zeros((n_words, n_classes))

        # TODO: This is where you have to write your code!
        # You need to compute the values of the prior and likelihood parameters
        # and place them in the variables called "prior" and "likelihood".
        # Examples:
        # prior[0] is the prior probability of a document being of class 0
        # likelihood[4, 0] is the likelihood of the fifth(*) feature being
        # active, given that the document is of class 0
        # (*) recall that Python starts indices at 0, so an index of 4
        # corresponds to the fifth feature!

        # ----------
        # Solution to Exercise 1

        # prior = P(y)
        # prior is estimated as the number of documents in the class divided by the total number of documents
        # y has shape (n_docs, 1)
        prior = np.bincount(y.flatten(), minlength=n_classes) / len(y)
        # prior has shape (n_classes,)
        
        # likelihood = P(x|y)
        # likelihood[i, j] is estimated as the absolute frequency of the word i in the class j divided by the total number of words in class j 
        for j in range(n_classes):
            # we focus on the documents of class j
            mask = (y.flatten() == j)
            x_j = x[mask, :]  # shape: (num_docs_in_class_j, n_words)
            # we sum the number of times each word appears in the documents of class j
            word_counts = np.sum(x_j, axis=0)  # shape: (n_words,)
            # adding smoothing
            likelihood[:, j] = (word_counts + 1) / (np.sum(word_counts) + n_words)

        # End solution to Exercise 1
        # ----------

        params = np.zeros((n_words + 1, n_classes))
        for i in range(n_classes):
            params[0, i] = np.log(prior[i])
            params[1:, i] = np.nan_to_num(np.log(likelihood[:, i]))
        self.likelihood = likelihood
        self.prior = prior
        self.trained = True
        return params
