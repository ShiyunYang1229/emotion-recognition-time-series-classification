#=====================#
# Author: Shiyun Yang #
#=====================#

"""
Custom Voting Classifier modified from Scikit-learn VotingClassifier
- Uses Pre-trained Models
- predict function:  returns predicted class and average probability for that class for display purpose
"""

import numpy as np


class VotingClassifier(object):
    def __init__(self, estimators, voting='hard', weights=None):
        self.estimators = [e[1] for e in estimators]
        self.named_estimators = dict(estimators)
        self.voting = voting
        self.weights = weights

    def fit(self, X, y, sample_weight=None):
        raise NotImplementedError
        
    def predict(self, X):
        """ Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        maj : array-like, shape = [n_samples]
            Predicted class labels.
        prob: array-like, shape = [n_samples]
            Average of predicted probabilities of the winning class. 
        """
        prob = []
        if self.voting == 'soft':   # soft voting
            # get winning class
            maj = np.argmax(self.predict_proba(X), axis=1)
            # get average for all probabilies predicted for class maj
            all_probs = self._collect_probas(X)
            all_probs = np.array([xi[:, 1] for xi in all_probs])
            for i in range(len(maj)): 
                prob_i = all_probs[i]
                if maj[i] == 0: 
                    prob.append(np.average(prob_i[prob_i<0.5]))
                else: 
                    prob.append(np.average(prob_i[prob_i>0.5]))
        else:  # 'hard' voting
            predictions = self._predict(X)
            maj = np.apply_along_axis(lambda x:
                                      np.argmax(np.bincount(x,
                                                weights=self.weights)),
                                      axis=1,
                                      arr=predictions.astype('int'))
            all_probs = self._collect_probas(X)
            all_probs = np.array([xi[:, 1] for xi in all_probs])
            for i in range(len(maj)): 
                prob_i = all_probs[i]
                if maj[i] == 0: 
                    prob.append(np.average(prob_i[prob_i<0.5]))
                else: 
                    prob.append(np.average(prob_i[prob_i>0.5]))
        return maj, prob

    def _collect_probas(self, X):
        """Collect results from clf.predict calls. """
        probas = np.asarray([clf.predict_proba(X) for clf in self.estimators])
        probas = np.transpose(probas, (1, 0, 2))
        return probas

    def _predict_proba(self, X):
        """Predict class probabilities for X in 'soft' voting """
        if self.voting == 'hard':
            raise AttributeError("predict_proba is not available when"
                                 " voting=%r" % self.voting)
        avg = np.average(self._collect_probas(X), axis=1, weights=self.weights)
        return avg

    @property
    def predict_proba(self):
        """Compute probabilities of possible outcomes for samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        avg : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.
        """
        return self._predict_proba

    def transform(self, X):
        """Return class labels or probabilities for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        If `voting='soft'`:
          array-like = [n_classifiers, n_samples, n_classes]
            Class probabilities calculated by each classifier.
        If `voting='hard'`:
          array-like = [n_samples, n_classifiers]
            Class labels predicted by each classifier.
        """

        if self.voting == 'soft':
            return self._collect_probas(X)
        else:
            return self._predict(X)

    def _predict(self, X):
        """Collect results from clf.predict calls. """
        return np.asarray([clf.predict(X) for clf in self.estimators]).T