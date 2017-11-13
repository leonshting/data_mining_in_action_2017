from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
import numpy as np


TREE_PARAMS_DICT = {'max_depth':100, 'min_samples_leaf':5}
TAU = 0.07


class SimpleGB(BaseEstimator):
    def __init__(self, tree_params_dict, iters=100, tau=1e-1):
        self.tree_params_dict = tree_params_dict
        self.iters = iters
        self.tau = tau
        
    def fit(self, X_data, y_data):
        self.estimators = []
        curr_pred = 0
        for iter_num in xrange(self.iters):
            algo = DecisionTreeRegressor(
                max_depth=self.tree_params_dict.get('max_depth', 10), 
                min_samples_leaf = self.tree_params_dict.get('min_samples_leaf', 2)
            )
            algo.fit(X_data, 2*(y_data - self.predict(X_data)))
            self.estimators.append(algo)
    
    def predict(self, X_data):
        res = np.zeros(X_data.shape[0])
        for estimator in self.estimators:
            res += self.tau * estimator.predict(X_data)
        return res
