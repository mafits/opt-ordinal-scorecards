"""
CAIM
=====

# CAIM (class-attribute interdependence maximization) algorithm for
        supervised discretization

.. note::
    "L. A. Kurgan and K. J. Cios (2004), CAIM discretization algorithm in
    IEEE Transactions on Knowledge and Data Engineering, vol. 16, no. 2, pp. 145-153, Feb. 2004.
    doi: 10.1109/TKDE.2004.1269594"
    .. _a link: http://ieeexplore.ieee.org/document/1269594/

.. module:: caimcaim
   :platform: Unix, Windows
   :synopsis: A simple, but effective discretization algorithm

"""


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CAIMD(BaseEstimator, TransformerMixin):

    def __init__(self, categorical_features='auto'):
        """
        CAIM discretization class

        Parameters
        ----------
        categorical_features : 'auto' or 'all' or list/array of indices or list of labels
        Specify what features are treated as categorical.
        - 'auto' (default): Only those features whose number of unique values exceeds the number of classes
                            of the target variable by 2 times or more
        - array of indices: array of categorical feature indices
        - list of labels: column labels of a pandas dataframe

        Example
        ---------
        >>> from caimcaim import CAIMD
        >>> from sklearn.datasets import load_iris
        >>> iris = load_iris()
        >>> X = iris.data
        >>> y = iris.target

        >>> caim = CAIMD()
        >>> x_disc = caim.fit_transform(X, y)
        """

        if isinstance(categorical_features, str): # 'auto' or 'all'
            self._features = categorical_features
            self.categorical = None
        elif (isinstance(categorical_features, list)) or (isinstance(categorical_features, np.ndarray)): # array of indices or list of labels
            self._features = None
            self.categorical = categorical_features
        else: # wrong type
            raise CategoricalParamException(
                "Wrong type for 'categorical_features'. Expected 'auto', an array of indicies or labels.")

    def fit(self, X, y):
        """
        Fit CAIM
        Parameters
        ----------
        X : array-like, pandas dataframe, shape [n_samples, n_feature]
            Input array can contain missing values
        y:  array-like, pandas dataframe, shape [n_samples]
            Target variable. Must be categorical.
        Returns
        -------
        self
        """

        self.split_scheme = dict()
        if isinstance(X, pd.DataFrame): # if X is a pandas dataframe
            # self.indx = X.index
            # self.columns = X.columns
            if isinstance(self._features, list): # if the categorical features is a list of labels
                self.categorical = [X.columns.get_loc(label) for label in self._features]  # get the indices of the labels
            X = X.values
            y = y.values
        if self._features == 'auto':
            self.categorical = self.check_categorical(X, y) # check which features are categorical 
        categorical = self.categorical
        print('Categorical', categorical)

        min_splits = np.unique(y).shape[0] 
        # minimum number of splits (number of cut off points; number of intervals - 1) 
        # is the number of classes of the target variable

        for j in range(X.shape[1]): # for each feature
            if j in categorical: # skip if feature categorical
                continue
            xj = X[:, j] # get the feature
            xj = xj[np.invert(np.isnan(xj))] # remove missing values
            
            new_index = xj.argsort() # indices that would sort the array
            xj = xj[new_index] # sort the feature
            yj = y[new_index] # sort the target variable
            
            allsplits = np.unique(xj)[1:-1].tolist()  # potential split points are unique values of the feature
            global_caim = -1 # best CAIM value globally (in all while iterations)
            mainscheme = [xj[0], xj[-1]] # scheme is D; initial scheme is the minimum and maximum values of the feature
            best_caim = 0 # best CAIM value (in current while iteration)
            k = 1
            
            # while 
            # - the number of splits is less than the minimum number of splits
            # - or 
            #   - the global CAIM is less than the best CAIM
            #   - and there are still potential split points
            while (k <= min_splits) or ((global_caim < best_caim) and (allsplits)):
                split_points = np.random.permutation(allsplits).tolist() # random permutation of potential split points
                best_scheme = None # best discretization scheme
                best_point = None # best split point
                best_caim = 0 # best CAIM value
                k = k + 1
                
                while split_points: # while there are still potential split points
                    scheme = mainscheme[:] # copy the main scheme
                    sp = split_points.pop() # get the last potential split point
                    scheme.append(sp) 
                    scheme.sort() 
                    c = self.get_caim(scheme, xj, yj) # get the CAIM of the scheme
                    if c > best_caim: # if better than the best CAIM, update
                        best_caim = c
                        best_scheme = scheme
                        best_point = sp 
                    
                if (k <= min_splits) or (best_caim > global_caim):
                    mainscheme = best_scheme
                    global_caim = best_caim
                    try:
                        allsplits.remove(best_point) # remove the best split point from the potential split points
                    except ValueError:
                        raise NotEnoughPoints('The feature #' + str(j) + ' does not have' +
                                              ' enough unique values for discretization!' +
                                              ' Add it to categorical list!')

            self.split_scheme[j] = mainscheme
            print('#', j, ' GLOBAL CAIM ', global_caim)
        return self

    def transform(self, X):
        """
        Discretize X using a split scheme obtained with CAIM.
        Parameters
        ----------
        X : array-like or pandas dataframe, shape [n_samples, n_features]
            Input array can contain missing values
        Returns
        -------
        X_di : sparse matrix if sparse=True else a 2-d array, dtype=int
            Transformed input.
        """

        if isinstance(X, pd.DataFrame): 
            self.indx = X.index
            self.columns = X.columns
            X = X.values
        
        X_di = X.copy() # discretized X (initially a copy)
        categorical = self.categorical

        scheme = self.split_scheme 
        # scheme is array of cut off points (for each feature)
        
        for j in range(X.shape[1]):
            if j in categorical:
                continue
            sh = scheme[j] # get the cut off points for the feature
            sh[-1] = sh[-1] + 1 # add one to the last cut off point, to ensure that the last interval is inclusive
            xj = X[:, j] # get the feature
            
            # xi = xi[np.invert(np.isnan(xi))]
            
            # for all cut off points
            for i in range(len(sh) - 1):
                # get the indices of the values in the interval [sh i, sh i+1[
                ind = np.where((xj >= sh[i]) & (xj < sh[i + 1]))[0] 
                X_di[ind, j] = i # assign this interval id
        if hasattr(self, 'indx'):
            return pd.DataFrame(X_di, index=self.indx, columns=self.columns)
        return X_di

    def fit_transform(self, X, y):
        """
        Fit CAIM to X,y, then discretize X.
        Equivalent to self.fit(X).transform(X)
        """
        self.fit(X, y)
        return self.transform(X)

    # returns the CAIM value of a scheme (D)
    # CAIM(C, D | F) = 1/n * sum(1 to n) (maxr**2/M+r) 
    def get_caim(self, scheme, xi, y):
        sp = self.index_from_scheme(scheme[1:-1], xi)
        sp.insert(0, 0)
        sp.append(xi.shape[0])
        n = len(sp) - 1
        isum = 0
        for j in range(n):
            init = sp[j]
            fin = sp[j + 1]
            Mr = xi[init:fin].shape[0]
            val, counts = np.unique(y[init:fin], return_counts=True)
            maxr = counts.max()
            isum = isum + (maxr / Mr) * maxr
        return isum / n

    def index_from_scheme(self, scheme, x_sorted):
        split_points = []
        for p in scheme:
            split_points.append(np.where(x_sorted > p)[0][0])
        return split_points

    # check which features are categorical
    # if the number of unique values is less than 2 * number of classes of the target variable
    # then the feature is categorical
    def check_categorical(self, X, y): 
        categorical = [] 
        ny2 = 2 * np.unique(y).shape[0] # 2 * number of classes of the target variable
        for j in range(X.shape[1]): # for each feature
            xj = X[:, j] # get the feature
            xj = xj[np.invert(np.isnan(xj))] 
            if np.unique(xj).shape[0] < ny2:
                categorical.append(j)
        return categorical


class CategoricalParamException(Exception):
    # Raise if wrong type of parameter
    pass


class NotEnoughPoints(Exception):
    # Raise if a feature must be categorical, not continuous
    pass
