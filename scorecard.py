import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

# ordinal to binary 
from sbc import SBC

# discretization thresholds
from libraries.caimcaim import CAIMD # https://github.com/airysen/caimcaim/blob/master/caimcaim/caimcaim.py

# objective function
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# regularization
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet



class Scorecard():
    def __init__(self):
        self.X = None
        self.X_columns = None
        self.X_disc = []
        self.y = None
        self.thresholds = None
        self.sbc = SBC()
        self.model = None
        self.weights = None
        self.nonzero_weights = None
        self.show_prints = True
        self.use_sbc = False

    
    def fit(self, X, y, thresholds_method, encoding_method, model_method, use_sbc=False, show_prints=True):
        self.X = X
        self.X_columns = X.columns
        self.y = y
        self.show_prints = show_prints
        self.use_sbc = use_sbc
        
        # transform from ordinal to binary
        og_X = self.X
        og_y = self.y
        if use_sbc:
            if show_prints: print("SBC reduction")
            self.X, self.y, self.X_columns, _ = self.sbc.reduction(self.X, self.y, h=1)
                    
        # discretization thresholds
        if show_prints: print('\ndiscretization thresholds')
        if thresholds_method == "CAIM": self.discretize_caim()
        elif thresholds_method == "INF_BINS": self.discretize_infbins()
        
        # encoding
        if show_prints: print('\nencoding')
        if encoding_method == "1_OUT_OF_K": self.disc_1_out_of_k()
        elif encoding_method == "DIFF_CODING": self.disc_diff_coding()
            
        # model (get weights)
        if show_prints: print('\nmodel')
        if model_method == "RSS": self.rss()
        elif model_method == "ML": self.max_likelihood()
        elif model_method == "MARGIN_MAX": self.margin_max()
        
        # show weights
        if show_prints: print(f"{model_method} weights:\n", self.weights)
        plt.figure()
        plt.bar(self.weights['Feature'], self.weights['Weight'])
        plt.xticks(rotation=90)
        plt.title('ML weights')
        plt.show()
        
        # get nonzero weights
        self.nonzero_weights = self.weights[self.weights['Weight'] != 0]
        if self.nonzero_weights.shape[0] < self.weights.shape[0]:
            print("num of non-zero weights: ", self.nonzero_weights.shape[0])
            print("num of zero weights: ", self.weights.shape[0] - self.nonzero_weights.shape[0])
            print(self.nonzero_weights)
            plt.figure()
            plt.bar(self.nonzero_weights['Feature'], self.nonzero_weights['Weight'])
            plt.xticks(rotation=90)
            plt.title('ML non-zero weights')
            plt.show()
        
        return self.model, self.weights
        
            
        
    
    
    # discretization thresholds
    # CAIM
    def discretize_caim(self):
        print("num of features: ", self.X.shape[1])
        caim = CAIMD()
        X = self.X.copy()
        if(self.use_sbc):
            sbc_column = self.X.columns[-1]
            X = self.X.iloc[:, -1]
        self.thresholds = caim.fit_transform(X, self.y) # fit() and transform()
        
        # get thresholds from caim.split_scheme (dict with column index : thresholds)
        # transform all values to floats
        # and keys with column indexes to column names
        self.thresholds = {self.X_columns[i]: [float(val) for val in value] for i, (key, value) in enumerate(caim.split_scheme.items())}
        
        if self.show_prints: print("\nthresholds ", self.thresholds)
        if self.show_prints: print("num of bins: ")
        for i, (key, value) in enumerate(self.thresholds.items()):
            if self.show_prints: print(f"  {key}: {len(value)+1}")
    
    # INFINITESIMAL BINS
    # thresholds are the points in between 2 consecutive values in the sorted list
    def discretize_infbins(self):
        self.thresholds = {}
        for col in self.X_columns:
            # sort unique values
            sorted_col = np.unique(self.X[col])
            # get thresholds
            self.thresholds = (sorted_col[:-1] + sorted_col[1:]) / 2
            self.thresholds[col] = self.thresholds.tolist()
        
        if self.show_prints: print("\nthresholds ", self.thresholds)
        if self.show_prints: print("num of bins: ")
        for key, value in self.thresholds.items():
            if self.show_prints: print(f"  {key}: {len(value)+1}")

        return self.thresholds
    
    
    
    
    # encoding
    @staticmethod
    def get_bins(thresholds, values):
        bins = np.digitize(values, thresholds)
        return bins
        # list of bin number for each row

    def disc_1_out_of_k(self):
        for col in self.X_columns:
            t = self.thresholds[col]
            print("t ", t)
            x = self.X[col]
            print(x)
            bins = self.get_bins(t,x) # gets bin number of each row
            bins_df = pd.get_dummies(bins, prefix=f'feat{col}-bin', prefix_sep='').astype(int) # one hot encoding
            for i in range(1, len(self.thresholds[col]) + 1):
                if f'feat{col}-bin{i}' not in bins_df.columns:
                    bins_df[f'feat{col}-bin{i}'] = 0
            bins_df = bins_df.drop(columns=f'feat{col}-bin0', errors='ignore')
            self.X_disc.append(bins_df)
        self.X_disc = pd.concat(self.X_disc, axis=1)
    
    def disc_diff_coding(self):
        for col in self.X_columns:
            bins = self.get_bins(self.thresholds[col], self.X[col]) # gets bin number of each row
            num_bins = len(self.thresholds[col]) + 1
            bin_df = pd.DataFrame(0, index=self.X.index, columns=[f'feat{col}-bin{i}' for i in range(1, num_bins)])
            for i in range(1, num_bins):
                bin_df[f'feat{col}-bin{i}'] = (bins >= i).astype(int)
            self.X_disc.append(bin_df)
        self.X_disc = pd.concat(self.X_disc, axis=1)
    
    
    
    
    # model
    def grid_search(self, model, param_grid, cv=10):
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv)
        grid_search.fit(self.X_disc, self.y)
        return grid_search
    
    def get_weights(self):
        self.model.fit(self.X_disc, self.y)
        weights = self.model.coef_[0]
        feature_names = self.X_disc.columns
        self.weights = pd.DataFrame({'Feature': feature_names, 'Weight': weights})
        #return weights_df

    # RSS --> não está a funcionar pq Lasso tá à espera de valores contínuos
    def rss(self): 
        linear_regression = Lasso()
        param_grid = {'alpha': [0.001, 0.01, 0.1, 0.4, 0.6, 0.9, 0.99, 1.0]}
        grid_search_rss = self.grid_search(linear_regression, param_grid)
        if self.show_prints: print("RSS best parameters: ", grid_search_rss.best_params_)
        self.model = grid_search_rss.best_estimator_
        self.get_weights()
    
    # maximum likelihood (GLM with binomial response and logit link function)
    def max_likelihood(self):
        logistic = LogisticRegression(solver = 'liblinear', penalty = 'l1')
        alpha_values = [0.001, 0.01, 0.1, 0.4, 0.6, 0.9, 1.0]
        param_grid = {'C': [1/a for a in alpha_values]} # inverse of regularization strength
        grid_search_logistic = self.grid_search(logistic, param_grid)
        if self.show_prints: print("ML best parameters: ", grid_search_logistic.best_params_)
        best_alpha = 1/grid_search_logistic.best_params_['C']
        if self.show_prints: print("ML best alpha: ", best_alpha)
        self.model = grid_search_logistic.best_estimator_
        self.get_weights()

    
    # margin maximization (linear SVM)
    def margin_max(self):
        param_grid = {
            'C': [2**i for i in range(-10, 11)],
            'class_weight': ['balanced', None],
        }
        svm = SVC(kernel='linear')
        #svm = svm_problem(app_y, disc_app_X)
        grid_search_svm = self.grid_search(svm,  param_grid)
        self.model = grid_search_svm.best_estimator_
        self.weights = self.get_weights()
