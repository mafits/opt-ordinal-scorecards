import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

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
        self.og_X = None
        self.og_y = None
        self.thresholds = None
        self.sbc = SBC()
        self.model = None
        self.weights = None
        self.nonzero_weights = None
        self.show_prints = True
        self.use_sbc = False
        self.goal_num_nonzero_weights = None

    
    def fit(self, X, y, thresholds_method, encoding_method, model_method, use_sbc=False, num_nonzero_weights=None, show_prints=True):
        self.X = X
        self.X_columns = X.columns
        self.y = y
        self.show_prints = show_prints
        self.use_sbc = use_sbc
        self.goal_num_nonzero_weights = num_nonzero_weights 
        
        # transform from ordinal to binary
        self.og_X = self.X
        self.og_y = self.y
        if use_sbc:
            print("SBC reduction")
            self.X, self.y, self.X_columns, _ = self.sbc.reduction(self.X, self.y, h=1)
                    
        # discretization thresholds
        print('\ndiscretization thresholds')
        if thresholds_method == "CAIM": self.discretize_caim()
        elif thresholds_method == "INF_BINS": self.discretize_infbins()
        
        # encoding
        print('\nencoding')
        if encoding_method == "1_OUT_OF_K": self.disc_1_out_of_k()
        elif encoding_method == "DIFF_CODING": self.disc_diff_coding()
            
        # model (get weights)
        print('\nmodel')
        if model_method == "RSS": self.rss()
        elif model_method == "ML": self.max_likelihood()
        elif model_method == "MARGIN_MAX": self.margin_max()
        
        # show weights
        if show_prints: 
            print(f"{model_method} weights:\n", self.weights)
            plt.figure()
            plt.bar(self.weights['Feature'], self.weights['Weight'])
            plt.xticks(rotation=90)
            plt.title('ML weights')
            plt.show()
        
        # get nonzero weights
        self.nonzero_weights = self.weights[self.weights['Weight'] != 0]
        if self.nonzero_weights.shape[0] < self.weights.shape[0]:
            print("num of zero weights: ", self.weights.shape[0] - self.nonzero_weights.shape[0])
            print("num of non-zero weights: ", self.nonzero_weights.shape[0])
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
        if self.show_prints: print("num of features: ", self.X.shape[1])
        caim = CAIMD()
        
        # remove sbc_column and take care of it later
        X_aux = self.X.copy()
        if self.use_sbc:
            sbc_column = self.X.columns[-1]
            if self.show_prints: print("sbc_column: ", sbc_column)
            # remove sbc_column from X_aux
            X_aux = X_aux.drop(columns=[sbc_column])

        # get thresholds
        self.thresholds = caim.fit_transform(X_aux, self.y) # fit() and transform()
        
        # get thresholds from caim.split_scheme (dict with column index : thresholds)
        # transform all values to floats
        # and keys with column indexes to column names
        self.thresholds = {X_aux.columns[i]: [float(val) for val in value] for i, (key, value) in enumerate(caim.split_scheme.items())}
        
        # do thresholds for sbc_column = the values of the column
        if self.use_sbc:
            self.thresholds[sbc_column] = {float(val) for val in self.X[sbc_column]}
            self.thresholds[sbc_column] = list(self.thresholds[sbc_column])
        
        # print thresholds
        if self.show_prints: print("\nthresholds ", self.thresholds)
        print("num of bins: ")
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
            col_thresholds = (sorted_col[:-1] + sorted_col[1:]) / 2
            self.thresholds[col] = col_thresholds.tolist()
        
        if self.show_prints: print("\nthresholds ", self.thresholds)
        print("num of bins: ")
        for key, value in self.thresholds.items():
            if self.show_prints: print(f"  {key}: {len(value)+1}")

        return self.thresholds
    
    
    
    
    # encoding
    @staticmethod
    def get_bins(thresholds, values):
        bins = np.digitize(values, thresholds)
        return bins
        # list of bin number for each row
        
    # 1 out of k
    def disc_1_out_of_k(self):
        self.X_disc = []
        for col in self.X_columns:
            bins = self.get_bins(self.thresholds[col], self.X[col]) # gets bin number of each row
            bins_df = pd.get_dummies(bins, prefix=f'feat{col}-bin', prefix_sep='').astype(int) # one hot encoding
            
            # add missing columns
            for i in range(1, len(self.thresholds[col]) + 1):
                if f'feat{col}-bin{i}' not in bins_df.columns:
                    bins_df[f'feat{col}-bin{i}'] = 0
            
            # remove first column (bin0)
            bins_df = bins_df.drop(columns=f'feat{col}-bin0', errors='ignore')
            
            # add bins of the column to the list
            self.X_disc.append(bins_df)
        
        self.X_disc = pd.concat(self.X_disc, axis=1)
    
    # differential coding
    def disc_diff_coding(self):
        self.X_disc = []
        for col in self.X_columns:
            bins = self.get_bins(self.thresholds[col], self.X[col]) # gets bin number of each row
            num_bins = len(self.thresholds[col]) + 1
            bin_df = pd.DataFrame(0, index=self.X.index, columns=[f'feat{col}-bin{i}' for i in range(1, num_bins)])
            for i in range(1, num_bins):
                bin_df[f'feat{col}-bin{i}'] = (bins >= i).astype(int)
            
            self.X_disc.append(bin_df)
        self.X_disc = pd.concat(self.X_disc, axis=1)
    
    
    
    # model
    def custom_scorer(self, estimator, X, y):
        estimator.fit(X, np.ravel(y))
        weights = estimator.coef_[0]
        num_nonzero_weights = np.sum(weights != 0)
        return -abs(num_nonzero_weights - self.goal_num_nonzero_weights)
    
    def grid_search(self, model, param_grid, cv=10):
        if self.goal_num_nonzero_weights is None:
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv)
        else:
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring=self.custom_scorer)
        
        grid_search.fit(self.X_disc, np.ravel(self.y))
        return grid_search
    
    def get_weights(self):
        self.model.fit(self.X_disc,  np.ravel(self.y))
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
        

    def cross_val_score(self, n_splits=10):                   
        kf = StratifiedKFold(n_splits=n_splits)
        MSEs = [] # mean squared error
        accuracies = [] 
        AUCs = [] # area under the ROC curve
 
        for train_index, test_index in kf.split(self.X, self.y):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            self.model.fit(X_train, np.ravel(y_train))
            y_pred = self.model.predict(X_test)
            
            MSEs.append(mean_squared_error(y_test, y_pred))
            accuracies.append((np.array(y_pred) == np.array(y_test)).mean())
            #AUCs.append(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
            if hasattr(self.model, "predict_proba"):
                AUCs.append(roc_auc_score(y_test, self.model.predict_proba(X_test)[:, 1]))
            else:
                AUCs.append(roc_auc_score(y_test, y_pred))
            
        if self.show_prints: print("MSEs: ", MSEs)
        if self.show_prints: print("accuracies: ", accuracies)
        if self.show_prints: print("AUCs: ", AUCs)
    
        if self.show_prints: print("mean MSE: ", np.mean(MSEs))
        print("mean accuracy: ", np.mean(accuracies))
        if self.show_prints: print("mean AUC: ", np.mean(AUCs))
        return np.mean(MSEs), np.mean(accuracies), np.mean(AUCs)
 