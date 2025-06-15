import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# evaluation 
from sklearn.metrics import balanced_accuracy_score, mean_squared_error, roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve

# ordinal to binary 
from sbc import SBC

# discretization thresholds
from libraries.caimcaim import CAIMD # https://github.com/airysen/caimcaim/blob/master/caimcaim/caimcaim.py

# objective function
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from skglm.estimators import MCPRegression
from skglm.datafits import Huber
from skglm.penalties import MCPenalty
from skglm.estimators import GeneralizedLinearEstimator
from skglm.solvers import AndersonCD
from sparselm.model import AdaptiveLasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet




class Scorecard():
    def __init__(self):
        # data
        self.X = None
        self.X_columns = None
        self.y = None
        self.categorical = None
        
        # train and test data
        self.test_X = None
        self.test_y = None
        self.train_X = None
        self.train_y = None
        
        # original data (before SBC, discretization and encoding)
        self.test_y_og = None
        self.train_X_og = None
        self.train_y_og = None
        self.test_X_og = None

        # discretization thresholds and encoded data
        self.thresholds = None
        self.X_disc = []
        self.test_X_disc = None
        
        self.sbc = SBC()
        self.use_sbc = False
        
        # model and weights
        self.model = None
        self.weights = None
        self.nonzero_weights = None
        self.encoding_method = None
        self.model_method = None    
        self.params = None  # parameters for the model, if needed
        
        # metrics
        self.goal_num_nonzero_weights = None
        self.accuracy = None
        
        self.show_prints = True
        
        
        

    
    def fit(self, X, y, categorical_columns, thresholds_method, encoding_method, model_method, use_sbc=False, mapping=None, num_nonzero_weights=None, show_prints=True, params=None):
        self.X = X
        self.y = y
        self.categorical = categorical_columns
        self.encoding_method = encoding_method
        self.model_method = model_method
        self.use_sbc = use_sbc
        self.show_prints = show_prints
        self.goal_num_nonzero_weights = num_nonzero_weights 
        self.params = params
        
        # get train and test data (75% train, 25% test)
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(self.X, self.y, test_size=0.25, random_state=42)
        
        # transform from ordinal to binary
        self.train_X_og = self.train_X.copy()
        self.train_y_og = self.train_y.copy()
        self.test_X_og = self.test_X.copy()
        self.test_y_og = self.test_y.copy()
        
        # if the problem is ordinal, use SBC to transform it to binary
        if use_sbc:
            print("SBC reduction of train set")
            self.train_X, self.train_y = self.sbc.reduction(self.train_X, self.train_y, mapping)
            print("\nSBC reduction of test set")
            self.test_X, self.test_y = self.sbc.reduction(self.test_X, self.test_y, mapping)
        
        
        # discretization thresholds
        print('\ndiscretization thresholds')
        if thresholds_method == "CAIM": self.discretize_caim()
        elif thresholds_method == "INF_BINS": self.discretize_infbins()
        
        # encoding
        print('\nencoding')
        if encoding_method == "1_OUT_OF_K": self.X_disc = self.disc_1_out_of_k(self.train_X)
        elif encoding_method == "DIFF_CODING": self.X_disc = self.disc_diff_coding(self.train_X)
            
        # model (get weights)
        print('\nmodel')
        if model_method == "RSS": self.rss()
        elif model_method == "ML": self.max_likelihood()
        elif model_method == "MM": self.margin_max()
        elif model_method == "BEYOND_L1": self.beyond_l1()
        elif model_method == "ADAPTIVE_LASSO": self.adaptive_lasso()
        
        # show weights
        if show_prints and self.weights is not None: 
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
            else:
                print("all weights are non-zero")
                if model_method == "ADAPTIVE_LASSO":
                    print("number of weights bigger than 1.0e-20: ", np.sum(np.abs(self.weights['Weight']) > 1.0e-20))
        
        return self.model, self.weights
        
            
        
    
    
    # discretization thresholds
    # CAIM
    def discretize_caim(self):
        index_categorical = [self.train_X.columns.get_loc(col) for col in self.categorical]
        caim = CAIMD(list(self.categorical))
        
        # remove sbc_column
        X_aux = self.train_X.copy()
        if self.use_sbc:
            sbc_column = self.train_X.columns[-1]
            # remove sbc_column from X_aux
            X_aux = X_aux.drop(columns=[sbc_column])

        # get thresholds
        caim.fit_transform(X_aux, self.train_y) # fit() and transform()
        
        # get thresholds from caim.split_scheme (dict with column index : thresholds)
        # transform all values to floats
        # and keys with column indexes to column names 
        index_non_categorical = [i for i in range(X_aux.shape[1]) if i not in index_categorical]
        self.thresholds = {X_aux.columns[index_non_categorical[i]]: [float(val) for val in values] for i, (key, values) in enumerate(caim.split_scheme.items())}
        
        # for categorical features
        # sort the unique values and make thresholds be the values in between each pair of consecutive values
        for i, col in enumerate(self.categorical):
            self.thresholds[col] = np.unique(self.train_X[col].astype(str))
            self.thresholds[col] = list(self.thresholds[col])
            
        # do thresholds for sbc_column (= the values of the column)
        if self.use_sbc:
            self.thresholds[sbc_column] = {float(val) for val in self.train_X[sbc_column]}
            self.thresholds[sbc_column] = list(self.thresholds[sbc_column])
        
        # print thresholds
        if self.show_prints: 
            print("\nthresholds ", self.thresholds)
            print("num of bins: ")
            for i, (key, value) in enumerate(self.thresholds.items()):
                if i in index_categorical:
                    print(f"  {key}: {len(value)}")
                else:
                    # +1 because the number of bins is the number of thresholds + 1
                    # e.g. if thresholds are [2, 4, 6], then there are 4 bins: (-inf, 2), [2, 4), [4, 6), [6, inf)
                    print(f"  {key}: {len(value)+1}")
            
    
    # INFINITESIMAL BINS
    # thresholds are the points in between 2 consecutive values in the sorted list
    def discretize_infbins(self):
        self.thresholds = {}
        for col in self.train_X.columns:
            # if the column is categorical, use unique values as thresholds
            if col in self.categorical:
                self.thresholds[col] = np.unique(self.train_X[col].astype(str))
                self.thresholds[col] = list(self.thresholds[col])
            # if the column is numerical, use the points in between 2 consecutive values as thresholds
            else:
                sorted_col = np.unique(self.train_X[col])
                sorted_col = sorted_col.astype(float)  # ensure the values are floats
                col_thresholds = (sorted_col[:-1] + sorted_col[1:]) / 2
                self.thresholds[col] = col_thresholds.tolist()
        
        if self.show_prints: 
            print("\nthresholds ", self.thresholds)
            print("num of bins: ")
            for key, value in self.thresholds.items():
                if key in self.categorical:
                    print(f"  {key}: {len(value)}")
                else:
                    print(f"  {key}: {len(value)+1}")

        return self.thresholds
    
    
    
    
    # encoding
    # 1 out of k
    def disc_1_out_of_k(self, X_to_encode):
        X_disc = []
        # for each column in X_to_encode, create a one-hot encoding of the bins
        for col in X_to_encode.columns:
            if col in self.categorical:
                bin = pd.Categorical(X_to_encode[col], categories=self.thresholds[col]).codes 
                num_bins = len(self.thresholds[col])
            else:
                bin = np.digitize(X_to_encode[col], self.thresholds[col]) # gets bin number of each row
                X_to_encode[col] = X_to_encode[col].astype(float) 
                num_bins = len(self.thresholds[col]) + 1 
            bins_df = pd.get_dummies(bin, prefix=f'feat{col}-bin', prefix_sep='').astype(int) # one hot encoding
            
            # add missing columns (if some bins are not present in the data)
            missing_cols = []
            for i in range(1, num_bins):
                col_name = f'feat{col}-bin{i}'
                if col_name not in bins_df.columns:
                    missing_cols.append(pd.Series(0, index=bins_df.index, name=col_name))
            if missing_cols:
                bins_df = pd.concat([bins_df] + missing_cols, axis=1)
            
            # remove first column (bin0)
            bins_df = bins_df.drop(columns=f'feat{col}-bin0', errors='ignore')
            
            bins_df = bins_df.reindex(sorted(bins_df.columns), axis=1)
            
            # add bins of the column to the list
            X_disc.append(bins_df)
        X_disc = pd.concat(X_disc, axis=1)
        
        # show self.X_disc
        if self.show_prints: 
            print("X_disc shape: ", X_disc.shape)
            print("X_disc columns: ", X_disc.columns)
            print("X_disc head: ", X_disc.head())
        
        return X_disc
    
    
    # differential coding
    def disc_diff_coding(self, X_to_encode):
        X_disc = []
        for col in X_to_encode.columns:
            if col in self.categorical:
                bin = pd.Categorical(X_to_encode[col], categories=self.thresholds[col]).codes
                num_bins = len(self.thresholds[col])
            else:
                X_to_encode[col] = X_to_encode[col].astype(float)
                bin = np.digitize(X_to_encode[col], self.thresholds[col]) # gets bin number of each row
                num_bins = len(self.thresholds[col]) + 1
            
            bin_df = pd.DataFrame(0, index=X_to_encode.index, columns=[f'feat{col}-bin{i}' for i in range(1, num_bins)])
            for i in range(1, num_bins):
                bin_df[f'feat{col}-bin{i}'] = (bin >= i).astype(int)
            
            X_disc.append(bin_df)
        
        X_disc = pd.concat(X_disc, axis=1)
        
        # sort columns    
        if self.show_prints: 
            print("X_disc shape: ", X_disc.shape)
            print("X_disc columns: ", X_disc.columns)
            print("X_disc head: ", X_disc.head())
        
        return X_disc
    
    
    
    # model
    # custom scorer for grid search, that returns the number of non-zero weights
    def custom_scorer(self, estimator, X, y):
        estimator.fit(X, np.ravel(y))
        weights = estimator.coef_[0]
        num_nonzero_weights = np.sum(weights != 0)
        return -abs(num_nonzero_weights - self.goal_num_nonzero_weights)
    
    def grid_search(self, model, param_grid, cv=10):
        # if a goal number of non-zero weights is not specified, use accuracy as the scoring metric        
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv)
        # else, use custom scorer
        if self.goal_num_nonzero_weights is not None: 
            grid_search.scoring= self.custom_scorer
        
        if self.show_prints: grid_search.verbose=2
        grid_search.fit(self.train_X, np.ravel(self.train_y))

        return grid_search
        

    # RSS
    def rss(self): 
        linear_regression = Lasso()
        
        # if params are provided, use them
        if self.params is not None:
            # put params in Lasso
            for key, value in self.params.items():
                setattr(linear_regression, key, value)
            self.model = linear_regression
        
        # else, use grid search to find the best parameters
        else:
            param_grid = {'alpha': [0.001, 0.01, 0.1, 0.4, 0.6, 0.9, 0.99, 1.0]}
            grid_search_rss = self.grid_search(linear_regression, param_grid)
            if self.show_prints: print("RSS best parameters: ", grid_search_rss.best_params_)
            self.model = grid_search_rss.best_estimator_
        
        # get weights
        self.get_weights()
    
    # maximum likelihood (GLM with binomial response and logit link function)
    def max_likelihood(self):
        logistic = LogisticRegression(solver = 'liblinear', penalty = 'l1', max_iter=10000)
        
        # if params are provided, use them
        if self.params is not None:
            for key, value in self.params.items():
                setattr(logistic, key, value)
            self.model = logistic
        
        # else, use grid search to find the best parameters
        else:
            alpha_values = [0.001, 0.01, 0.1, 0.4, 0.6, 0.9, 1.0]
            param_grid = {
                'C': [1/a for a in alpha_values], # inverse of regularization strength
                'class_weight': ['balanced', None]
            }
            
            grid_search_logistic = self.grid_search(logistic, param_grid)
            best_alpha = 1/grid_search_logistic.best_params_['C']

            if self.show_prints: print("ML best parameters: ", grid_search_logistic.best_params_)
            if self.show_prints: print("ML best alpha: ", best_alpha)
            self.model = grid_search_logistic.best_estimator_
        
        # get weights
        self.model.fit(self.X_disc,  np.ravel(self.train_y))
        weights = self.model.coef_[0]
        feature_names = self.X_disc.columns
        self.weights = pd.DataFrame({'Feature': feature_names, 'Weight': weights})
    
    # margin maximization (linear SVM)
    def margin_max(self):
        svm = SVC(kernel='linear')
        
        # if params are provided, use them
        if self.params is not None:
            # put params in SVC
            for key, value in self.params.items():
                setattr(svm, key, value)
            self.model = svm
        
        # else, use grid search to find the best parameters
        else: 
            param_grid = {
                'C': [2**i for i in range(-10, 8)],
                'class_weight': ['balanced', None]#,
                #'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
            }
            grid_search_svm = self.grid_search(svm,  param_grid, cv=5)
            self.model = grid_search_svm.best_estimator_
            if self.show_prints: print("MM best parameters: ", grid_search_svm.best_params_)

        # get weights
        self.model.fit(self.X_disc,  np.ravel(self.train_y))
        weights = self.model.coef_[0]
        feature_names = self.X_disc.columns
        self.weights = pd.DataFrame({'Feature': feature_names, 'Weight': weights})
        
    def beyond_l1(self):
        self.model = GeneralizedLinearEstimator(
            datafit=Huber(delta=1.),
            penalty=MCPenalty(alpha=1e-2, gamma=3),
            solver=AndersonCD()
        )
        self.model.fit(self.X_disc, np.ravel(self.train_y))
        
        # get weights
        weights = self.model.coef_.ravel()
        feature_names = self.X_disc.columns
        self.weights = pd.DataFrame({'Feature': feature_names, 'Weight': weights})
        
    def adaptive_lasso(self):
        alasso = AdaptiveLasso(fit_intercept=False)
        if self.params is not None:
            for key, value in self.params.items():
                setattr(alasso, key, value)
            self.model = alasso
        else:
            param_grid = {'alpha': np.logspace(-8, 2, 10)}
            grid_search_alasso = self.grid_search(alasso, param_grid)
            if self.show_prints: print("Adaptive Lasso best parameters: ", grid_search_alasso.best_params_)
            self.model = grid_search_alasso.best_estimator_
        
        # get weights
        self.model.fit(self.X_disc,  np.ravel(self.train_y))
        weights = self.model.coef_
        feature_names = self.X_disc.columns
        self.weights = pd.DataFrame({'Feature': feature_names, 'Weight': weights})


    # evaluate the model on the test set
    def evaluate(self):
        if self.show_prints: print("\nevaluate")
        
        # get encoded version of test set
        if self.show_prints: print("encoding test set")
        if self.encoding_method == "1_OUT_OF_K": self.test_X_disc = self.disc_1_out_of_k(self.test_X)
        elif self.encoding_method == "DIFF_CODING": self.test_X_disc = self.disc_diff_coding(self.test_X)
        
        # evaluate the model on the test set
        y_pred = self.model.predict(self.test_X_disc)
        #y_pred_proba = self.model.predict_proba(self.test_X_disc)[:, 1]
        
        if self.model_method == "BEYOND_L1" or self.model_method == "ADAPTIVE_LASSO" or self.model_method == "RSS":
            # round weights in y_pred to closer integers
            y_pred = np.round(y_pred).astype(int)
            
        
        # show predictions vs true values
        if self.use_sbc:
            y_pred = self.sbc.classif(y_pred, do_mapping=False)
            if(self.sbc.mapping is not None):
                self.test_y_og = self.sbc.apply_mapping(pd.Series(self.test_y_og))
        if self.show_prints:            
            results_df = pd.DataFrame({'predictions': y_pred, 'true values': self.test_y_og})
            print(results_df.head(10))
        
        
            
        # calculate and show metrics
        mse = mean_squared_error(self.test_y_og, y_pred)
        accuracy = accuracy_score(self.test_y_og, y_pred)
        balanced_accuracy = balanced_accuracy_score(self.test_y_og, y_pred)
        #auc = roc_auc_score(self.test_y_og, y_pred_proba)
        
        print("mse: ", mse)
        print("accuracy: ", accuracy)
        print("balanced accuracy: ", balanced_accuracy)
        #print("auc: ", auc)
        
        y_pred_2 = self.model.predict(self.X_disc)
        if self.model_method == "BEYOND_L1" or self.model_method == "ADAPTIVE_LASSO" or self.model_method == "RSS":
            # round weights in y_pred to closer integers
            y_pred_2 = np.round(y_pred_2).astype(int)
            
        if self.use_sbc:
            y_pred_2 = self.sbc.classif(y_pred_2, do_mapping=False)
            if(self.sbc.mapping is not None):
                self.train_y_og = self.sbc.apply_mapping(pd.Series(self.train_y_og))
        
        if self.show_prints: 
            print("accuracy on train set: ", accuracy_score(self.train_y_og, y_pred_2))
            print("mse on train set: ", mean_squared_error(self.train_y_og, y_pred_2))
            print("balanced accuracy on train set: ", balanced_accuracy_score(self.train_y_og, y_pred_2))
                
        return mse, accuracy#, auc

    
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
            accuracies.append(accuracy_score(y_test, y_pred))
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

    def plot_learning_curve(self, scoring='accuracy', cv=10):

        train_sizes, train_scores, test_scores = learning_curve(
            self.model, self.X_disc, np.ravel(self.y), cv=cv, scoring=scoring,
            train_sizes=np.linspace(0.1, 1.0, 10), shuffle=True, random_state=42
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        plt.figure()
        plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
        plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
        plt.title('Learning Curve')
        plt.xlabel('Training examples')
        plt.ylabel(scoring.capitalize())
        plt.legend(loc='best')
        plt.grid()
        plt.show()

    def plot_accuracy_vs_sparsity(self, caim_accuracy, caim_num_zero_weights, infbins_accuracy, infbins_num_zero_weights, thresholds=[0.1, 0.01, 0.001, 0.0001, 0]):
        accuracies = []
        sparsities = []
        
        # get array of weights from self.weights 
        weights = self.weights['Weight'].values.flatten()
        
        if self.show_prints: print(f"CAIM, accuracy: {caim_accuracy}, sparsity: {caim_num_zero_weights}")
        
        for threshold in thresholds:
            # put weights to 0 if its value (in absolute) is less then the threshold
            selected_weights = np.where(np.abs(weights) >= threshold, weights, 0)
            sparsity = int(np.sum(selected_weights != 0)) # number of non-zero weights
            

            # calculate y_pred with selected weights using logistic regression
            selected_weights_series = pd.Series(selected_weights, index=self.test_X_disc.columns)
            logits = self.test_X_disc.values @ selected_weights_series
            probs = 1 / (1 + np.exp(-logits))
            y_pred = (probs >= 0.5).astype(int)
            
            # calculate accuracy           
            accuracy = accuracy_score(self.test_y, y_pred)
            accuracies.append(accuracy)
            sparsities.append(sparsity)
            if self.show_prints: print(f"threshold: {threshold}, accuracy: {accuracy}, sparsity: {sparsity}")
        
        if self.show_prints: print(f"infbins, accuracy: {infbins_accuracy}, sparsity: {infbins_num_zero_weights}")

        # plot
        plt.figure(figsize=(8, 5))
        plt.scatter(sparsities, accuracies, color='red')
        cmap = cm.get_cmap('viridis', len(thresholds))
        colors = [cmap(i) for i in range(len(thresholds))]
        for i, (x, y) in enumerate(zip(sparsities, accuracies)):
            plt.scatter(x, y, color=colors[i], label=f'{thresholds[i]}')
            plt.text(x, y, f'{thresholds[i]}', fontsize=9, ha='right', va='bottom', color=colors[i])
        # CAIM point
        plt.scatter(caim_num_zero_weights, caim_accuracy, color='blue', marker='*', s=150, label='CAIM')
        plt.text(caim_num_zero_weights, caim_accuracy, 'CAIM', fontsize=10, ha='left', va='bottom', color='blue')
        # inf bins threshold=0 point
        plt.scatter(infbins_num_zero_weights, infbins_accuracy, color='green', marker='o', s=60, label='0')
        plt.text(infbins_num_zero_weights, infbins_accuracy, 'infbins', fontsize=10, ha='left', va='bottom', color='green')

        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], label=f'{thresholds[i]}', markersize=8) for i in range(len(thresholds))]
        handles.append(plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='blue', label='CAIM', markersize=12))
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='infbins', markersize=8))
        plt.legend(handles=handles, title='Thresholds', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xlabel('sparsity (number of non-zero weights)')
        plt.ylabel('accuracy')
        plt.title('accuracy vs sparsity')
        plt.show()