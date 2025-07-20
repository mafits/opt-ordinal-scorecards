import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import re

# evaluation 
from sklearn.metrics import balanced_accuracy_score, mean_squared_error, roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix
import seaborn as sns

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
from sklearn.base import clone
from sklearn.model_selection import KFold, ParameterGrid






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
        self.train_and_val_X = None
        self.train_and_val_y = None
        
        # original train and test data 
        self.test_y_og = None
        self.train_X_og = None
        self.train_y_og = None
        self.test_X_og = None

        # discretization thresholds and encoded data
        self.thresholds_method = None
        self.encoding_method = None
        self.thresholds = None
        self.encoded_train_X = None
        self.test_X_disc = None
        
        self.sbc = SBC()
        self.use_sbc = False
        self.K = None  # number of ordinal categories
        self.mapping = None  # mapping for ordinal categories, if needed
        
        # model and weights
        self.model = None
        self.weights = None
        self.nonzero_weights = None
        self.model_method = None    
        self.params = None  # parameters for the model
        self.file_name = None  # risk slim data file name

        # metrics
        self.goal_num_nonzero_weights = None
        self.accuracy = None
        self.show_prints = True
        
       


    
    def fit(self, X, y, categorical_columns, thresholds_method, encoding_method, model_method, params=None, use_sbc=False, K=None, mapping=None, file_name=None, show_prints=True):
        self.X = X
        self.y = y
        self.categorical = categorical_columns
        self.thresholds_method = thresholds_method
        self.encoding_method = encoding_method
        self.model_method = model_method
        self.params = params
        self.use_sbc = use_sbc
        self.K = K
        self.mapping = mapping
        self.show_prints = show_prints
        self.file_name = file_name
        
        # get train and test data (75% train, 25% test)
        self.train_and_val_X, self.test_X, self.train_and_val_y, self.test_y = train_test_split(self.X, self.y, test_size=0.25, random_state=42)

        # transform from ordinal to binary
        self.train_X_og = self.train_and_val_X.copy()
        self.train_y_og = self.train_and_val_y.copy()
        self.test_X_og = self.test_X.copy()
        self.test_y_og = self.test_y.copy()
        
        # model
        if model_method == "RSS": self.rss()
        elif model_method == "ML": self.max_likelihood()
        elif model_method == "MM": self.margin_max()
        elif model_method == "BEYOND_L1": self.beyond_l1()
        elif model_method == "ADAPTIVE_LASSO": self.adaptive_lasso()
        elif model_method == "RiskSLIM": self.risk_slim()
        
        # get weights
        self.get_weights()
        
        return self.model, self.weights
        
            
        
    # discretization thresholds
    def get_thresholds(self, X, y):
        if self.thresholds_method == "CAIM":
            return self.discretize_caim(X, y)
        elif self.thresholds_method == "INFINITESIMAL_BINS":
            return self.discretize_infbins(X)
        else:
            raise ValueError(f"Unknown thresholds method: {self.thresholds_method}")

    # CAIM
    def discretize_caim(self, X, y):
        thresholds = {}
        index_categorical = [X.columns.get_loc(col) for col in self.categorical]
        caim = CAIMD(list(self.categorical))
        
        # get thresholds
        caim.fit_transform(X, y) # fit() and transform()
        
        # get thresholds from caim.split_scheme (dict with column index : thresholds)
        # transform all values to floats
        # and keys with column indexes to column names 
        index_non_categorical = [i for i in range(X.shape[1]) if i not in index_categorical]
        thresholds = {X.columns[index_non_categorical[i]]: [float(val) for val in values] for i, (key, values) in enumerate(caim.split_scheme.items())}
        
        # for categorical features
        # sort the unique values and make thresholds be the values in between each pair of consecutive values
        for i, col in enumerate(self.categorical):
            thresholds[col] = np.unique(X[col].astype(str))
            thresholds[col] = list(thresholds[col])

        if self.categorical:
            thresholds = {col: thresholds[col] for col in X.columns if col in thresholds}

        return thresholds
                   
    # INFINITESIMAL BINS
    # thresholds are the points in between 2 consecutive values in the sorted list
    def discretize_infbins(self, X):
        thresholds = {}
        for col in X.columns:
            # if the column is categorical, use unique values as thresholds
            if col in self.categorical:
                thresholds[col] = np.unique(X[col].astype(str))
                thresholds[col] = list(thresholds[col])
            # if the column is numerical, use the points in between 2 consecutive values as thresholds
            else:
                sorted_col = np.unique(X[col])
                sorted_col = sorted_col.astype(float)  # ensure the values are floats
                col_thresholds = (sorted_col[:-1] + sorted_col[1:]) / 2
                thresholds[col] = col_thresholds.tolist()
            
        return thresholds

    
    
    # encoding
    def get_encoded_X(self, X, thresholds):
        if self.encoding_method == "1_OUT_OF_K":
            return self.disc_1_out_of_k(X, thresholds)
        elif self.encoding_method == "DIFF_CODING":
            return self.disc_diff_coding(X, thresholds)
        else:
            raise ValueError(f"Unknown encoding method: {self.encoding_method}")
    
    # 1 out of k
    def disc_1_out_of_k(self, X, thresholds):
        encoded_X = []
        # for each column in X_to_encode, create a one-hot encoding of the bins
        for col in X.columns:
            if col in self.categorical:
                bin = pd.Categorical(X[col], categories=thresholds[col]).codes 
                num_bins = len(thresholds[col])
            else:
                bin = np.digitize(X[col], thresholds[col]) # gets bin number of each row
                X[col] = X[col].astype(float) 
                num_bins = len(thresholds[col]) + 1 
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
            encoded_X.append(bins_df)
        
        # concatenate all encoded columns
        encoded_X = pd.concat(encoded_X, axis=1)

        return encoded_X
    
    # differential coding
    def disc_diff_coding(self, X, thresholds):
        encoded_X = []
        for col in X.columns:
            if col in self.categorical:
                bin = pd.Categorical(X[col], categories=thresholds[col]).codes
                num_bins = len(thresholds[col])
            else:
                X[col] = X[col].astype(float)
                bin = np.digitize(X[col], thresholds[col]) # gets bin number of each row
                num_bins = len(thresholds[col]) + 1
            
            bin_df = pd.DataFrame(0, index=X.index, columns=[f'feat{col}-bin{i}' for i in range(1, num_bins)])
            for i in range(1, num_bins):
                bin_df[f'feat{col}-bin{i}'] = (bin >= i).astype(int)
            
            encoded_X.append(bin_df)

        # concatenate all encoded columns
        encoded_X = pd.concat(encoded_X, axis=1)

        return encoded_X
      

    
    # model
    def grid_search_scoring(self, estimator, X, y, val_X, val_y):
        # learn thresholds from the training data
        thresholds = self.get_thresholds(X, y)
        
        # if ordinal problem, do sbc to get binary version
        if self.use_sbc:
            sbc = SBC()
            sbc_X, sbc_y = sbc.reduction(X, y, self.K, self.mapping)
        
            # do thresholds for sbc columns (the last K-2 columns from sbc_X, have one threshold each: 0.5)
            sbc_columns = sbc_X.columns[-(sbc.K-2):]
            for col in sbc_columns:
                thresholds[col] = [0.5]
            
            X = sbc_X.copy()
            y = sbc_y.copy()
    
        # get encoded version of training X
        encoded_X = self.get_encoded_X(X, thresholds)
        
        # fit the model
        estimator.fit(encoded_X, np.ravel(y))

        # VALIDATION
        # if ordinal problem, do sbc to get binary version of validation set
        if self.use_sbc:
            sbc = SBC()
            sbc_val_X, sbc_val_y = sbc.reduction(val_X, val_y, self.K, self.mapping)
            val_X = sbc_val_X.copy()

        # encode the validation set given the thresholds
        encoded_val_X = self.get_encoded_X(val_X, thresholds)
        
        
        # predict with the model
        predictions = estimator.predict(encoded_val_X)
        # tranform predictions to ordinal target
        predictions = sbc.classif(predictions)
        # calculate accuracy
        accuracy = accuracy_score(val_y, predictions)

        return accuracy

    def grid_search(self, model, param_grid, cv=5):
        # get combinations of parameters
        param_combinations = list(ParameterGrid(param_grid))
        
        # save the best parameters and score
        best_score = 0
        best_params = None
        results = []
        
        # for each combination of parameters, do cross-validation
        for params in param_combinations:
            print(f"testing parameters: {params}")
            scores = []
            
            kf = KFold(n_splits=cv, shuffle=True, random_state=42)
            # split the data into train and validation sets
            for train_index, val_index in kf.split(self.train_and_val_X):
                print("  fold ", len(scores) + 1)
                train_X, val_X = self.train_and_val_X.iloc[train_index], self.train_and_val_X.iloc[val_index]
                train_y, val_y = self.train_and_val_y.iloc[train_index], self.train_and_val_y.iloc[val_index]

                # clone the model and set parameters
                model = clone(model)
                model.set_params(**params)
                
                # get the score for the current fold
                score = self.grid_search_scoring(model, train_X, train_y, val_X, val_y)
                scores.append(score)
            
            # calculate the mean score for the current combination of parameters
            mean_score = np.mean(scores)
            results.append((params, mean_score))
            print(f"  mean score: {mean_score}")

            if mean_score > best_score:
                best_score = mean_score
                best_params = params
            
        return best_params, best_score, results

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
            alpha_values = [0.001, 0.01, 0.1, 0.4, 0.6, 0.9, 0.99, 1.0]
            param_grid = {
                'C': [1/a for a in alpha_values], # inverse of regularization strength
                'class_weight': ['balanced', None]
            }

            best_params, best_score, results = self.grid_search(logistic, param_grid)
            self.model = clone(logistic)
            self.model.set_params(**best_params)
            
            best_alpha = 1 / best_params['C']
            print("best parameters: ", best_params)
            print("best alpha: ", best_alpha)
            print("best score: ", best_score)

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
            best_params, best_score, results = self.grid_search(svm,  param_grid, cv=5)
            self.model = clone(svm)
            self.model.set_params(**best_params)
            
            print("best parameters: ", best_params)
            print("best score: ", best_score)

    # beyond l1 (MCP)
    def beyond_l1(self):
        self.model = GeneralizedLinearEstimator(
            datafit=Huber(delta=1.),
            penalty=MCPenalty(alpha=1e-2, gamma=3),
            solver=AndersonCD()
        )

    # adaptive lasso
    def adaptive_lasso(self):
        alasso = AdaptiveLasso(fit_intercept=False)
        if self.params is not None:
            for key, value in self.params.items():
                setattr(alasso, key, value)
            self.model = alasso
        else:
            param_grid = {'alpha': np.logspace(-8, 2, 10)}
            best_params, best_score, results = self.grid_search(alasso, param_grid)
            self.model = clone(alasso)
            self.model.set_params(**best_params)
            
            print("best parameters: ", best_params)
            print("best score: ", best_score)
        
    # risk slim
    def replace_feat(self, col):
        if col.startswith('feat'):
            match = re.match(r'feat(\d+)', col)
            if match:
                idx = int(match.group(1))
                return col.replace(f'feat{idx}', self.X_columns.tolist()[idx])
        return col
    
    def rename_column_names(self, data):
        for col in data.columns[1:]:
            feat_number = col.split('-')[0].replace('feat', '')
            # if feat_number is not a number, skip it
            if not feat_number.isdigit():
                continue
            col_thresholds = self.thresholds[int(feat_number)]
            bin_number = col.split('-')[1].replace('bin', '')
            if bin_number == '0':
                new_col_name = col.split('-')[0] + '_lessorequal' + str(col_thresholds[0])
            else:
                new_col_name = col.split('-')[0] + '_greaterthan' + str(col_thresholds[int(bin_number)-1])
            data.rename(columns={col: new_col_name}, inplace=True)

        #  - feature names to original names
        data.columns = [self.replace_feat(col) for col in data.columns]

    def risk_slim(self):
        X = self.train_and_val_X.copy()
        y = self.train_and_val_y.copy()
        thresholds = self.get_thresholds(X, y)

        if self.use_sbc:
            sbc = SBC()
            sbc_X, sbc_y = sbc.reduction(X, y, self.K, self.mapping)
            sbc_columns = sbc_X.columns[-(sbc.K-2):]
            for col in sbc_columns:
                thresholds[col] = [0.5]
            
            X = sbc_X.copy()
        
        encoded_X = self.get_encoded_X(X, thresholds)
        
        data = encoded_X.copy()
        data.insert(0, 'binary_label', y)
        data['binary_label'] = data['binary_label'].replace({0: -1})
        self.rename_column_names(data)
        data_name = 'datasets/riskslim/' + self.file_name if self.file_name else 'datasets/sbc/risk_slim_data.csv'
        data.to_csv(data_name, index=False)


    def get_weights(self):
        # learn thresholds from the training data
        self.thresholds = self.get_thresholds(self.train_and_val_X, self.train_and_val_y)

        # if ordinal problem, do sbc to get binary version
        sbc = SBC()
        sbc_X, sbc_y = sbc.reduction(self.train_and_val_X, self.train_and_val_y, self.K, self.mapping)

        # do thresholds for sbc columns (the last K-2 columns from sbc_X, have one threshold each: 0.5)
        sbc_columns = sbc_X.columns[-(sbc.K-2):]
        for col in sbc_columns:
            self.thresholds[col] = [0.5]

        # get encoded version of training X
        self.encoded_train_X = self.get_encoded_X(sbc_X, self.thresholds)

        # fit the model
        self.model.fit(self.encoded_train_X, np.ravel(sbc_y))

        # get the weights
        weights = self.model.coef_[0]
        
        # show the features and their weights as a DataFrame
        self.weights = pd.DataFrame({
            'Feature': self.encoded_train_X.columns,
            'Weight': weights
        })
        print("\nFeature weights:")
        print(self.weights)

        # get non-zero weights
        self.non_zero_weights = self.weights[self.weights['Weight'] != 0]
        print("\nNon-zero weights:")
        print(self.non_zero_weights)


    # evaluate the model on the test set
    def evaluate(self):
        if self.use_sbc:
            sbc = SBC()
            sbc_test_X, sbc_test_y = sbc.reduction(self.test_X, self.test_y, self.K, self.mapping)
            test_X = sbc_test_X.copy()
        
        encoded_test_X = self.get_encoded_X(test_X, self.thresholds)

        # predict with the model
        test_predictions = self.model.predict(encoded_test_X)
        
        if self.model_method == "BEYOND_L1" or self.model_method == "ADAPTIVE_LASSO" or self.model_method == "RSS":
            # round weights in y_pred to closer integers
            test_predictions = np.round(test_predictions).astype(int)

        if self.use_sbc:
            # transform predictions to ordinal target
            test_predictions = sbc.classif(test_predictions)

            mapped_test_predictions = sbc.apply_mapping(pd.Series(test_predictions), self.mapping)
            mapped_test_y = sbc.apply_mapping(self.test_y, self.mapping)
        
        # show predictions vs true values side by side
        results_df = pd.DataFrame({'True Value': mapped_test_y.values, 'Prediction': mapped_test_predictions.values})
        print(results_df)

        # calculate and show metrics
        mse = mean_squared_error(mapped_test_y, mapped_test_predictions)
        accuracy = accuracy_score(mapped_test_y, mapped_test_predictions)
        balanced_accuracy = balanced_accuracy_score(mapped_test_y, mapped_test_predictions)
        logistic_loss = np.mean(np.log(1 + np.exp(-mapped_test_predictions * mapped_test_y)))
        # number of far off predictions (more than 1 unit away from the true value)
        if self.use_sbc: far_off = np.sum(np.abs(mapped_test_predictions - mapped_test_y) > 1)
        # number of non zero weights
        num_non_zero_weights = np.sum(self.weights != 0)
        # model size = number of non-zero weights / number of all weights
        model_size = num_non_zero_weights / len(self.weights)
        
        print("mse: ", mse)
        print("accuracy: ", accuracy)
        print("balanced accuracy: ", balanced_accuracy)
        print("logistic loss: ", logistic_loss)
        if self.use_sbc:print("number of far off predictions: ", far_off)
        print("number of non-zero weights: ", num_non_zero_weights)
        print("model size (non-zero weights / all weights): ", model_size)
        
        # confusion matrix
        cm = confusion_matrix(mapped_test_y, mapped_test_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(mapped_test_y), yticklabels=np.unique(mapped_test_y))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
        
        test_predictions_2 = self.model.predict(self.encoded_train_X)
        if self.model_method == "BEYOND_L1" or self.model_method == "ADAPTIVE_LASSO" or self.model_method == "RSS":
            # round weights in y_pred to closer integers
            test_predictions_2 = np.round(test_predictions_2).astype(int)

        if self.use_sbc:
            # transform predictions to ordinal target
            test_predictions_2 = sbc.classif(test_predictions_2)
            mapped_test_predictions_2 = sbc.apply_mapping(pd.Series(test_predictions_2), self.mapping)
            mapped_train_y = sbc.apply_mapping(self.train_y_og, self.mapping)
        
        print("accuracy on train set: ", accuracy_score(mapped_train_y, mapped_test_predictions_2))
        print("mse on train set: ", mean_squared_error(mapped_train_y, mapped_test_predictions_2))
        print("balanced accuracy on train set: ", balanced_accuracy_score(mapped_train_y, mapped_test_predictions_2))
        print("logistic loss on train set: ", np.mean(np.log(1 + np.exp(-mapped_test_predictions_2 * mapped_train_y))))

        
    def show_scorecard(self):        
        # make a table with Feature Name | Bin | Weight
        scorecard_rows = []
        scorecard_table = pd.DataFrame(columns=['Feature', 'Bin', 'Points'])
        for col in self.X.columns:
            # get the weights for the column
            col_weights = self.weights[self.weights['Feature'].str.contains(col)]
            if col_weights.empty:
                continue
            
            # get the bins for the column
            bins = []
            if col in self.categorical:
                bins = self.thresholds[col]
            else:
                thresholds = self.thresholds[col]
                num_bins = len(thresholds) + 1
                for i in range(num_bins):
                    if i == 0:
                        lower = -np.inf
                        upper = thresholds[0]
                    elif i == num_bins - 1:
                        lower = thresholds[-1]
                        upper = np.inf
                    else:
                        lower = thresholds[i - 1]
                        upper = thresholds[i]
                    bins.append(f'bin{i+1}: {{{lower}, {upper}}}')
                    # take first bin
                bins.remove(bins[0])
            
            # add rows to the scorecard - for each col name, for each bin
            for i, bin in enumerate(bins):
                points = col_weights['Weight'].iloc[i]
                # get second part of bin name (e.g. bin1: {lower, upper} -> lower, upper)
                bin_val = bin.split(': ')[1] if ': ' in bin else bin
                bin_val = bin_val.replace('{', '[').replace('}', '[')
                if points != 0.0:
                    scorecard_rows.append({'Feature': col, 'Bin': bin_val, 'Points': points})
        
        scorecard_table = pd.DataFrame(scorecard_rows, columns=['Feature', 'Bin', 'Points'])
        print(scorecard_table)
        
        # get weights of feat called sbc-column
        if self.use_sbc:
            sbc_columns = [col for col in self.weights['Feature'] if col.startswith('featsbcol')]
            sbc_weights = self.weights[self.weights['Feature'].isin(sbc_columns)]
            print("\nSBC columns weights:")
            print(sbc_weights)
            
            

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
        