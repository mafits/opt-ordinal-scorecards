import numpy as np
import pandas as pd


class SBC():
    def __init__(self):
        self.K = None
        self.s = None
        self.mapping = None
        self.sbc_column = None
    
    
    def reduction(self, X, y, K=None, mapping=None, h=1, s=None):
        # num of classes
        if K is not None:
            self.K = K
        else:
            self.K = len(np.unique(y))
        #print("original number of features: ", X.shape[1])
        #print("original number of target classes: ", self.K)
        #print("original number of observations: ", X.shape[0])
        
        # num of classes defining each hyperplane
        if s is None:
            self.s = self.K-1
        else:
            self.s = s
        
        # if class labels not integer, convert to integer
        if not np.issubdtype(y.dtype, np.integer):
            # if a mapping is not provided, create one
            if mapping is None:
                new_y = pd.Series(pd.factorize(y)[0])
                self.mapping = dict(enumerate(pd.factorize(y)[1]))
                # show the mapping 
                #print("using mapping: ", self.mapping)
                y = new_y
            else:
                # if mapping starts with 0, shift it to start with 1
                if 0 in mapping.keys():
                    #print("mapping starts with 0, shifting to start with 1")
                    mapping = {k+1: v for k, v in mapping.items()}
               
                # apply mapping dictionary to y
                #print("using mapping: ", mapping)
                y = pd.Series(y.map(lambda v: {v_:k_ for k_, v_ in mapping.items()}[v]))
                self.mapping = mapping
        
        # for each point, create (K-1) replicas each with (K-2) new features
        # the new target label is a binary label
        new_X = []
        new_y = []
        for i in range(X.shape[0]): # for each point
            k = y.iloc[i]  # original class label
            for q in range(self.K - 1):  # for each replica (= number of hyperplanes)
                new_variables = [0] * (self.K - 2) # create (K-2) new variables
                
                # if q is not the first replica, set the (q-1)th variable to 1
                if q > 0:
                    new_variables[q-1] = 1
                
                # create a new point by concatenating the original point with the new variables
                new_point = np.concatenate((X.iloc[i, :], new_variables))
                new_X.append(new_point)
                
                # create the binary label
                if k-1 <= q: 
                    new_y.append(0) # C1  
                else:  
                    new_y.append(1) # C2

        new_X = pd.DataFrame(new_X).reset_index(drop=True)
        # rename last (K-2) columns to sbcol1, sbcol2, ..., sbcol(K-2)
        new_X.columns = list(X.columns) + [f'sbcol{i+1}' for i in range(self.K-2)]
        
        new_y = pd.DataFrame(new_y).reset_index(drop=True)
        # rename binary label column
        new_y.columns = ['binary_label']
        
        new_data = pd.concat([new_X, new_y], axis=1)

        # print some information about the new data
        #print("new number of features: ", new_X.shape[1], " (original number of features +", self.K - 2, ")")
        #print("new number of target classes: ", len(np.unique(new_y)))
        #print("new number of observations: ", new_X.shape[0], " (original number of observations *", self.K - 1, ")")
        #print(new_data.head())
        
        return new_X, new_y
    
    
    
    def classif(self, pred_sbc_y, do_mapping=True, show_print=True):
        # get classification of all replicas of each point
        all_labels = [pred_sbc_y[i:i + self.s] for i in range(0, len(pred_sbc_y), self.s)]
        all_labels = np.array(all_labels)
        #if show_print:
            #print(all_labels)

        # get the class of the point
        
        # SISTEMA DE VOTOS
        # if all replicas are 0, then the class is K
        # if one is 1 and the rest are 0, then the class is K-1
        # if two are 1 and the rest are 0, then the class is K-2
        # ...
        # if all replicas are 1, then the class is 1
        final_labels = np.sum(all_labels, axis=1) + 1
        
        
        # TAMANHO DA SEQUENCIA DE 1s
        '''final_labels = []
        for binary_labels in all_labels:
            count_ones = 0
            for label in binary_labels:
                if label == 1:
                    count_ones += 1
                else:
                    break
            final_labels.append(count_ones + 1)'''

        #if show_print:
            #print("predicted labels (before mapping): ", final_labels)

        # transform back to original labels using the mapping
        if do_mapping == True and self.mapping is not None:
            #if show_print:
               # print("mapping: ", self.mapping)
            final_labels = pd.Series(final_labels).map(self.mapping).values
            #if show_print:
                #print("predicted labels (after mapping): ", final_labels)

        return final_labels
    
    
    def apply_mapping(self, y, mapping=None):
        if mapping is not None:
            self.mapping = mapping
        if (y.dtype == 'int'):
            return pd.Series(y).map(self.mapping)
        else:
            return pd.Series(y.map(lambda v: {v_:k_ for k_, v_ in self.mapping.items()}[v]))
