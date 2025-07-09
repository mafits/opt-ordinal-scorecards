import numpy as np
import pandas as pd


class SBC():
    def __init__(self):
        self.K = None
        self.s = None
        self.mapping = None
        self.sbc_column = None
    
    
    def reduction(self, X, y, mapping=None, h=1):
        # num of classes
        self.K = len(np.unique(y))
        
        # print some information about the original data
        print("number of features: ", X.shape[1])
        print("original num target classes: ", self.K)
        print("original num observations: ", X.shape[0])
        
        # num of parallel hyperplanes to be created (and replicas)
        self.s = self.K-1
        
        # if class labels not integer, convert to integer
        if not np.issubdtype(y.dtype, np.integer):
            if mapping is None:
                new_y = pd.Series(pd.factorize(y)[0])
                self.mapping = dict(enumerate(pd.factorize(y)[1]))
                # show the mapping 
                print("mapping: ", self.mapping)
                y = new_y
            else:
                # apply mapping dictionary to y
                print("using provided mapping: ", mapping)
                y = pd.Series(y.map(lambda v: {v_:k_ for k_, v_ in mapping.items()}[v]))
                self.mapping = mapping
        
        # for each point, create s replicas each with a new feature in [0, h, h*2, ... h*(s-1)]
        # the new label is a binary label
        new_X = []
        new_y = []
        for i in range(X.shape[0]): # for each point
            for j in range(self.s): # for each replica
                new_X.append(np.append(X.iloc[i].values, h*j)) 
                new_label = y.iloc[i] > j
                new_y.append(new_label.astype(int))
        
        new_X = pd.DataFrame(new_X).reset_index(drop=True)
        new_y = pd.DataFrame(new_y).reset_index(drop=True)
        new_data = pd.concat([new_X, new_y], axis=1)
        # rename binary label column
        new_data.columns = list(new_X.columns) + ['binary_label']
        new_y = new_y.rename(columns={0: 'binary_label'})
        
        # rename the last column to 'sbc_value'
        new_X.rename(columns={new_X.columns[-1]: 'sbc_value'}, inplace=True)
        self.sbc_column = 'sbc_value'
       
        # print some information about the new data
        print("new num features: ", new_X.shape[1])
        print("new num target classes: ", len(np.unique(new_y)))
        print("new num observations: ", new_X.shape[0], " (original num observations *", self.s, ")")
        print(new_data.head())
        
        return new_X, new_y
    
    
    
    def classif(self, pred_sbc_y, do_mapping=True):
        # get classification of all replicas of each point
        all_labels = [pred_sbc_y[i:i + self.s] for i in range(0, len(pred_sbc_y), self.s)]
        all_labels = np.array(all_labels)
        print(all_labels)
        
        # get the class of the point
        # if all replicas are 0, then the class is 0
        # if one is 1 and the rest are 0, then the class is 1
        # if two are 1 and the rest are 0, then the class is 2
        # ...
        # if all replicas are 1, then the class is K
        final_labels = np.sum(all_labels, axis=1)
        print("predicted labels (before mapping): ", final_labels)
        
        # transform back to original labels using the mapping
        if do_mapping == True and self.mapping is not None:
            print("mapping: ", self.mapping)
            final_labels = pd.Series(final_labels).map(self.mapping).values
            print("predicted labels (after mapping): ", final_labels)
        
        return final_labels
    
    
    def apply_mapping(self, y):
        if (y.dtype == 'int'):
            return pd.Series(y).map(self.mapping)
        else:
            return pd.Series(y.map(lambda v: {v_:k_ for k_, v_ in self.mapping.items()}[v]))
