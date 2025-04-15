import numpy as np
import pandas as pd


class SBC():
    def __init__(self):
        self.K = None
        self.s = None
    
    
    def reduction(self, X, y, h=1):
        # num of classes
        self.K = len(np.unique(y))
        
        print("original num classes: ", self.K)
        print("original num observations: ", X.shape[0])
        
        # num of parallel hyperplanes to be created (and replicas)
        self.s = self.K-1
        
        # if class labels not integer, convert to integer
        if not np.issubdtype(y.dtype, np.integer):
            new_y = pd.Series(pd.factorize(y)[0])
            # show the mapping
            mapping = pd.Series(pd.factorize(y)[1], index=np.unique(new_y))
            print("mapping: ", mapping)
            y = new_y
        
        # for each point, create s replicas each with a new feature in [0, h, h*2, ... h*(s-1)]
        # the new label is a binary label
        new_X = []
        new_y = []
        for i in range(X.shape[0]): # for each point
            for j in range(self.s): # for each replica
                new_X.append(np.append(X.iloc[i].values, h*j))
                new_label = y.iloc[i] <= j
                new_y.append(new_label.astype(int))
        
        new_X = pd.DataFrame(new_X).reset_index(drop=True)
        new_y = pd.DataFrame(new_y).reset_index(drop=True)
        new_data = pd.concat([new_X, new_y], axis=1)
        # rename binary label column
        new_data.columns = list(new_X.columns) + ['binary_label']
        
        print("new num classes: ", len(np.unique(new_y)))
        print("new num observations: ", new_X.shape[0], " (original num observations *", self.s, ")")
        
        print(new_data.head())
        
        return new_X, new_y, new_data
    
    
    
    def classif(self, pred_sbc_y):
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
        final_labels = np.argmax(all_labels, axis=1)    
        
        return final_labels