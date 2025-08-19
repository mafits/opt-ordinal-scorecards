import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score

def datareplication (X, K, y = None):
    N = np.shape(X)[0]

    #replicate x
    extra_features = np.concatenate( (np.zeros((1,K-2)), np.identity(K-2)), axis = 0)
    extra_features = np.tile (extra_features, (N,1))
    XX = np.repeat (X, K-1, axis = 0)
    XX= np.concatenate((XX, extra_features), axis = 1)

    #replicate y
    yy = None
    if y is not None:
        aux = np.tri(K).T[1:,:]
        yy = aux[:, y].T.reshape(-1, 1, order='C')

    return XX, yy

############ EXAMPLE OF USING 
def generate_synthetic_data1 (N): #AS IN ARTICLE FROM 2007 data replication method
    points = np.random.uniform(0, 1, size=(N, 2))
    scores = 10*(points[:,0]-0.5)*(points[:,1]-0.5)
    scores = scores + np.random.normal(0, 0.125, np.shape(scores))
    bin_edges = [-1, -0.1, 0.25, 1]
    bin_number = np.digitize(scores, bin_edges)
    K = len (bin_edges) + 1
    return points, bin_number, K


def main():
    X, y, K = generate_synthetic_data1 (10000)    
    # Scatter plot with colormap
    #plt.scatter(X[:,0], X[:,1], s=4, c=y, cmap='viridis')
    #plt.colorbar()  # Add a colorbar to show the mapping
    #plt.show()

    XX, yy = datareplication (X, K, y)
    print("XX: ", XX)
    print("yy: ", yy)

    #test classifier
    #clf = svm.SVC(kernel='poly', degree = 2, C = 100)
    clf = svm.SVC(kernel='rbf', gamma='scale', degree = 2, C = 100)
    clf.fit(XX, np.reshape(yy, (-1)))

    y_pred = clf.predict(XX)  #predict in training data for simplicity
    y_pred = np.reshape (y_pred, (-1, K-1))
    y_pred = np.sum(y_pred, axis=1)

    cm = confusion_matrix(y, y_pred)
    accuracy = accuracy_score(y, y_pred)
    #print (cm, accuracy)

if __name__ == "__main__":
    #print __doc__
    main()


#################### SCORECARD

