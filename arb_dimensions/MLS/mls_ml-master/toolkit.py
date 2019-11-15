import numpy as np
from numpy import linalg as LA
from sklearn.linear_model import LinearRegression

def data_split(data):
    '''
    This function just gives us a randomized split of the data into 80% training
    data and 20% testing data. It returns a list of lists. The first element is
    the list of training indices, the second element is the list of testing
    indices. 
    '''
    indices = np.random.permutation(data[0].shape[0])
    split_index = np.floor(0.80*len(indices))
    training_indices = indices[:split_index.astype(int)] 
    test_indices     = indices[split_index.astype(int):]
    return [training_indices, test_indices] 

def linear_model(features, labels, training_indices):
    '''
    This function is using the scikit-learn package to build our linear
    regression models.
    '''
    return LinearRegression().fit(features[training_indices],
                                  labels[training_indices])

def evaluate_model(model,predictions,actual):
    '''
    This functions is going to build an array that will be written as a csv
    file. The first entry of the csv file will be the function built by our
    regression method. The second will be the error found on the test set.
    '''
    return np.array([[str(model.intercept_) + 
                        ' + '+ str(model.coef_[0]) + 'x'],
                       [LA.norm(np.subtract(predictions,actual))]])
