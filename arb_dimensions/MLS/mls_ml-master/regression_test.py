import numpy as np
import toolkit as tk
import non_eval
from numpy import linalg as LA
from sklearn.linear_model import LinearRegression

simple_filtered_data_file = non_eval.main("./Input/simple_data.csv",100, degree_approx= 3)
crazy_filtered_data_file = non_eval.main("./Input/crazy_data.csv",100,degree_approx = 3)

simple_data  = np.transpose(
               np.genfromtxt(simple_filtered_data_file, delimiter = ','))

crazy_data   = np.transpose(
               np.genfromtxt(crazy_filtered_data_file, delimiter = ','))

simple_features = simple_data[0].reshape(-1,1)
simple_labels   = simple_data[1]

crazy_features = crazy_data[0].reshape(-1,1)
crazy_labels   = crazy_data[1]

simple_data_indices = tk.data_split(simple_data)
crazy_data_indices  = tk.data_split(crazy_data)

simple_model = tk.linear_model(simple_features,
                               simple_labels,
                               simple_data_indices[0])

crazy_model  = tk.linear_model(crazy_features,
                               crazy_labels,
                               crazy_data_indices[0])

simple_predictions = simple_model.predict(
                     simple_features[simple_data_indices[1]])

crazy_predictions  = crazy_model.predict(
                     crazy_features[crazy_data_indices[1]])

simple_evaluation = tk.evaluate_model(simple_model,
                                      simple_predictions,
                                      simple_data[1][simple_data_indices[1]])

crazy_evaluation = tk.evaluate_model(crazy_model,
                                     crazy_predictions,
                                     crazy_data[1][crazy_data_indices[1]])

np.savetxt('Output/simple_evaluation.csv', simple_evaluation, delimiter= ',', fmt='%s')
np.savetxt('Output/crazy_evaluation.csv', crazy_evaluation, delimiter= ',', fmt='%s')
