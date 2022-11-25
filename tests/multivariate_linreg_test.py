from requests import post
import json
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

diabetes_X, diabetes_Y = datasets.load_diabetes(return_X_y = True)
diabetes_X = diabetes_X.T
diabetes_X = np.row_stack([diabetes_X, np.full((1, diabetes_X.shape[1]), 1)])
diabetes_Y = diabetes_Y[:, np.newaxis].T

dataset = {
    'method' : 'normal_eq',
    'train_x': diabetes_X.tolist(),
    'train_y': diabetes_Y.tolist(),
    'epochs' : 100,
    'learning_rate': 0.1,
    'include_hist': True,
}

response = post('http://127.0.0.1:5000/api/linear-regression-mul', 
                headers = {
                    'Content-Type' : 'application/json'
                }, 
                data = json.dumps(dataset)).json()

print(json.dumps(response, indent = 1))
