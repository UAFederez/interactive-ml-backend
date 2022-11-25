from requests import post
import json
from sklearn.datasets import make_classification
import numpy as np

X_data, Y_data = make_classification(n_samples     = 100, 
                                     n_features    = 2, 
                                     n_informative = 1, 
                                     n_redundant   = 0,  
                                     n_clusters_per_class = 1,
                                     random_state  = 69)
X_data = X_data.T
Y_data = Y_data.reshape(1, -1) 

request_body = {
    'train_x': X_data.tolist(),
    'train_y': Y_data.tolist(),
    'epochs' : 100,
    'learning_rate': 0.25,
    'include_hist': True
}

print(X_data.shape)

response = post('http://127.0.0.1:5000/api/logistic-regression',
                headers = {
                    'Content-Type' : 'application/json'
                },
                data = json.dumps(request_body)).json() 

print(json.dumps(response, indent = 1))
