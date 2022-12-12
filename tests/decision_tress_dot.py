from requests import post
import json
from sklearn.datasets import make_classification
import pandas as pd
import graphviz
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.utils import shuffle
from sklearn.datasets import make_classification

x1, y1 = make_circles(factor = 0.8, noise = 0.01, random_state=32)

X = x1.T
Y = y1.reshape(-1, 1).T

#plt.scatter(x = X[0, :], y = X[1, :], c = Y[0, :])
#plt.show()

train_features = [
    {
        'name': 'feature1',
        'values': X[0, :].tolist(),
    },
    {
        'name': 'feature2',
        'values': X[1, :].tolist(),
    }
]
train_labels = {
    'name': 'target',
    'values': Y[0, :].tolist()
}

request_body = {
    'task': 'train',
    'train_features': train_features,
    'train_labels': train_labels,
}

#print(json.dumps(request_body, indent = 4))


response = post('http://127.0.0.1:5000/api/decision-trees', 
                headers = {
                    'Content-Type' : 'application/json'
                }, 
                data = json.dumps(request_body)).json()

#src = graphviz.Source(response['graphviz_output'], format="png")
#src.view()
#print(json.dumps(response, indent = 4))

test_data = []

for idx in range(X.shape[1]):
    test_data.append({ 
        'feature1': float(X[0, idx]),
        'feature2': float(X[1, idx])
    })

print(json.dumps(response['tree_serialized'], indent = 4))

request_body = {
    'task': 'predict',
    'serialized_tree': response['tree_serialized'],
    'test_data': test_data
}

response = post('http://127.0.0.1:5000/api/decision-trees', 
                headers = {
                    'Content-Type' : 'application/json'
                }, 
                data = json.dumps(request_body)).json()

y_pred = np.array(response['predicted'])
y_true = Y[0, :]

print('y_pred:', y_pred)
print('y_true:', y_true)
print('accuracy', np.mean(y_pred == y_true))
