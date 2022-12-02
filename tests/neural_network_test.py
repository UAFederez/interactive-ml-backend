from requests import post, get
import json
import numpy as np

import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.utils import shuffle
from sklearn.datasets import make_classification

x1, y1 = make_circles(factor = 0.8, noise = 0.01, random_state=32)

X = x1.T
Y = y1.reshape(-1, 1).T


request_body = {
    'train_x' : X.tolist(),
    'train_y' : Y.tolist(),
    'epochs'  : 200,
    'learning_rate': 0.25,
    'layer_sizes': [2, 32, 32, 32, 1],
    'layer_activations': ['relu', 'relu', 'relu', 'sigmoid'],
    'loss_function': 'binary_crossentropy'
}

post_response = post('http://127.0.0.1:5000/api/neural-network-bin',
                headers = {
                    'Content-Type' : 'application/json'
                },
                data = json.dumps(request_body)).json() 

x1_min = np.min(X.T[:, 0]) - 0.25
x1_max = np.max(X.T[:, 0]) + 0.25
x2_min = np.min(X.T[:, 1]) - 0.25
x2_max = np.max(X.T[:, 1]) + 0.25

(x1_min, x1_max), (x2_min, x2_max)
x1 = np.linspace(x1_min, x1_max, 75)
x2 = np.linspace(x2_min, x2_max, 75)

print(x1_min, x1_max, x2_min, x2_max)
pred_data = np.array([ (x, y) for y in x2 for x in x1 ]).T

request_body = {
        'predict_from_x': True,
        'train_x': pred_data.tolist(),
        'model': post_response['model'],
}

post_response = post('http://127.0.0.1:5000/api/neural-network-bin',
                headers = {
                    'Content-Type' : 'application/json'
                },
                data = json.dumps(request_body)).json() 

yp = post_response['y_pred']

pred = np.row_stack([pred_data, yp]).T

fig, ax = plt.subplots()
fig.set_size_inches((6, 6))
ax.scatter(pred[:, 0], pred[:, 1], c = pred[:, 2], marker = ',', cmap = 'coolwarm')
ax.scatter(X.T[:, 0], X.T[:, 1], c = Y.T, marker = 'o',  cmap = 'coolwarm', edgecolors='black')
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title('Decision bounds of the neural network')
plt.show()

print(yp)
