from requests import post
import json
import numpy as np

def generate_linear_dataset(w_true, b_true, x_low, x_high, total):
    x_val  = np.linspace(x_low, x_high, num = total)
    f_true = lambda x : (w_true * x + b_true)
    y_true = f_true(x_val) + 0.1 * np.random.randn(x_val.shape[0])
    train  = np.column_stack([x_val, y_true])
    
    return train

w_true = 0.62
b_true = 0.19

train = generate_linear_dataset(w_true, b_true, x_low = -1.0, x_high = 1.0, total = 50)
print(train)
print(train.shape)

dataset = {
    'method' : 'direct',
    'train_x': train[:, 0].tolist(),
    'train_y': train[:, 1].tolist(),
    'epochs' : 100,
    'learning_rate': 0.1
}

response = post('http://127.0.0.1:5000/api/linear-regression-uni', headers = { 'Content-Type' : 'application/json' }, data = json.dumps(dataset)).json()

print(response['weight'], response['bias'])
