from flask import Flask, request
from flask_restful import Api, Resource
import json
import numpy as np

class LogisticRegression(Resource):
    def calc_optimal_params_grad_descent(self, train_X, train_Y, num_epochs, learning_rate):
        # Additional column for the bias
        bias_row = np.full((1, train_X.shape[1]), 1)
        train_X  = np.row_stack([train_X, bias_row ])
        
        n, m = train_X.shape
        w    = np.random.randn(n, 1)
        print(w.shape)

        param_hist    = np.zeros((num_epochs, n))
        accuracy_hist = []
        
        loss_hist = np.zeros(num_epochs)
        
        for i in range(num_epochs):
            Z_val  = np.dot(w.T, train_X)
            Y_pred = 1 / (1 + np.exp(-Z_val))
            
            bce_loss = np.mean((Y_pred - train_Y) ** 2) / 2.0
            loss_hist[i] = bce_loss

            accuracy_hist.append(np.mean(np.round(Y_pred) == train_Y))

            param_hist[i] = w[:, 0]
            
            # Calculate gradients
            grad_W = np.dot(train_X, (Y_pred - train_Y).T) / m

            # Update the weights
            w = w - (learning_rate * grad_W)
            
        return w, loss_hist, param_hist, accuracy_hist

    def post(self):
        data = request.json
        train_x = np.array(data['train_x'])
        train_y = np.array(data['train_y'])
        epochs         = data['epochs']
        learning_rate  = data['learning_rate']

        weights, loss_hist, param_hist, accuracy_hist = self.calc_optimal_params_grad_descent(train_x, train_y, epochs, learning_rate)

        response = {
            'status' : 'success',
            'weights': weights[:,0].tolist()
        }

        if 'include_hist' in data and data['include_hist'] == True:
            response['history'] = {
                'loss'      : loss_hist.tolist(),
                'param_hist': param_hist.tolist(),
                'accuracy_hist': accuracy_hist
            }

        return response


