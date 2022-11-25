from flask import Flask, request
from flask_restful import Api, Resource
import json
import numpy as np

class LinearModelMultivariate(Resource):
    def calc_optimal_params_grad_descent(self, train_X, train_Y, num_epochs, learning_rate):    
        n, m = train_X.shape
        w    = np.random.randn(n, 1)

        param_hist = np.zeros((num_epochs, n))
        
        loss_hist = np.zeros(num_epochs)
        
        for i in range(num_epochs):
            Y_pred = np.dot(w.T, train_X)
            J_cost = np.mean((Y_pred - train_Y) ** 2) / 2.0
            loss_hist[i] = J_cost
            
            # Calculate gradients
            grad_W = np.dot(train_X, (Y_pred - train_Y).T) / m
            param_hist[i] = w[:, 0]
            
            # Update the weights
            w = w - (learning_rate * grad_W)
            
        return w, loss_hist, param_hist

    def post(self):
        data = request.json

        train_x        = np.array(data['train_x'])
        train_y        = np.array(data['train_y'])
        epochs         = data['epochs']
        learning_rate  = data['learning_rate']
        method         = data['method']

        response = dict()

        if method == 'gradient_descent':
            weights, loss_hist, param_hist = self.calc_optimal_params_grad_descent(train_x, train_y, epochs, learning_rate)
            response = {
                'status'     : 'success',
                'weights'    : weights[:, 0].tolist(),
            }
            if 'include_hist' in data and data['include_hist']:
                response['history'] = {
                    'loss'    : loss_hist.tolist(),
                    'weights' : param_hist.tolist()
                }
                
        elif method == 'normal_eq':
            weights = np.dot(np.linalg.inv(np.dot(train_x, train_x.T)), np.dot(train_x, train_y.T))
            response = {
                'status' : 'success',
                'weights': weights[:,0].tolist(),
            }
        else:
            response = {
                'status': 'invalid_method',
            }
        return response

