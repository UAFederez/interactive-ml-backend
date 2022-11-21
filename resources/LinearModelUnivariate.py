from flask import Flask, request
from flask_restful import Api, Resource
import json
import numpy as np

class LinearModelUnivariate(Resource):
    def calc_gradients(self, w, b, x_train, y_true, y_pred):
        grad_w = np.dot(y_pred - y_true, x_train)
        grad_b = np.mean((y_pred - y_true))
        
        return grad_w, grad_b

    def calc_optimal_params_grad_descent(self, epochs, learning_rate, x_train, y_train):
        w = np.random.randn(1) * 5.0
        b = np.random.randn(1) * 5.0
        L = np.random.randn(epochs)
        
        param_hist = np.zeros((epochs, 2))

        for i in range(epochs):
            y_pred   = w * x_train + b
            cost_val = np.mean((y_pred - y_train) ** 2) / 2.0
            
            param_hist[i, 0] = w[0]
            param_hist[i, 1] = b[0]
            
            grad_w, grad_b = self.calc_gradients(w, b, x_train, y_train, y_pred)
            grad_w, grad_b
            
            L[i] = cost_val

            w = w - learning_rate * grad_w
            b = b - learning_rate * grad_b    

        return w[0], b[0], L, param_hist

    def calc_optimal_params_system_of_eq(self, x_train, y_train):
        m = x_train.shape[0]

        w_denom = (m * np.sum(x_train ** 2)) - (np.sum(x_train) ** 2)
        w_numer = (m * np.dot(y_train, x_train)) - (np.sum(y_train) * np.sum(x_train))

        w_new = w_numer / w_denom
        w_new

        b_numer = (np.sum(y_train) * np.sum(x_train ** 2)) - (np.dot(y_train, x_train) * np.sum(x_train))
        b_denom = (m * np.sum(x_train ** 2)) - (np.sum(x_train)) ** 2

        b_new = b_numer / b_denom

        return (w_new, b_new)

    def post(self):
        data = request.json
        train_x        = np.array(data['train_x'])
        train_y        = np.array(data['train_y'])
        method         = data['method']

        if method == 'gradient_descent':
            epochs         = data['epochs']
            learning_rate  = data['learning_rate']
            w_gd, b_gd, loss_hist, param_hist = self.calc_optimal_params_grad_descent(epochs, learning_rate, train_x, train_y)
            response = {
                'weight'    : w_gd,
                'bias'      : b_gd,
            }
            if 'include_hist' in data and data['include_hist']:
                response['loss_hist']  = loss_hist.tolist()
                response['param_hist'] =  param_hist.tolist()

        elif method == 'direct':
            w_se, b_se = self.calc_optimal_params_system_of_eq(train_x, train_y)
            response = {
                'status' : 'success',
                'weight' : w_se,
                'bias'   : b_se,
            }
        else:
            response = {
                'status': 'invalid method',
            }
        return response

