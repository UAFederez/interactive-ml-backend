from flask import Flask, request
from flask_restful import Api, Resource
import json
import numpy as np

class NeuralNetworkImpl:
    def __init__(self, weights, biases, activations, loss_function):
        self.weights = weights
        self.biases  = biases
        self.layer_activations = activations
        self.loss_function = loss_function

        self.activation = {
            'sigmoid': lambda z : 1 / (1 + np.exp(-z)),
            'tanh'   : lambda z : np.tanh(z),
            'relu'   : lambda z : np.maximum(0, z)
        }
        self.activation_prime = {
            'sigmoid': lambda a_out, z_out : a_out * (1 - a_out),
            'tanh'   : lambda a_out, z_out : 1 - (np.tanh(a_out) ** 2),
            'relu'   : lambda a_out, z_out : a_out > 0.0
        }
        self.losses_func = {
            'binary_crossentropy': lambda y_pred, y_true: -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        }
        
        self.losses_prime = {
            'binary_crossentropy': lambda y_pred, y_true: (y_pred - y_true) / (y_pred * (1 - y_pred))
        }
        self.metrics_func = {
            'binary_class_accuracy': lambda y_pred, y_true: np.mean(np.round(y_pred) == y_true)
        }
        
        self.loss_func  = self.losses_func[loss_function]
        self.loss_prime = self.losses_prime[loss_function] 

    @staticmethod
    def from_architecture(layer_sizes, activations, loss_function):
        rng = np.random.default_rng()
        weights = [ rng.standard_normal((layer_sizes[idx], n)) * (1.0 / np.sqrt(2.0 * layer_sizes[idx]))
                         for idx, n in enumerate(layer_sizes[1:]) ]
        biases  = [ np.zeros((n, 1)) for n in layer_sizes[1:] ]
        layer_activations = activations

        return NeuralNetworkImpl(weights, biases, layer_activations, loss_function)
       
        
    def train(self, train_X, train_Y, num_epochs, learning_rate, metrics = None):
        len_ds = train_X.shape[1]
        
        loss_hist    = [ 0 for x in range(num_epochs) ]
        metrics_hist = { m : [] for m in metrics } 
        
        for epoch in range(num_epochs):
            z_outs = [ train_X ]
            a_outs = [ train_X ]
        
            # Forward pass
            for idx, (weights, biases) in enumerate(zip(self.weights, self.biases)):
                g_func = self.activation[self.layer_activations[idx]]
                Z_prev = a_outs[idx]
                Z_next = np.dot(weights.T, Z_prev) + biases
                A_next = g_func(Z_next)
                
                z_outs.append(Z_next)
                a_outs.append(A_next)
                
            # Compute loss
            loss = np.mean(self.loss_func(a_outs[-1], train_Y))
            loss_hist[epoch] = loss
            
            for metric in metrics:
                metrics_hist[metric].append(self.metrics_func[metric](a_outs[-1], train_Y))
            
            # Backward pass
            dLoss_dAs = [ self.loss_prime(a_outs[-1], train_Y) ]
            
            dLoss_dws = []
            dLoss_dbs = []
            
            for idx in reversed(range(len(self.weights))):
                a_curr = a_outs[idx + 1]
                z_curr = z_outs[idx + 1]
                a_prev = a_outs[idx]
                
                g_prime  = self.activation_prime[self.layer_activations[idx]]
                dLoss_dZ = dLoss_dAs[len(self.weights) - 1 - idx] * g_prime(a_curr, z_curr)
                
                dLoss_dw = np.dot(a_prev, dLoss_dZ.T) / len_ds
                dLoss_db = np.mean(dLoss_dZ, axis = 1, keepdims = True)
                
                dLoss_dws.insert(0, dLoss_dw)
                dLoss_dbs.insert(0, dLoss_db)
                
                dLoss_dAprev = np.dot(self.weights[idx], dLoss_dZ)
                dLoss_dAs.append(dLoss_dAprev)
            
            # Update step
            for idx in range(len(self.weights)):
                self.weights[idx] = self.weights[idx] - (dLoss_dws[idx] * learning_rate)
                self.biases[idx]  = self.biases[idx]  - (dLoss_dbs[idx] * learning_rate)
            
        return loss_hist, metrics_hist
    
    def predict(self, test_X):
        z_outs = test_X
        a_outs = test_X
        
        for idx, (weights, biases) in enumerate(zip(self.weights, self.biases)):
            g_func = self.activation[self.layer_activations[idx]]
            Z_prev = a_outs
            Z_next = np.dot(weights.T, Z_prev) + biases
            A_next = g_func(Z_next)

            z_outs = Z_next
            a_outs = A_next
                
        return a_outs[-1]

class NeuralNetwork(Resource):
    
    def serialize_model(self, neural_net):
        return { 
            'weights' : [ weight_matrix.tolist() for weight_matrix in neural_net.weights ],
            'biases'  : [ biases_vector.tolist() for biases_vector in neural_net.biases ],
            'layer_activations': [ layer_activation for layer_activation in neural_net.layer_activations ],
            'loss_function' : neural_net.loss_function
        }

    def predict_model(self, data):
        data = request.json
        train_x = np.array(data['train_x'])

        print('predict', train_x.shape)

        weights = [ np.array(weight_matrix) for weight_matrix in data['model']['weights'] ]
        biases  = [ np.array(biases_vector) for biases_vector in data['model']['biases'] ]
        layer_activations = [ layer_activation for layer_activation in data['model']['layer_activations'] ]
        loss_function = data['model']['loss_function']

        model  = NeuralNetworkImpl(weights, biases, layer_activations, loss_function)
        y_pred = model.predict(train_x)

        return {
            'status': 'success',
            'y_pred': y_pred.tolist()
        }

    def train_model(self, data):
        train_x = np.array(data['train_x'])
        train_y = np.array(data['train_y'])
        epochs         = data['epochs']
        learning_rate  = data['learning_rate']

        print('train', train_x.shape)

        model = NeuralNetworkImpl.from_architecture(
            data['layer_sizes'], 
            data['layer_activations'], 
            data['loss_function']
        )

        loss_hist, metrics_hist = model.train(train_x, train_y, 
                                              num_epochs = min(epochs, 1000), 
                                              learning_rate = learning_rate, 
                                              metrics = ['binary_class_accuracy'])

        response = {
            'status' : 'success',
            'model'  : self.serialize_model(model)
        }

        if 'include_hist' in data and data['include_hist'] == True:
            response['history'] = {
                'loss'      : loss_hist,
                'metrics'   : metrics_hist
            }

        return response

    def post(self):
        data = request.json
        if 'predict_from_x' in data and data['predict_from_x']:
            return self.predict_model(data)
        else:
            return self.train_model(data)
