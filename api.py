from flask import Flask, request
from flask_restful import Api, Resource
from flask_cors import CORS
from resources.LinearModelUnivariate import LinearModelUnivariate
from resources.LinearModelMultivariate import LinearModelMultivariate
from resources.LogisticRegression import LogisticRegression
from resources.NeuralNetworks import NeuralNetwork
from resources.KMeansClustering import KMeansClustering
from resources.DecisionTrees import DecisionTrees

app  = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
api  = Api(app)

api.add_resource(LinearModelUnivariate, '/api/linear-regression-uni')
api.add_resource(LinearModelMultivariate, '/api/linear-regression-mul')
api.add_resource(LogisticRegression, '/api/logistic-regression')
api.add_resource(NeuralNetwork, '/api/neural-network')
api.add_resource(KMeansClustering, '/api/kmeans-clustering')
api.add_resource(DecisionTrees, '/api/decision-trees')

if __name__ == "__main__":
    app.run(debug = True)
