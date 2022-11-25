from flask import Flask, request
from flask_restful import Api, Resource
from flask_cors import CORS
from resources.LinearModelUnivariate import LinearModelUnivariate
from resources.LinearModelMultivariate import LinearModelMultivariate
from resources.LogisticRegression import LogisticRegression

app  = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
api  = Api(app)

api.add_resource(LinearModelUnivariate, '/api/linear-regression-uni')
api.add_resource(LinearModelMultivariate, '/api/linear-regression-mul')
api.add_resource(LogisticRegression, '/api/logistic-regression')
#api.add_resource(LinearModelAPI, '/neural-network-bin')
#api.add_resource(LinearModelAPI, '/neural-network-bin')
#api.add_resource(LinearModelAPI, '/decision-trees')
#api.add_resource(LinearModelAPI, '/convnet')
#api.add_resource(LinearModelAPI, '/kmeans')

if __name__ == "__main__":
    app.run(debug = True)
