from flask import Flask, request
from flask_restful import Api, Resource
from resources.LinearModelUnivariate import LinearModelUnivariate
from resources.LinearModelMultivariate import LinearModelMultivariate

app = Flask(__name__)
api = Api(app)

api.add_resource(LinearModelUnivariate, '/api/linear-regression-uni')
api.add_resource(LinearModelMultivariate, '/api/linear-regression-mul')
#api.add_resource(LinearModelAPI, '/logistic-regression-bin')
#api.add_resource(LinearModelAPI, '/logistic-regression-mul')
#api.add_resource(LinearModelAPI, '/neural-network-bin')
#api.add_resource(LinearModelAPI, '/neural-network-bin')
#api.add_resource(LinearModelAPI, '/decision-trees')
#api.add_resource(LinearModelAPI, '/kmeans')

if __name__ == "__main__":
    app.run(debug = True)
