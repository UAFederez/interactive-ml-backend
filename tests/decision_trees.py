from requests import post
import json
from sklearn.datasets import make_classification
import pandas as pd
import graphviz
import numpy as np

dataset = pd.read_csv('datasets/titanic_train.csv')
dataset = dataset.loc[:, ~dataset.columns.isin(['Cabin', 'Name', 'PassengerId', 'Ticket'])]
dataset.loc[dataset['Age'].isna(), 'Age'] = dataset['Age'].mean()
dataset.loc[dataset['Embarked'].isna(), 'Embarked'] = dataset['Embarked'].value_counts().idxmax()

dataset['Sex_YN']  = dataset['Sex'].replace({ 'male': 0, 'female': 1 })
dataset['Age_z'] = (dataset['Age'] - dataset['Age'].mean()) / dataset['Age'].std()
dataset['Parch_z'] = (dataset['Parch'] - dataset['Parch'].mean()) / dataset['Parch'].std()
dataset['Fare_z']  = (dataset['Fare'] - dataset['Fare'].mean()) / dataset['Fare'].std()
dataset['Pclass_z']  = (dataset['Pclass'] - dataset['Pclass'].mean()) / dataset['Pclass'].std()

target     = 'Survived'
predictors = [ 'Sex_YN', 'Age_z', 'Parch_z', 'Fare_z', 'Pclass_z']

for label in dataset['Embarked'].unique():
    dataset['Embarked_' + label] = (dataset['Embarked'] == label).replace({ True: 1, False: 0 })
    dataset['Embarked_' + label] = dataset['Embarked_' + label].astype('category')
    predictors.append('Embarked_' + label)
    
    
dataset = dataset.loc[:, predictors + [target]]
dataset = dataset.astype({
    'Sex_YN': 'category',
    'Survived': 'category',
})
target_vc = dataset['Survived'].value_counts()
dataset = pd.concat([dataset, dataset[dataset[target] == 1].sample(frac = target_vc[0] / target_vc[1], replace = True, random_state = 0)])
dataset = dataset.iloc[np.random.permutation(len(dataset))].reset_index(drop = True)

train_features = []

for pred in predictors:
    train_features.append({
        'name'  : pred,
        'values': dataset[pred].tolist()
    })

request_body = {
    'task': 'train',
    'train_features': train_features,
    'train_labels'  : {
        'name'  : target,
        'values': dataset[target].tolist()
    }
}

print(json.dumps(request_body, indent = 4))

#print({feature['name'] : feature['values'] for feature in request_body['train_features']})

response = post('http://127.0.0.1:5000/api/decision-trees', 
                headers = {
                    'Content-Type' : 'application/json'
                }, 
                data = json.dumps(request_body)).json()

src = graphviz.Source(response['graphviz_output'], format="png")
# src.view()
# print(json.dumps(response, indent = 1))

test_data = []

for idx in range(len(dataset)):
    test_data.append({
        key: dataset.iloc[idx][key] for key in predictors
    })

request_body = {
    'task': 'predict',
    'serialized_tree': response['tree_serialized'],
    'test_data': test_data,
}

response = post('http://127.0.0.1:5000/api/decision-trees', 
                headers = {
                    'Content-Type' : 'application/json'
                }, 
                data = json.dumps(request_body)).json()

y_pred = np.array(response['predicted'])
y_true = np.array(dataset[target].tolist())

print('y_pred', y_pred)
print('y_true', y_true)
print('accuracy', np.mean(y_pred == y_true))
