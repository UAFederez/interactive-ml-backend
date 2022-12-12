from flask import Flask, request
from flask_restful import Api, Resource
import json
import graphviz
import pandas as pd
import numpy as np

def binary_cross_entropy_series(p):
    a = pd.Series(p)
    a = a.apply(lambda p : -(p * np.log(p) + (1-p) * np.log(1-p)) if (p > 0.0 and p < 1.0) else 0.0)
    return a

class DecisionNode:
    def __init__(self):
        self.left_child  = None
        self.right_child = None
    
    def find_weighted_entropy(self, data, predictor, target):
        raise NotImplementedError
    
    def find_optimal_split(self, data, predictor, target):
        raise NotImplementedError

class LeafNode(DecisionNode):
    def __init__(self, class_label):
        super().__init__()
        self.class_label = class_label
        
class BinaryDecision(DecisionNode):
    def __init__(self):
        super().__init__()
        
    def find_weighted_entropy(self, data, predictor, target):
        data      = data[[predictor, target]].copy()
        
        # Fraction of target values grouped by predictor values
        pred_grouped  = data.groupby(predictor)[target].value_counts(normalize = True)
        pred_fraction = pred_grouped[(slice(None), 1)]
        pred_entropy  = binary_cross_entropy_series(pred_fraction)
        
        pred_split    = data.groupby(predictor)[target].count()
        pred_split    = pred_split / pred_split.sum()
        
        self.weighted_entropy = np.dot(pred_split, pred_entropy)
        return self.weighted_entropy
        
class NumericDecision(DecisionNode):
    def __init__(self):
        super().__init__()
    
    def find_weighted_entropy(self, data, predictor, target):
        data = data[[predictor, target]].copy()
        data = data.sort_values(by = [predictor], axis = 0, ascending = True).reset_index()

        value_counts = data[target].value_counts(normalize = True)

        if 1 in value_counts.index:
            root_positive_frac = value_counts[1]
        else:
            root_positive_frac = 1.0 - value_counts[0]

        data['pct_y_leq']  = (data[target].astype('float64').cumsum() / data.shape[0])
        data['pct_y_g']    = (root_positive_frac - data['pct_y_leq']).astype('float64')
        data['bce_y_leq']  = binary_cross_entropy_series(data['pct_y_leq'])
        data['bce_y_g']    = binary_cross_entropy_series(data['pct_y_g'])
        data['frac_leq']   = pd.Series(np.arange(1, data.shape[0] + 1)) / data.shape[0]
        data['frac_g']     = 1 - data['frac_leq']
        data['weighted_e'] = data['frac_leq'] * data['bce_y_leq'] + data['frac_g'] * data['bce_y_g']
        
        return data
    
    def find_optimal_split(self, data, predictor, target):
        entropy_data   = self.find_weighted_entropy(data, predictor, target)
        self.entropy   = entropy_data.loc[entropy_data['weighted_e'].idxmin(), 'weighted_e']
        self.split_val = entropy_data.loc[entropy_data['weighted_e'].idxmin(), predictor]
        
        return self.split_val, self.entropy

def plot_decision_tree(root, dot, depth = 0):
    global curr_id
    if not root:
        return None
    
    if isinstance(root, NumericDecision):
        node_label = '{}\n<={:.2f}'.format(root.predictor, root.split_val)
    else:
        node_label = root.class_label if isinstance(root, LeafNode) else root.predictor
    
    node_id = '\"{}\"'.format(hash(root))
    dot.node(node_id, '{}'.format(node_label))
    
    left  = plot_decision_tree(root.left_child, dot, depth + 1)
    right = plot_decision_tree(root.right_child, dot, depth + 1)

    if left:  dot.edge(node_id, left , label = 'false' if depth == 0 else '')
    if right: dot.edge(node_id, right, label = 'true' if depth == 0 else '')
    
    return node_id

class DecisionTrees(Resource):
    def build_tree(self, data, predictors, target, curr_depth = 0, max_depth = 12, info_gain_min = 1e-3, entropy_min = 1e-3, min_samples_leaf = 1):
        categorical = [ pred for pred in predictors if pd.api.types.is_categorical_dtype(data[pred]) ]
        numeric     = [ pred for pred in predictors if pd.api.types.is_numeric_dtype(data[pred]) ]
        
        value_counts   = data[target].value_counts(normalize = True)

        if 1 in value_counts.index:
            parent_entropy = binary_cross_entropy_series(value_counts[1])[0]
        else:
            parent_entropy = 1.0 - binary_cross_entropy_series(value_counts[0])[0]

        entropy_vals   = []
        
        # Calculate entropy for each categorical variable
        test_node = BinaryDecision()
        for category in categorical:
            entropy   = test_node.find_weighted_entropy(data, category, target)
            info_gain = parent_entropy - entropy
            entropy_vals.append({
                'predictor': category,
                'entropy'  : entropy,
                'info_gain': info_gain
            })
        
        # Calculate entropy for each numeric variable
        test_node = NumericDecision()
        for num in numeric:
            split_val, entropy = test_node.find_optimal_split(data, num, target)
            info_gain = parent_entropy - entropy
            entropy_vals.append({
                'predictor': num,
                'entropy'  : entropy,
                'split_val': split_val,
                'info_gain': info_gain
            })
        
        # Choose the optimal split variable with the highest information gain
        optimal_split = max(entropy_vals, key = lambda x : x['info_gain'])
        
        left_split  = None
        right_split = None
        new_node    = None
        if optimal_split['predictor'] in categorical:
            new_node = BinaryDecision()
            new_node.predictor = optimal_split['predictor']
            new_node.entropy   = optimal_split['entropy']
            new_node.info_gain = optimal_split['info_gain']
            
            left_split  = data[data[optimal_split['predictor']] == 0]
            right_split = data[data[optimal_split['predictor']] != 0]
        else:
            new_node = NumericDecision()
            new_node.predictor = optimal_split['predictor']
            new_node.entropy   = optimal_split['entropy']
            new_node.info_gain = optimal_split['info_gain']
            new_node.split_val = optimal_split['split_val']
            
            left_split  = data[data[optimal_split['predictor']] <= new_node.split_val]
            right_split = data[data[optimal_split['predictor']]  > new_node.split_val]

        allowed_to_split    = (curr_depth + 1 < max_depth) and (optimal_split['info_gain'] > info_gain_min)
        has_remaining_data  = (len(left_split) >= min_samples_leaf) and (len(right_split) >= min_samples_leaf)

        will_continue_split = allowed_to_split and has_remaining_data
        
        # Recursively build the decision tree
        if will_continue_split:
            new_node.left_child  = self.build_tree(left_split , predictors, target, curr_depth + 1, max_depth)
            new_node.right_child = self.build_tree(right_split, predictors, target, curr_depth + 1, max_depth)
        else:
            # Check if one of either split is empty but the other one isn't. In such case then
            # there is only one leaf node for the non-empty split
            if len(right_split) == 0 and len(left_split) > 0:
                return LeafNode(left_split[target].value_counts().idxmax())
            elif len(left_split) == 0 and len(right_split) > 0:
                return LeafNode(right_split[target].value_counts().idxmax())

            left_class  = left_split[target].value_counts().idxmax()
            right_class = right_split[target].value_counts().idxmax()
            
            new_node.left_child  = LeafNode(left_class)
            new_node.right_child = LeafNode(right_class)
        
        return new_node

    def prune_tree(self, tree):
        if not tree:
            return None
        
        tree.left_child  = self.prune_tree(tree.left_child)
        tree.right_child = self.prune_tree(tree.right_child)

        # Remove leaves for which the majority class is the same
        if isinstance(tree.left_child, LeafNode) and isinstance(tree.right_child, LeafNode):
            if tree.left_child.class_label == tree.right_child.class_label:
                return LeafNode(tree.left_child.class_label)
        
        return tree

    def deserialize_tree(self, root):
        if not root:
            return None
        
        node = None

        if root['type'] == 'binary_decision':
            node = BinaryDecisionNode()
            node.predictor = root['predictor']
        elif root['type'] == 'numeric_decision':
            node = NumericDecision()
            node.predictor = root['predictor']
            node.split_val = root['split_value']
        elif root['type'] == 'leaf_decision_node':
            node = LeafNode(root['label'])

        if 'left_child' in root:
            node.left_child = self.deserialize_tree(root['left_child'])
        if 'right_child' in root:
            node.right_child = self.deserialize_tree(root['right_child'])

        return node

    def serialize_tree(self, root):
        if not root:
            return {}

        node_data = {}

        if isinstance(root, BinaryDecision):
            node_data['type'] = 'binary_decision'
            node_data['predictor'] = root.predictor
        elif isinstance(root, NumericDecision):
            node_data['type'] = 'numeric_decision'
            node_data['predictor'] = root.predictor
            node_data['split_value'] = float(root.split_val)
        elif isinstance(root, DecisionNode):
            node_data['type']  = 'leaf_decision_node'
            node_data['label'] = int(root.class_label)

        if root.left_child is not None:
            node_data['left_child'] = self.serialize_tree(root.left_child)

        if root.right_child is not None:
            node_data['right_child'] = self.serialize_tree(root.right_child)

        return node_data

    def predict(self, record, root):
        if isinstance(root, LeafNode):
            return root.class_label

        if isinstance(root, NumericDecision):
            if record[root.predictor] <= root.split_val:
                return self.predict(record, root.left_child)
            else:
                return self.predict(record, root.right_child)
        elif isinstance(root, BinaryDecision):
            if record[root.predictor] == 0.0:
                return self.predict(record, root.left_child)
            else:
                return self.predict(record, root.right_child)
        
    def post(self):
        data = request.json
        task = data['task']

        if task == 'predict':
            tree_data = data['serialized_tree']
            tree_root = self.deserialize_tree(tree_data)

            test_data = data['test_data']
            predicted_labels = []

            for record in test_data:
                predicted_labels.append(self.predict(record, tree_root))

            return {
                'predicted': predicted_labels,
            }


        elif task == 'train':
            train_features = { feature['name']: feature['values']  for feature in data['train_features'] }
            train_label_name, train_labels = data['train_labels']['name'], data['train_labels']['values']

            min_samples_leaf = data['min_samples_leaf']
            max_depth = data['max_depth']

            dataset = pd.DataFrame.from_dict(train_features)
            dataset[train_label_name] = train_labels

            predictors = [ feature['name'] for feature in data['train_features'] ]
            target     = train_label_name

            tree = self.build_tree(dataset, predictors, target, max_depth = max_depth, min_samples_leaf = min_samples_leaf)

            if 'will_prune' in data and data['will_prune'] == True:
                tree = self.prune_tree(tree)

            dot = graphviz.Digraph(comment = 'Decision Tree Plot')
            dot.attr('node', shape = 'box', nodesep = '0.1')
            plot_decision_tree(tree, dot)

            serialized_tree = self.serialize_tree(tree)

            return {
                'graphviz_output': dot.source,
                'tree_serialized': serialized_tree
            }
        else:
            return { 'error': 'invalid task supplied' }
