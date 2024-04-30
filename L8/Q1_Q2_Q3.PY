import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

# Function to calculate entropy
def entropy(labels):
    counts = Counter(labels)
    total = len(labels)
    ent = 0
    for count in counts.values():
        prob = count / total
        ent -= prob * np.log2(prob)
    return ent

# Function to calculate information gain
def information_gain(data, feature, target):
    total_entropy = entropy(data[target])
    
    feature_entropy = 0
    values = data[feature].unique()
    for value in values:
        subset = data[data[feature] == value]
        prob = len(subset) / len(data)
        feature_entropy += prob * entropy(subset[target])
    
    gain = total_entropy - feature_entropy
    return gain

# Function to find the best root node based on information gain
def find_best_root(data, features, target):
    gains = {}
    for feature in features:
        gains[feature] = information_gain(data, feature, target)
    return max(gains, key=gains.get)

# Function to bin continuous features into categorical
def bin_continuous(data, feature, num_bins=10, bin_type='equal_width'):
    if bin_type == 'equal_width':
        bins = np.linspace(data[feature].min(), data[feature].max(), num_bins + 1)
    elif bin_type == 'equal_frequency':
        bins = np.quantile(data[feature], np.linspace(0, 1, num_bins + 1))
    else:
        raise ValueError("Invalid bin_type. Use 'equal_width' or 'equal_frequency'.")
    
    binned_feature = pd.cut(data[feature], bins, include_lowest=True)
    return binned_feature

# Decision Tree class using information gain
class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def build_tree(self, data, features, target, depth=0):
        # Stopping conditions
        if len(set(data[target])) == 1:
            return data[target].iloc[0]
        if len(features) == 0 or depth >= self.max_depth or data.shape[0] < self.min_samples_split:
            return data[target].mode()[0]

        best_feature = find_best_root(data, features, target)
        
        tree = {best_feature: {}}
        
        for value in data[best_feature].unique():
            subset = data[data[best_feature] == value]
            remaining_features = [f for f in features if f != best_feature]
            tree[best_feature][value] = self.build_tree(subset, remaining_features, target, depth + 1)
        
        return tree

    def fit(self, data, features, target):
        self.tree = self.build_tree(data, features, target)
        return self

    def predict(self, instance, tree=None):
        if tree is None:
            tree = self.tree
        
        for feature, branches in tree.items():
            if feature in instance:
                feature_value = instance[feature]
                if feature_value in branches:
                    subtree = branches[feature_value]
                    if isinstance(subtree, dict):
                        return self.predict(instance, subtree)
                    else:
                        return subtree
        return None

    def predict_batch(self, data):
        predictions = data.apply(lambda row: self.predict(row), axis=1)
        return predictions

# Example use case
data = pd.read_excel("customer.xlsx")

# Binning a continuous feature
data["binned_feature"] = bin_continuous(data, "your_continuous_feature", num_bins=5, bin_type='equal_width')

# Define features and target
features = [col for col in data.columns if col != "target"]  # Adjust based on your data
target = "target"

# Create and fit the decision tree
tree = DecisionTree(max_depth=5, min_samples_split=3)
tree.fit(data, features, target)

# Make predictions
predictions = tree.predict_batch(data[features])

# Display predictions
print("Predictions:\n", predictions)
