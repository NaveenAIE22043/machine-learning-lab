import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Function to evaluate a classifier with cross-validation
def assess_classifier(model, X_data, y_data):
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    scores = {metric: cross_val_score(model, X_data, y_data, cv=5, scoring=metric).mean() for metric in metrics}
    return scores

# Load data from an Excel file into a DataFrame
data = pd.read_excel('customerdata.xlsx')

# Encode target labels as numeric
label_map = {'Yes': 1, 'No': 0}
data['High Value Tx'] = data['High Value Tx'].map(label_map)

# Separate features from the target
feature_matrix = data.drop(columns=['Customer', 'High Value Tx']).values.astype(float)
target_vector = data['High Value Tx'].values

# Normalize the feature matrix
normalized_features = feature_matrix / feature_matrix.max(axis=0)

# Define a dictionary of classifiers to compare
ml_classifiers = {
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Naïve Bayes": GaussianNB(),
    "CatBoost": CatBoostClassifier(logging_level='Silent'),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Store results in a dictionary
classifier_results = {}

# Evaluate each classifier with cross-validation
for clf_name, clf_instance in ml_classifiers.items():
    results = assess_classifier(clf_instance, normalized_features, target_vector)
    classifier_results[clf_name] = {
        "Accuracy": results["accuracy"],
        "Precision": results["precision"],
        "Recall": results["recall"],
        "F1 Score": results["f1"]
    }

# Convert results dictionary to DataFrame for tabular display
results_df = pd.DataFrame(classifier_results).T  # Transpose to get classifiers as rows
print("Classifier Performance:\n", results_df)
