import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn.metrics import accuracy_score

# Custom perceptron class with sigmoid activation
class CustomPerceptron(BaseEstimator):
    def __init__(self, n_features, lr=0.01, max_epochs=1000):
        self.n_features = n_features
        self.lr = lr
        self.max_epochs = max_epochs
        self.weights = np.random.rand(n_features)
        self.bias = np.random.rand()

    # Sigmoid activation function
    def activation(self, z):
        return 1 / (1 + np.exp(-z))

    # Predict output based on input
    def compute(self, x):
        weighted_sum = np.dot(x, self.weights) + self.bias
        return self.activation(weighted_sum)

    # Train perceptron with given data
    def fit(self, X, y):
        for _ in range(self.max_epochs):
            for i in range(len(X)):
                input_data = X[i]
                expected_output = y[i]
                prediction = self.compute(input_data)
                error = expected_output - prediction
                self.weights += self.lr * error * input_data
                self.bias += self.lr * error
        return self

    # Make predictions
    def predict(self, X):
        outputs = self.compute(X)
        return np.where(outputs >= 0.5, 1, 0)

    # Return parameters for the model
    def get_params(self, deep=True):
        return {
            'n_features': self.n_features,
            'lr': self.lr,
            'max_epochs': self.max_epochs
        }

# Load the data from an Excel file
data = pd.read_excel('customerdata.xlsx')

# Encode categorical data into numeric values
label_map = {'Yes': 1, 'No': 0}
data['High Value Tx'] = data['High Value Tx'].map(label_map)

# Feature extraction and normalization
features = data.drop(columns=['Customer', 'High Value Tx']).values
target = data['High Value Tx'].values

# Normalize features
features_normalized = features / features.max(axis=0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    features_normalized, target, test_size=0.2, random_state=42
)

# Hyperparameter grid with different distribution
param_grid = {
    'lr': uniform(0.001, 0.1),
    'max_epochs': randint(100, 2000)
}

# Create RandomizedSearchCV with modified hyperparameters
custom_perceptron = CustomPerceptron(n_features=X_train.shape[1])
random_cv = RandomizedSearchCV(
    estimator=custom_perceptron,
    param_distributions=param_grid,
    n_iter=10,
    cv=5,
    scoring='accuracy'
)

# Fit RandomizedSearchCV and get best parameters
random_cv.fit(X_train, y_train)
best_parameters = random_cv.best_params_
print("Best Hyperparameters:", best_parameters)

# Train the best perceptron and test on the test set
best_perceptron = CustomPerceptron(n_features=X_train.shape[1], **best_parameters)
best_perceptron.fit(X_train, y_train)

y_predictions = best_perceptron.predict(X_test)
final_accuracy = accuracy_score(y_test, y_predictions)
print(f"Test Accuracy with Best Parameters: {final_accuracy:.2f}")
