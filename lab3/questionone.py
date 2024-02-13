import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load data from Excel file
file_path = "C:\\Users\\navee\\OneDrive\\Desktop\\mlcode\\lab3\\Lab Session1 Data.xlsx"
df = pd.read_excel(file_path, sheet_name="Purchase data")
# A2: Use the Pseudo-Inverse to calculate the model vector X
# Segregate data into matrices A and C
A = df.iloc[:, 1:4].values  # Assuming the relevant columns are 2nd, 3rd, and 4th
C = df.iloc[:, 4].values  # Assuming the 5th column is the payment column

# Using Pseudo-Inverse to calculate the model vector X
pseudo_inverse_A = np.linalg.pinv(A)
model_vector_X = np.dot(pseudo_inverse_A, C)

# Display model vector X
print("Model Vector X:")
print(model_vector_X)

# A3: Mark customers as RICH or POOR based on payments
df['Customer_Class'] = np.where(df['Payment (Rs)'] > 200, 'RICH', 'POOR')

# Prepare features (A) and target (Customer_Class) for classification
X = df.iloc[:, 1:4].values  # Features
y = df['Customer_Class'].values  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression classifier
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

# Display results
print("\nLogistic Regression Classifier Results:")
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report_str)
