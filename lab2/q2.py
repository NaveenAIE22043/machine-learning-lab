import numpy as np

def manhattan_distance(point1, point2):
    return np.sum(np.abs(point1 - point2))

def knn_classifier(X_train, y_train, X_test, k):
    predictions = []
    for test_point in X_test:
        distances = [(manhattan_distance(train_point, test_point), label) for train_point, label in zip(X_train, y_train)]
        distances.sort()  
        neighbors = distances[:k] 
        
        class_votes = {}
        for _, label in neighbors:
            if label in class_votes:
                class_votes[label] += 1
            else:
                class_votes[label] = 1
        
        predicted_label = max(class_votes, key=class_votes.get)
        predictions.append(predicted_label)
    
    return predictions

X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y_train = np.array(['A', 'A', 'B', 'B'])
X_test = np.array([[1.5, 2.5], [3.5, 4.5]])
k = 2

predictions = knn_classifier(X_train, y_train, X_test, k)
print("Predictions:", predictions)
