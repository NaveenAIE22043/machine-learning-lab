import math

def euclidean_distance(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must be of the same dimension")
    
    sum_squared_diff = sum((x - y)**2 for x, y in zip(vector1, vector2))
    return math.sqrt(sum_squared_diff)

def manhattan_distance(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must be of the same dimension")
    
    return sum(abs(x - y) for x, y in zip(vector1, vector2))

vector1 = [1, 2, 3]
vector2 = [4, 5, 6]

euclidean_dist = euclidean_distance(vector1, vector2)
manhattan_dist = manhattan_distance(vector1, vector2)

print("Euclidean Distance:", euclidean_dist)
print("Manhattan Distance:", manhattan_dist)
