
import numpy as np
def matrix_power(matrix, m):
    return np.linalg.matrix_power(matrix, m)
a = int(input(" Enter  matrix dimension:"))
matrix = np.random.randint(0,10,(a,a))
m = int(input("Enter the power:"))
result = matrix_power(matrix, m)
print(result)