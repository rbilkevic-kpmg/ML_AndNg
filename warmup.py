import numpy as np

# initialize matrix A
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [9, 8, 9]])

# initialize a vector
v = np.array([1, 2, 3]).reshape(3, 1)

# get the size of A
dim_A = A.shape

# get the size of v
dim_v = v.shape

print(dim_A, dim_v)

# initialize matrix A and B
A = np.array([
    [1, 2, 3],
    [4, 5, 6]])
B = np.array([
    [1, 3, 4],
    [1, 1, 1]])

# initialize constant s
s = 2

# See how element-wise addition works
add_AB = A + B

# See how element-wise subtraction works
sub_AB = A - B

# See how scalar multiplication works
mult_As = A * s

# Divide A by s
div_As = A / s

# What happens if we have a Matrix + scalar?
add_As = A + s

# Initialize matrix A
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]])

# Initialize vector v
v = np.array([1, 1, 1]).reshape(3, 1)

# Multiply A * v
Av = np.dot(A, v)

# Initialize a 3 by 2 matrix
A = np.array([
    [1, 2],
    [3, 4],
    [5, 6]])

# Initialize a 2 by 1 matrix
B = np.array([1, 2]).reshape(2, 1)

# We expect a resulting matrix of (3 by 2)*(2 by 1) = (3 by 1)
mult_AB = np.dot(A, B)

# Initialize random matrices A and B
A = np.array([
    [1, 2],
    [4, 5]])
B = np.array([
    [1, 1],
    [0, 2]])

# Initialize a 2 by 2 identity matrix
I = np.identity(2)

# What happens when we multiply I*A ?
IA = np.dot(I, A)

# How about A*I ?
AI = np.dot(A, I)

print("AI equal to IA?", np.array_equal(AI, IA))

# Compute A*B
AB = np.dot(A, B)

# Is it equal to B*A?
BA = np.dot(B, A)

print("AB equal to BA?", np.array_equal(AB, BA))

# Initialize matrix A
A = np.array([
    [1, 2, 0],
    [0, 5, 6],
    [7, 0, 9]
])

# Transpose A
A_trans = A.T

# Take the inverse of A
A_inv = np.linalg.inv(A)
print(A)
print(A_inv)

# What is A^(-1)*A?
A_invA = np.dot(np.linalg.inv(A), A)
print(A_invA)