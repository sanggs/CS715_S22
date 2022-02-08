import numpy as np

A = np.array([[0, 1], [1, 0]])
print(A)
print(np.linalg.svd(A))

lambd, x = np.linalg.eig(A)
x_inv = np.zeros_like(x)
x_inv[0, 0] = x[1, 1]
x_inv[0, 1] = -x[0, 1]
x_inv[1, 0] = -x[1, 0]
x_inv[1, 1] = x[0, 0]
print(x_inv)

b = np.matmul(x, np.diag(lambd))
b = np.matmul(b, x_inv)

print(b)
