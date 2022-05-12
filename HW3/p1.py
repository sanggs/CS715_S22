import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, diags
from scipy.sparse.linalg import spsolve
import time
import matplotlib.pyplot as plt

def construct_1D_laplace(m):
    h = 1./m
    diagonal = 2. * np.ones(m-1)
    off_diag = -1. * np.ones(m-2)
    return (1./(h*h)) * (np.diag(diagonal, 0) + np.diag(off_diag, 1) + np.diag(off_diag, -1))

def sample_points_1D(m):
    h = 1./m
    x = np.linspace(h, 1-h, num=m-1)
    return x, np.zeros_like(x) 

def get_rhs(x, const):
    return np.sin(const*x*x)

def compute_residual_norm(A, u, f):
    return np.linalg.norm(f - np.dot(A, u), np.inf)

def direct_solve(A, f):
    return np.linalg.solve(A, f)

def build_sparse_A(m):
    h = 1./m
    A_sparse = csc_matrix(diags([-1, 2, -1], [-1, 0, 1], shape=(m-1, m-1)))/(h**2)
    return A_sparse

def sparse_solve(A_sparse, f):
    return spsolve(A_sparse,f)

def verify_eigen_values(A):
    n = A.shape[0]
    h = 1./(n+1)
    v_num, _ = np.linalg.eig(A)
    v_num = np.sort(v_num)
    k = np.arange(1, n+1)
    v_an = (4/(h*h)) * np.sin((k*np.pi)/(2*n+2)) * np.sin((k*np.pi)/(2*n+2))
    print("Inf Norm of diff in eigenvalues for m = {} is {}".format(n+1, np.linalg.norm(v_an-v_num, np.inf)))

m = 8
A = construct_1D_laplace(m)
verify_eigen_values(A)

ms = [1000, 2000, 4000]
rn = []
for m in ms:
    const = 100
    A = construct_1D_laplace(m)
    x, u = sample_points_1D(m)
    f = get_rhs(x, const)
    u_ds = direct_solve(A, f)
    rn.append(np.linalg.norm(u_ds, np.inf))
    print("\t h = {}, res = {}".format( 1./m, compute_residual_norm(A, u_ds, f)))

print("Convergence rate = {}".format((rn[0]-rn[1])/(rn[1]-rn[2])))

m = 10000
const = 1000
A = construct_1D_laplace(m)
x, u = sample_points_1D(m)
f = get_rhs(x, const)
start = time.time()
u_ds = direct_solve(A, f)
end = time.time() - start
print("Time taken for solve = {}".format(end))

plt.plot(x, u_ds, label='solution u')
plt.title("Solution for m = {}".format(m))
plt.ylabel("u (solution)")
plt.xlabel("x")
plt.legend()
plt.show()

m = 100000
const = 1000

A_sparse = build_sparse_A(m)
x, u = sample_points_1D(m)
f = get_rhs(x, const)

start = time.time()
u_ss = sparse_solve(A_sparse, f)
end = time.time() - start
print("Residual norm for sparse solve = {} for m = {}".format(np.linalg.norm(f-A_sparse.dot(u_ss), np.inf), m))
print("Time taken for sparse solve = {} for m = {}".format(end, m))
