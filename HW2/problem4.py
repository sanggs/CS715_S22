import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
from scipy import linalg as scplinalg
from torch import permute
import time

def construct_1D_laplace(m):
    diagonal = -2. * np.ones(m)
    off_diag = np.ones(m-1)
    return np.diag(diagonal, 0) + np.diag(off_diag, 1) + np.diag(off_diag, -1)

def construct_rhs(m):
    b = np.arange(m)
    b = b.astype(np.float64)
    return b

def get_spectral_radius(mat):
    ei, _ = np.linalg.eig(mat)
    sr = np.abs(ei)
    sr = np.max(sr)
    return sr

def get_singular_value_min_max(mat):
    sg = np.linalg.svd(mat, compute_uv=False)
    return np.max(sg), np.min(sg)

def get_frobenius_norm(mat):
    return np.linalg.norm(mat, 'fro')

def get_2_norm(mat):
    return np.linalg.norm(mat, 2)

def compute_condition_number(mat):
    return np.linalg.cond(mat)

# Get Spectral radius, condition number, sigma_max/min, matrix norm fro, 2
def get_matrix_characteristics(mat):
    sr = get_spectral_radius(mat)
    sg_max, sg_min = get_singular_value_min_max(mat)
    norm_f = get_frobenius_norm(mat)
    norm_2 = get_frobenius_norm(mat)
    cnum = compute_condition_number(mat)
    return sr, sg_max, sg_min, cnum, norm_f, norm_2

# QR Decomposition
def get_QR_decomposition(mat):
    q, r = np.linalg.qr(mat)
    return q, r

def check_qr_correctness(mat, q, r):
    diff = np.linalg.norm(mat - np.matmul(q, r))
    print("Diff in QR = {}".format(diff))

def solve_using_qr(q, r, b):
    start = time.time()
    y = np.matmul(np.transpose(q), b)
    end = time.time() - start
    x = scplinalg.solve_triangular(r, y)
    return x, end

# LU Decomposition
def get_lu_decomposition(mat):
    p, l, u = scp.linalg.lu(mat, permute_l=False)
    return p, l, u

def check_lu_correctness(mat, p, l, u):
    diff = np.linalg.norm(mat - p @ l @ u)
    print("Diff in LU = {}".format(diff))

def solve_using_lu(p, l, u, b):
    b1 = np.matmul(np.transpose(p), b)
    y = scplinalg.solve_triangular(l, b1, lower=True)
    x = scplinalg.solve_triangular(u, y, lower=False)
    return x 

# SVD Decomposition
def get_svd_decomposition(mat):
    u, sigma, vh = np.linalg.svd(mat)
    return u, sigma, vh

def check_svd_correctness(mat, u, sigma, vh):
    diff = np.linalg.norm(mat - u @ np.diag(sigma) @ vh)
    print("Diff in SVD = {}".format(diff))

def solve_using_svd(u, sigma, vh, b):
    c = np.dot(np.transpose(u),b)
    w = np.dot(np.diag(1./sigma),c)
    # Vh x = w <=> x = Vh.H w (where .H stands for hermitian = conjugate transpose)
    x = np.dot(np.transpose(vh),w)
    return x

# Cholesky Decomposition
def get_cholesky_decomposition(mat):
    L = np.linalg.cholesky(mat)
    return L

def check_cholesky_correctness(mat, L):
    diff = np.linalg.norm(mat - np.matmul(L, np.transpose(L)))
    print("Diff in Cholesky = {}".format(diff))

def solve_using_cholesky(L, b):
    y = scplinalg.solve_triangular(L, b, lower=True)
    x = scplinalg.solve_triangular(np.transpose(L), y, lower=False)
    return x

# Solve using numpy.linalg.solve
def solve_using_np_solve(mat, b):
    x = np.linalg.solve(mat, b)
    return x

# Verify norm of |b-Ax|
def verify_solution(mat, x, b):
    diff = np.linalg.norm(np.matmul(mat, x) - b)
    print("|b - Ax| = {}".format(diff))

# 
def log_plot_qr_time():
    times = []
    m_list = []
    for m in range(3, 1001):
        mat = construct_1D_laplace(m)
        b = construct_rhs(m)
        time_qr = time.time()
        q, r = get_QR_decomposition(mat)
        y = np.matmul(np.transpose(q), b)
        time_qr = time.time() - time_qr
        y = y + 1
        times.append(np.log10(time_qr))
        m_list.append(np.log10(m))
    print(np.array(times[1:]) - np.array(times[0:-1])/(np.array(m_list[1:]) - np.array(m_list[0:-1])))
    plt.clf()
    plt.cla()
    plt.plot(m_list, times)
    plt.title("Log log plot of Time vs Matrix dimension")
    plt.xlabel("Matrix dimension log(m)")
    plt.ylabel("Time log(T)")
    plt.show()

def run_experiment(m):
    mat = construct_1D_laplace(m)
    b = construct_rhs(m)

    sr, sg_max, sg_min, cnum, norm_f, norm_2 = get_matrix_characteristics(mat)
    print("Spectral Radius    = {}".format(sr))
    print("Max Singular Value = {}".format(sg_max))
    print("Min Singular Value = {}".format(sg_min))
    print("Condition Number   = {}".format(cnum))
    print("Frobenius Norm     = {}".format(norm_f))
    print("2 Norm             = {}".format(norm_2))

    exit(0)

    # Log time vs Log dimension
    log_plot_qr_time()

    # Solve using QR
    print("Solve using QR")
    time_qr = time.time()
    q, r = get_QR_decomposition(mat)
    time_qr = time.time() - time_qr
    check_qr_correctness(mat, q, r)
    x_qr, time_qr1 = solve_using_qr(q, r, b)
    time_qr += time_qr1
    print("Time taken for QR decomposition and y=Q*b = {} s".format(time_qr))
    verify_solution(mat, x_qr, b)

    # Solve using LU
    print("Solve using LU")
    time_lu = time.time()
    p, l, u = get_lu_decomposition(mat)
    time_lu = time.time() - time_lu
    check_lu_correctness(mat, p, l, u)
    time_lu_solve = time.time()
    x_lu = solve_using_lu(p, l, u, b)
    time_lu += time.time() - time_lu_solve
    print("Time taken for LU decomposition and solve = {} s".format(time_lu))
    verify_solution(mat, x_lu, b)

    # Solve using SVD
    print("Solve using SVDs")
    time_svd = time.time()
    u, sigma, vh = get_svd_decomposition(mat)
    time_svd = time.time() - time_svd
    check_svd_correctness(mat, u, sigma, vh)
    time_svd_solve = time.time()
    x_svd = solve_using_svd(u, sigma, vh, b)
    time_svd += time.time() - time_svd_solve
    print("Time taken for SVD decomposition and solve = {} s".format(time_svd))
    verify_solution(mat, x_svd, b)

    # Solve using Cholesky
    print("Solve using Cholesky")
    time_cholesky = time.time()
    L = get_cholesky_decomposition(-1.0 * mat)
    time_cholesky = time.time() - time_cholesky
    check_cholesky_correctness(-1.0 * mat, L)
    time_cholesky_solve = time.time()
    x_cholesky = solve_using_cholesky(L, -1.0 * b)
    time_cholesky += time.time() - time_cholesky_solve
    print("Time taken for Cholesky factorization and Solve = {} s".format(time_cholesky))
    verify_solution(mat, x_cholesky, b)
    
    # Solve using np.linalg.solve
    print("Solve using numpy.linalg.solve")
    time_np = time.time()
    x = solve_using_np_solve(mat, b)
    time_np = time.time() - time_np
    print("Time taken for Numpy factorization and Solve = {} s".format(time_np))
    verify_solution(mat, x, b)

if __name__ == '__main__':
    m = 4000
    run_experiment(m)
