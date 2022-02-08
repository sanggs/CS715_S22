import numpy as np
import matplotlib.pyplot as plt

def classical_gram_schmidt(A):
    m = A.shape[0]
    n = A.shape[1]
    assert(m >= n)
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        v_j = A[:, j]
        for i in range(j):
            R[i][j] = np.dot(Q[:, i], A[:, j])
            v_j = v_j - R[i][j] * Q[:, i]
        R[j][j] = np.linalg.norm(v_j, 2)
        Q[:, j] = v_j/R[j][j]
    return Q, R

def modified_gram_schmidt(A):
    m = A.shape[0]
    n = A.shape[1]
    assert(m >= n)
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    V = np.zeros((m, n))
    for i in range(n):
        V[:, i] = np.copy(A[:, i])
    for i in range(n):
        R[i][i] = np.linalg.norm(V[:, i], 2)
        Q[:, i] = V[:, i]/R[i][i]
        for j in range(i+1, n):
            R[i][j] = np.dot(Q[:, i], V[:, j])
            V[:, j] = V[:, j] - R[i][j] * Q[:, i]
    return Q, R

def verify_classical_QR(m, n):
    A = np.random.rand(m, n)
    Q, R = classical_gram_schmidt(A)
    print( "Reconstruction error for A with Classical Gram Schmidt = {}".format( np.linalg.norm(A - np.matmul(Q, R))))

def verify_modified_QR(m, n):
    A = np.random.rand(m, n)
    Q, R = modified_gram_schmidt(A)
    print( "Reconstruction error for A with Modified Gram Schmidt = {}".format( np.linalg.norm(A - np.matmul(Q, R))))

def run_experiment2():
    m = 80
    sigma = np.linspace(-1, -80, m)
    sigma = 2**sigma
    
    A = np.random.rand(m, m)
    U, _ = np.linalg.qr(A)
    A = np.random.rand(m, m)
    V, _ = np.linalg.qr(A)

    A = np.matmul(np.matmul(U, np.diag(sigma)), np.transpose(V))

    _, Rc = classical_gram_schmidt(A)
    _, Rm = modified_gram_schmidt(A)

    plt.plot(np.diagonal(Rc), 'o', label='Classical Gram Schmidt')
    plt.plot(np.diagonal(Rm), '.',label='Modified Gram Schmidt')
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Number of singular values')
    plt.ylabel('Log of singular values')
    plt.title('Singular values from Classical, Modified Orthogonalization')
    plt.savefig('CGS_vs_MGS.png')
    return

if __name__ == '__main__':
    verify_classical_QR(100, 80)
    verify_modified_QR(100, 80)
    run_experiment2()
