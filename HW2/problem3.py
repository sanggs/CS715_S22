from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from torch import norm, threshold

def generate_random_matrix(m):
    mat = np.random.rand(m, m)
    return mat

def make_upper_traingular(mat):
    return np.triu(mat)

def normalize_matrix(mat, m):
    mat = mat / np.sqrt(m)
    return mat

def get_eigen_values(mat):
    v, _ = np.linalg.eig(mat)
    return v

def get_spectral_radius(eig_vals):
    return np.max(np.absolute(eig_vals))

def get_norm_2(mat):
    return np.linalg.norm(mat, 2)

def get_smallest_sigma(mat):
    sg_min = np.min(np.linalg.svd(mat, compute_uv=False))
    return sg_min

def plot_eig_vals(eig_values, m_sizes, markers):
    plt.clf()
    plt.cla()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    num_plots = len(m_sizes)
    for i in range(num_plots):
        ei = eig_values[i]
        ei_real = np.real(ei)
        ei_complex = np.imag(ei)
        m = m_sizes[i]
        y = np.arange(m)
        ax1.plot(ei_real, y, markers[i], label='Matrix dim = '+str(m))
        ax2.plot(ei_complex, y, markers[i], label='Matrix dim = '+str(m))
    ax1.legend()
    ax1.set_xlabel('Real part of eigenvalues')
    ax1.set_ylabel('Number of eigenvalues')
    ax1.set_title('Real part of eigenvalues')
    ax2.legend()
    ax2.set_xlabel('Imag part of eigenvalues')
    ax2.set_ylabel('Number of eigenvalues')
    ax2.set_title('Imaginary part of eigenvalues')
    # plt.legend()
    plt.show()

def plot_spectral_radius(sr, nm, m):
    plt.clf()
    plt.cla()
    plt.plot(m, sr, label='Spectral radius')
    plt.plot(m, nm, label='2-Norm')
    plt.title("Spectral Radius and 2-Norm of matrix")
    plt.xlabel("Matrix dimension")
    plt.ylabel("Spectral Radius/2-Norm")
    plt.legend()
    plt.show()

def plot_min_singular_distribution(m_list, threshold, markers):
    plt.clf()
    plt.cla()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    sg_min_tail = []
    j = 0
    for m in m_list:
        sg_min_list = []
        count = 0
        for i in range(80):
            mat = generate_random_matrix(m)
            sg_min = get_smallest_sigma(mat)
            sg_min_list.append(sg_min)
            if (sg_min < threshold):
                count += 1
        sg_min_tail.append(count)
        ax1.plot(sg_min_list, markers[j], label='Matrix dim = {}'.format(m))
        j += 1
    ax1.set_title('Singular value distribution')
    ax1.legend()
    ax1.set_xlabel("Matrix dimension")
    ax1.set_ylabel("Value of minimum singular value")
    ax2.plot(m_list, sg_min_tail, 'o-')
    ax2.set_title('Number of singular values less than {}'.format(threshold))
    ax2.set_xlabel("Matrix dimensions")
    plt.show()
    return

def run_experiment(m_list):
    eig_values = []
    spectral_radius = []
    norm_2 = []
    threshold = 1./np.power(2, 3)
    for m in m_list:
        mat = generate_random_matrix(m)
        mat = make_upper_traingular(mat)
        # mat = normalize_matrix(mat, m)
        ei = get_eigen_values(mat)
        eig_values.append(ei)
        sr = get_spectral_radius(ei)
        spectral_radius.append(sr)
        n2 = get_norm_2(mat)
        norm_2.append(n2)
    markers = ['>', 'o', '*', 's', "p", "+", "v", "^", "x"]
    plot_eig_vals(eig_values, m_list, markers)
    plot_spectral_radius(spectral_radius, norm_2, m_list)
    # m_copy = [m_list[4]]
    plot_min_singular_distribution(m_list, threshold, markers)

if __name__ == '__main__':
    m_list = [8, 16, 32, 64, 128, 256, 512, 1024]
    run_experiment(m_list)


