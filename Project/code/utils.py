import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def save_scalar_variable(var, filepath="var.npy"):
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    np.save(var.numpy(), filepath)

def norm(var):
    # return torch.sqrt(torch.max(torch.sum(var*var, dim = 0)))
    return torch.norm(var, float('inf'))

def visualize_slice(var_3d, slice_dim=0, slice_axis="x"):
    assert(len(var_3d.shape)==3)
    v = None
    if (slice_axis == "x"):
        v = var_3d[slice_dim, :, :].cpu().numpy()
        i = np.linspace(0, 1, var_3d.shape[1])
        j = np.linspace(0, 1, var_3d.shape[2])
        i, j = np.meshgrid(i, j, indexing='ij')
        
    elif (slice_axis == "y"):
        v = var_3d[:, slice_dim, :].cpu().numpy()
        i = np.linspace(0, 1, var_3d.shape[0])
        j = np.linspace(0, 1, var_3d.shape[2])
        i, j = np.meshgrid(i, j, indexing='ij')
    elif (slice_axis == "z"):
        v = var_3d[:, :, slice_dim].cpu().numpy()
        i = np.linspace(0, 1, var_3d.shape[0])
        j = np.linspace(0, 1, var_3d.shape[1])
        i, j = np.meshgrid(i, j, indexing='ij')
    assert(v is not None)
    ax = plt.axes(projection='3d')
    ax.plot_surface(i, j, v, rstride=1, cstride=1, cmap='viridis')
    ax.set_title('pressure slice_dim = {} slice_axis={}'.format(slice_dim, slice_axis))
    ax.set_zlim(-1,1)
    plt.show()

def show_convergence_rate(res_list):
    l = len(res_list)
    res_list = np.array(res_list)
    ratio = res_list[1:l] / res_list[0:l-1]
    # ax = plt.axes(projection='2d')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(ratio, 'o-')
    ax.set_xlabel ('Iterations')
    ax.set_ylabel('Ratio of residual |r_(k+1)|/|r_k|')
    ax.set_title('Rate of Convergence plot')
    plt.savefig("convergence_rate.png")

def show_3D_plot(var, frame=0):
    print("Writing output number {}".format(frame))
    if not os.path.exists("plots_html"):
        os.makedirs("plots_html")

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')

    w = var.shape[0]
    h = var.shape[1]
    d = var.shape[2]
    r_p = np.zeros([w, h, d, 3])
    r_p[:, :, :, 0] = var[:, :, :]

    x = np.linspace(0, w, w)
    y = np.linspace(0, h, h)
    z = np.linspace(0, d, d)
    
    x, y, z = np.meshgrid(x,y,z, indexing='ij')

    x = x.reshape(-1)
    y = y.reshape(-1)
    z = z.reshape(-1)
    r_p = r_p.reshape(-1, 3)

    # ax.scatter(x, y, z, s = r_p, marker='^', label='Norm={}'.format(norm(var)))
    # plt.legend()
    # plt.savefig("images/{}.png".format(frame))

    def plot(data):
        fig = go.Figure(data = data)
        fig.show()
        # plotly.offline.plot(fig, filename='plots_html/{}.html'.format(frame), auto_open=False)
        # plotly.offline.plot(fig, auto_open=False, image = 'png', 
        #     image_filename='plot_image', output_type='file', filename='plots_html/{}.html'.format(frame), validate=False)

    def create_plot_list(x, y, z, r, plot_list, name):
        data = go.Cone(
        x=x,
        y=y,
        z=z,
        u=r[:,0],
        v=r[:,1],
        w=r[:,2],
        colorscale='Blues',
        name=name,
        showlegend=True)
        plot_list.append(data)
        return plot_list

    plot_list = []
    plot_list = create_plot_list(x+0.5, y+0.5, z+0.5, r_p, plot_list, name="pressures")
    plot(plot_list)

