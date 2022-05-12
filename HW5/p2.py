import distmesh as dm
from matplotlib import projections
from matplotlib.collections import LineCollection, PolyCollection
import numpy as np
import matplotlib.pyplot as plt


def plot_fem_mesh(nodes_x, nodes_y, elements):
    '''
    Code excerpt taken from: https://stackoverflow.com/questions/52202014/how-can-i-plot-2d-fem-results-using-matplotlib
    '''
    plt.cla()
    plt.clf()
    for element in elements:
        x = [nodes_x[element[i]] for i in range(len(element))]
        y = [nodes_y[element[i]] for i in range(len(element))]
        plt.fill(x, y, edgecolor='black', fill=False)
    plt.scatter(nodes_x, nodes_y)
    plt.axis('equal')
    plt.title('Plotting FEM Mesh for circle domain')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.savefig('2a.png')
    # plt.show()

def extract_boundary(p, t):
    boundary_edges = dm.boundedges(p, t)
    boundary_nodes = np.reshape(boundary_edges, -1)
    boundary_nodes = np.unique(boundary_nodes)
    return boundary_nodes

def build_elemental_stiffness_matrix(p, t):
    uvw = np.array([p[1]-p[2], p[2]-p[0], p[0]-p[1]])
    determinant = 0.5 * np.abs(uvw[1, 0] * uvw[2, 1] - uvw[1, 1] * uvw[2, 0])
    assert(determinant != 0)
    dim = t.shape[0]
    Ak = np.zeros((dim, dim))
    for i in range(0, dim):
        for j in range(0, dim):
            Ak[i, j] = np.dot(uvw[i], uvw[j])
    Ak = 0.25 * 1./determinant * Ak
    return Ak

def FEM_Stiffness_Matrix(p, triangles):
    num_nodes = p.shape[0]
    dim = 3
    A = np.zeros((num_nodes, num_nodes))
    for tr in triangles:
        p_tr = np.array([p[tr[0]], p[tr[1]], p[tr[2]]])
        Ak = build_elemental_stiffness_matrix(p_tr, tr)
        for i in range(0, dim):
            for j in range(0, dim):
                A[tr[i], tr[j]] += Ak[i, j]
    return A

def plot_sparsity_pattern(A):
    plt.cla()
    plt.clf()
    plt.spy(A)
    plt.show()

def enforce_dirichlet(A, boundary_nodes):
    A[boundary_nodes, :] = 0
    A[boundary_nodes, boundary_nodes] = 1.0
    return A

def RHS(p, triangles):
    num_nodes = p.shape[0]
    dim = p.shape[1]+1
    b = np.zeros(num_nodes)
    for tr in triangles:
        uvw = np.array([p[tr[1]]-p[tr[2]], p[tr[2]]-p[tr[0]], p[tr[0]]-p[tr[1]]])
        determinant = uvw[1, 0] * uvw[2, 1] - uvw[1, 1] * uvw[2, 0]
        for i in range(dim):
            b[tr[i]] += 1./6. * determinant
    return b

def plot_solution(x, y, t, u):
    plt.cla()
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x, y, u, triangles=t, cmap=plt.cm.viridis)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def get_exact_solution(p):
    r_squared = p[:, 0] * p[:, 0] + p[:, 1] * p[:, 1]
    u = -0.25 * r_squared + 0.25
    return u

def discretize_and_solve_on_circle(h0=0.2):
    fd = lambda p: np.sqrt((p**2).sum(1))-1.0
    p, t = dm.distmesh2d(fd, dm.huniform, h0, (-1,-1,1,1))
    plot_fem_mesh(p[:, 0], p[:, 1], t)

    x = p[:, 0]
    y = p[:, 1]

    boundary_nodes = extract_boundary(p, t)

    A = FEM_Stiffness_Matrix(p, t)
    A = enforce_dirichlet(A, boundary_nodes)

    b = RHS(p, t)
    b[boundary_nodes] = 0

    u = np.linalg.solve(A, b)
    u_exact = get_exact_solution(p)

    print("h={} u={}".format(h0, np.linalg.norm(u)))

    # plt.cla()
    # plt.clf()
    # fig = plt.figure(figsize=plt.figaspect(0.5))
    # ax = fig.add_subplot(121, projection='3d')
    # ax.plot_trisurf(x, y, u, triangles=t, cmap=plt.cm.viridis)
    # ax.set_title("FEM Solution")
    # ax.set_xlabel("x-axis")
    # ax.set_ylabel("y-axis")

    # ax = fig.add_subplot(122, projection='3d')
    # ax.plot_trisurf(x, y, u_exact, triangles=t, cmap=plt.cm.viridis)
    # ax.set_title("Exact Solution")
    # ax.set_xlabel("x-axis")
    # ax.set_ylabel("y-axis")
    # plt.title("Exact solution")
    # plt.savefig("2b_fem_vs_exact.png")

    # plt.cla()
    # plt.clf()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_trisurf(x, y, u-u_exact, triangles=t, cmap=plt.cm.viridis)
    # plt.xlabel("x-axis")
    # plt.ylabel("y-axis")
    # plt.title("Plotting error against exact solution")
    # plt.savefig("2b_error.png")
    # plot_solution(x, y, t, u_exact)

def discretize_and_solve_on_ellipse():
    fd = lambda p: (p[:, 0]**2/(4) + p[:, 1]**2/(1))-1.0
    [p,t]=dm.distmesh2d(fd,dm.huniform,0.2,(-2,-1,2,1))
    
    x = p[:, 0]
    y = p[:, 1]

    boundary_nodes = extract_boundary(p, t)

    A = FEM_Stiffness_Matrix(p, t)
    A = enforce_dirichlet(A, boundary_nodes)

    b = RHS(p, t)
    b[boundary_nodes] = 0

    u = np.linalg.solve(A, b)

    plt.cla()
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x, y, u, triangles=t, cmap=plt.cm.viridis)
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.title("Plotting FEM Solution on Ellipse")
    plt.savefig("2c_ellipse.png")

    # plot_solution(x, y, t, u)

def discretize_and_solve_on_polygon():
    pv = np.array([(-0.4,-0.5),(0.4,-0.2),(0.4,-0.7),(1.5,-0.4),(0.9,0.1),
                   (1.6,0.8),(0.5,0.5),(0.2,1.0),(0.1,0.4),(-0.7,0.7),
                   (-0.4,-0.5)])
    fd = lambda p: dm.dpoly(p, pv)
    p, t = dm.distmesh2d(fd, dm.huniform, 0.1, (-1,-1, 2,1), pv)

    x = p[:, 0]
    y = p[:, 1]

    boundary_nodes = extract_boundary(p, t)

    A = FEM_Stiffness_Matrix(p, t)
    A = enforce_dirichlet(A, boundary_nodes)

    b = RHS(p, t)
    b[boundary_nodes] = 0

    u = np.linalg.solve(A, b)

    plt.cla()
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x, y, u, triangles=t, cmap=plt.cm.viridis)
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.title("Plotting FEM Solution on Polygon")
    plt.savefig("2c_polygon.png")

    # plot_solution(x, y, t, u)

def get_concave_mesh(h0):
    fd = lambda p: dm.ddiff(dm.drectangle(p,-1,1,-1,1), dm.dcircle(p,0,0,0.5))
    fh = lambda p: 0.05+0.3*dm.dcircle(p,0,0,0.5)
    p, t = dm.distmesh2d(fd, fh, h0, (-1,-1,1,1), [(-1,-1),(-1,1),(1,-1),(1,1)])
    return p, t

def discretize_and_solve_on_concave_region(h0):
    p, t = get_concave_mesh(h0)
    
    x = p[:, 0]
    y = p[:, 1]
    plot_fem_mesh(x, y, t)

    boundary_nodes = extract_boundary(p, t)
    # plt.scatter(p[boundary_nodes, 0], p[boundary_nodes, 1])
    # plt.show()

    A = FEM_Stiffness_Matrix(p, t)
    A = enforce_dirichlet(A, boundary_nodes)

    b = RHS(p, t)
    b[boundary_nodes] = 0

    u = np.linalg.solve(A, b)

    print("h={} Norm of u={}".format(h0, np.linalg.norm(u)))

    plt.cla()
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x, y, u, triangles=t, cmap=plt.cm.viridis)
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.title("Plotting FEM Solution on Concave region, h={}".format(h0))
    plt.savefig("2e_{}.png".format(h0))

    # plot_solution(x, y, t, u)

if __name__ == '__main__':
    discretize_and_solve_on_circle(h0=0.1)
    discretize_and_solve_on_circle(h0=0.2)
    discretize_and_solve_on_circle(h0=0.5)
    discretize_and_solve_on_ellipse()
    discretize_and_solve_on_polygon()
    discretize_and_solve_on_concave_region(0.01)
    discretize_and_solve_on_concave_region(0.02)
    discretize_and_solve_on_concave_region(0.05)    
