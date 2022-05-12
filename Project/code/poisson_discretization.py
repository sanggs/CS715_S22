import torch
import numpy as np
import sys
import os
# custom import
from conjugate_gradients import CGSolver
from jacobi_solver import JacobiSolver
import utils

class Poisson_Discretization:
    def __init__(self, device='cuda'):
        if device=='cuda':
            self.device = torch.device('cuda:0')
        else:
            self.device = 'cpu'
        # self.construct_stencil()
    
    def initialize_domain(self, domain_properties):
        self.width = domain_properties["width"]
        self.height = domain_properties["height"]
        self.depth = domain_properties["depth"]
        self.pad = False
    
    def initialize_pressures(self):
        self.pressures = torch.zeros([self.width, self.height, self.depth], dtype=torch.float32, device=self.device)
        # padding domain with Neumann cells
        # self.pressures = torch.nn.functional.pad(self.pressures, (1,1,1,1,1,1), "constant", 0)

    def initialize_auxillary(self):
        self.rhs = torch.zeros(size=[self.width, self.height, self.depth], dtype=torch.float32, device=self.device)
        self.lhs = torch.zeros(size=[self.width, self.height, self.depth], dtype=torch.float32, device=self.device)

    def set_domain_flags(self, dom_flags):
        self.dom_flags = dom_flags
        self.domain_flags = torch.tensor(dom_flags, dtype=torch.uint8, device=self.device)
        self.interior_cells = (self.domain_flags == 0).nonzero()
        self.dirichlet_cells = (self.domain_flags == 1).nonzero()
        if self.interior_cells is None:
            print("Domain has no interior cells")
        if self.dirichlet_cells is None:
            print("Domain has no dirichlet cells")
        if self.pad:
            self.interior_cells += 1
            self.dirichlet_cells += 1
        self.construct_system_matrix()
        # self.compute_indexing_ops()

    def compute_indexing_ops(self):
        offset = torch.zeros(self.interior_cells.shape, dtype=torch.long, device=self.device)
        offset[:, 0] = 1
        self.x_plus_indices = self.interior_cells + offset
        offset[:, 0] = -1
        self.x_minus_indices = self.interior_cells + offset
        offset[:, 0] = 0

        offset[:, 1] = 1
        self.y_plus_indices = self.interior_cells + offset
        offset[:, 1] = -1
        self.y_minus_indices = self.interior_cells + offset
        offset[:, 1] = 0

        offset[:, 2] = 1
        self.z_plus_indices = self.interior_cells + offset
        offset[:, 2] = -1
        self.z_minus_indices = self.interior_cells + offset

    def construct_stencil(self):
        b = np.zeros([3, 3, 3])
        b[1, 1, 1] = 6.
        b[1, 1, 0] = b[1, 0, 1] = b[1, 1, 2] = b[1, 2, 1] = -1.
        b[0, 1, 1] = -1.
        b[2, 1, 1] = -1.
        b = torch.tensor(b, dtype=torch.float, device=self.device)
        b.unsqueeze_(0).unsqueeze_(0)
        b.requires_grad = False
        self.conv = torch.nn.Conv3d(1, 1, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), bias=False)
        with torch.no_grad():
            self.conv.weight = torch.nn.Parameter(b)

    def get_linear_index(self, node_index):
        return node_index[2] + node_index[1] * self.depth + node_index[0] * self.height * self.depth

    def construct_system_matrix(self):
        offset_array = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
        offset_array = offset_array.astype(int)

        h = 1./(self.width-1)
        elemental_stiffness_matrix = np.array([[h/3, 0, 0, -(h/12), 0, -(h/12), -(h/12), -(h/12)],
            [0, h/3, -(h/12), 0, -(h/12), 0, -(h/12), -(h/12)],
            [0, -(h/12), h/3, 0, -(h/12), -(h/12), 0, -(h/12)],
            [-(h/12), 0, 0, h/3, -(h/12), -(h/12), -(h/12), 0],
            [0, -(h/12), -(h/12), -(h/12), h/3, 0, 0, -(h/12)],
            [-(h/12), 0, -(h/12), -(h/12), 0, h/3, -(h/12), 0],
            [-(h/12), -(h/12), 0, -(h/12), 0, -(h/12), h/3, 0],
            [-(h/12), -(h/12), -(h/12), 0, -(h/12), 0, 0, h/3]])
        dof = self.width * self.height * self.depth
        self.stiffness_matrix = np.zeros((dof, dof))

        # self.stiffness_matrix = np.zeros((self.width, self.height, self.depth, self.width, self.height, self.depth))

        # dir_nodes = set()

        for cx in range(self.width-1):
            for cy in range(self.height-1):
                for cz in range(self.depth-1):
                    cell_index = np.array([cx, cy, cz])
                    cell_index = cell_index.astype(int)
                    for i in range(8):
                        inode_idx_li = self.get_linear_index(cell_index+offset_array[i])
                        # if (torch.tensor((cell_index+offset_array[i])) in self.dirichlet_cells):
                        #     dir_nodes.add(inode_idx_li)
                        for j in range(8):
                            jnode_idx_li = self.get_linear_index(cell_index+offset_array[j])
                            # if (torch.tensor((cell_index+offset_array[j])) in self.dirichlet_cells):
                            #     dir_nodes.add(jnode_idx_li)
                            self.stiffness_matrix[inode_idx_li, jnode_idx_li] += elemental_stiffness_matrix[i, j]
        

        dir_nodes = self.dirichlet_cells[:, 2] + self.dirichlet_cells[:, 1] * self.depth + self.dirichlet_cells[:, 0] * self.height * self.depth

        self.stiffness_matrix[dir_nodes, :] = 0.0
        self.stiffness_matrix[dir_nodes, dir_nodes] = 1.0

        self.stiffness_matrix = self.stiffness_matrix.reshape((dof, dof))
        
        # import matplotlib.pyplot as plt
        # plt.spy(self.stiffness_matrix)
        # plt.show()
        # print(self.stiffness_matrix)

        # dir_nodes = np.array(list(dir_nodes))
        # dir_nodes = dir_nodes.astype(int)
        # self.stiffness_matrix[self.dirichlet_cells, :] = 0.0
        # self.stiffness_matrix[self.dirichlet_cells, self.dirichlet_cells] = 1.0
        self.stiffness_matrix = torch.tensor(self.stiffness_matrix, dtype=torch.float32)

    def _apply_3d_laplacian_conv(self, p):
        with torch.no_grad():
            q = self.conv(p)
        q.squeeze_(0).squeeze_(0)
        q = torch.nn.functional.pad(q, (1,1,1,1,1,1), "constant", 0)
        return q

    def _apply_3d_laplacian(self, p, q):
        q[:, :, :] = 0.0
        tq = q[self.interior_cells[:, 0], self.interior_cells[:,1], self.interior_cells[:, 2]]
        tq = 6.0 * p[self.interior_cells[:, 0], self.interior_cells[:, 1], self.interior_cells[:, 2]] # center
        
        tq += -1.0 * p[self.x_plus_indices[:, 0], self.x_plus_indices[:, 1], self.x_plus_indices[:, 2]] # x+1
        tq += -1.0 * p[self.x_minus_indices[:, 0], self.x_minus_indices[:, 1], self.x_minus_indices[:, 2]] # x-1
        
        tq += -1.0 * p[self.y_plus_indices[:, 0], self.y_plus_indices[:, 1], self.y_plus_indices[:, 2]] # x+1
        tq += -1.0 * p[self.y_minus_indices[:, 0], self.y_minus_indices[:, 1], self.y_minus_indices[:, 2]] # x-1

        tq += -1.0 * p[self.z_plus_indices[:, 0], self.z_plus_indices[:, 1], self.z_plus_indices[:, 2]] # x+1
        tq += -1.0 * p[self.z_minus_indices[:, 0], self.z_minus_indices[:, 1], self.z_minus_indices[:, 2]] # x-1
        
        q[self.interior_cells[:, 0], self.interior_cells[:,1], self.interior_cells[:, 2]] = tq
        return

    def _apply_fem_laplacian(self, p, q):
        shape = p.shape
        p = p.reshape((-1, 1))
        q = torch.matmul(self.stiffness_matrix, p)
        q = q.reshape(shape)
        return q

    def apply_3d_laplacian(self, p, q):
        q = self._apply_fem_laplacian(p, q)
        # q[:, :, :] = self._apply_3d_laplacian_conv(p[1:self.width+1, 1:self.height+1, 1:self.depth+1].unsqueeze(0).unsqueeze(0))[:, :, :]
        # self._apply_3d_laplacian(p, q)
        return q

    def reset_constrained_particles(self, v, value = 0.0):
        v[self.dirichlet_cells[:, 0], self.dirichlet_cells[:, 1], self.dirichlet_cells[:, 2]] = value

    def apply_perturbation(self):
        noise = torch.rand([self.width, self.height, self.depth], device=self.device)
        # noise[0, :, :] = 0.0
        # noise[self.width-1, :, :] = 0.0
        # noise = torch.nn.functional.pad(noise, (1,1,1,1,1,1), "constant", 0)
        self.reset_constrained_particles(noise, 0.0)
        self.pressures += noise
        print("Adding noise with norm {} to interior cells".format(torch.norm(noise)))

    def _compute_residual(self, lhs):
        self.residual = torch.zeros_like(lhs)
        self.residual = self.apply_3d_laplacian(lhs, self.residual)
        self.reset_constrained_particles(self.residual)
        self.residual = self.rhs - self.residual

    def compute_residual(self):
        self._compute_residual(self.pressures)

    def compute_rhs(self, rhs):
        rhs[:, :, :] = 0

    def _solve(self, lhs):
        self.reset_constrained_particles(self.rhs)
        
        cg = CGSolver(numIteration=2000, minConvergenceNorm=1e-8, width=self.width, height=self.height, depth=self.depth, poissonObject=self)
        cg.solve(lhs, self.rhs)

    def solve(self):
        self._solve(self.pressures)

    def JacobiSolve(self, num_smoother_iters, print_residual=False):
        damping_factor = 2./3.

        jb = JacobiSolver(self)
        jb.solve(self.rhs, self.pressures, damping_factor, num_smoother_iters, print_residual)

