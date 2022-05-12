from numpy.core.fromnumeric import reshape
import torch
import copy
import numpy as np
import sys
from termcolor import colored
import utils
from poisson_discretization import Poisson_Discretization
from poisson_refinement import Poisson_Refinement
from jacobi_solver import JacobiSolver

class MGPCGSolver:
    def __init__(self, domain_properties, lvls=1, device='cuda'):
        assert(lvls >= 1)
        self.levels = lvls
        self.discretizations = []
        for i in range(self.levels):
            self.discretizations.append(Poisson_Discretization(device))
        self.refine = Poisson_Refinement()
        self.discretizations[0].initialize_domain(domain_properties)
        self.discretizations[0].initialize_pressures()
        self.discretizations[0].initialize_auxillary()
        self.discretizations[0].set_domain_flags(domain_properties["flags"])

        # initialize coarse domains
        for i in range(self.levels-1):
            fine_dis = self.discretizations[i]
            coarse_dis = self.discretizations[i+1]

            # domain properties
            coarse_dom_props = {}
            coarse_dom_props["width"] = int(fine_dis.width/2)
            coarse_dom_props["height"] = int(fine_dis.height/2)
            coarse_dom_props["depth"] = int(fine_dis.depth/2)
            coarse_dis.initialize_domain(coarse_dom_props)
            coarse_dis.initialize_auxillary()

            # domain flags
            coarse_dom_flags = self.refine.coarsen_domain_flags(fine_dis.domain_flags, 
                [fine_dis.width, fine_dis.height, fine_dis.depth])
            coarse_dis.set_domain_flags(coarse_dom_flags)

        self.smoother_iters = 5
        self.viz_vcycle_output = False

        self.max_convergence_norm = 1e-7
        self.max_cg_iters = 400

    def multiplyWithA(self, p, q, lvl=0):
        q = self.discretizations[lvl].apply_3d_laplacian(p, q)
        return q 

    def projectToZero(self, v, lvl=0):
        self.discretizations[lvl].reset_constrained_particles(v, 0.0)
        return

    def compute_residual(self, u, b, lvl):
        q = torch.zeros_like(u)
        q = self.multiplyWithA(u, q, lvl)
        self.projectToZero(q, lvl)
        return b.sub(q)

    def apply_perturbation(self):
        self.discretizations[0].apply_perturbation()

    def run_v_cycle(self, u, b, v_cycles=1):
        fine_dis = self.discretizations[0]
        fine_dis.lhs[:, :, :] = 0.0
        # fine_dis.rhs[:, :, :] = b[:, :, :]
        for v in range(v_cycles):
            i = 0
            fine_dis = self.discretizations[i]
            coarse_dis = self.discretizations[i+1]
            # refine = self.refinements[i]
            q = torch.zeros_like(fine_dis.lhs)
            q = self.multiplyWithA(b, q)
            coarse_dis.rhs = self.refine.restrict((b + (1./9.) * q), fine_dis.width, fine_dis.height, fine_dis.depth)
            coarse_dis.lhs[:, :, :] = 0.0

            for i in range(1, self.levels-1, 1):
                # print(colored("Downward stroke: fine_lvl = {} coarse_lvl = {}".format(i, i+1), 'blue'))
                fine_dis = self.discretizations[i]
                coarse_dis = self.discretizations[i+1]
                # refine = self.refinements[i]
                smoother = JacobiSolver(fine_dis)
                smoother.solve(rhs=fine_dis.rhs, lhs=fine_dis.lhs, damping_factor=(2./3.), num_smoother_iters=self.smoother_iters)
                res = self.compute_residual(fine_dis.lhs, fine_dis.rhs, i)
                # print("Smoothing at level = {} residual norm = {}".format(i, utils.norm(fine_dis.residual)))
                coarse_dis.rhs = self.refine.restrict(res, fine_dis.width, fine_dis.height, fine_dis.depth)
                coarse_dis.lhs[:, :, :] = 0.0
            
            cdis = self.discretizations[self.levels-1]
            cdis._solve(cdis.lhs)
            cdis._compute_residual(cdis.lhs)
            # print("Full Solve at level {}. Residual = {}".format(self.levels-1, utils.norm(cdis.residual)))

            for i in range(self.levels-2, -1, -1):
                # print(colored("Upward stroke: fine_lvl = {} coarse_lvl = {}".format(i, i+1), 'blue'))
                fine_dis = self.discretizations[i]
                coarse_dis = self.discretizations[i+1]
                # refine = self.refinements[i]
                prolongated_res = self.refine.prolongate(coarse_dis.lhs, coarse_dis.width, coarse_dis.height, coarse_dis.depth)
                fine_dis.lhs += prolongated_res
                smoother = JacobiSolver(fine_dis)
                if (i== 0):
                    smoother.solve(rhs=b, lhs=fine_dis.lhs, damping_factor=(2./3.), num_smoother_iters=self.smoother_iters)
                else:
                    smoother.solve(rhs=fine_dis.rhs, lhs=fine_dis.lhs, damping_factor=(2./3.), num_smoother_iters=self.smoother_iters)
                # print(torch.norm(fine_dis.lhs))
                fine_dis._compute_residual(fine_dis.lhs)
                # print("Smoothing at level = {} residual norm = {}".format(i, utils.norm(fine_dis.residual)))

            fine_dis = self.discretizations[0]
            fine_dis._compute_residual(fine_dis.lhs)
            # print(colored("V-cycle num={} Residual Norm at finest level={}".format(v, utils.norm(fine_dis.residual)), 'yellow'))

            if (self.viz_vcycle_output):
                utils.visualize_slice(fine_dis.lhs, 8, "z")

        u[:, :, :] = self.discretizations[0].lhs[:, :, :]

    def solve(self):
        fine_dis = self.discretizations[0]
        self._solve(fine_dis.pressures, fine_dis.rhs, fine_dis.width, fine_dis.height, fine_dis.depth)

    def _solve(self, x, b, width, height, depth, viz=False):
        residual_list = []
        if (viz):
            utils.show_3D_plot(x, 0)

        fine_dis = self.discretizations[0]
        p = torch.zeros_like(fine_dis.pressures)
        r = torch.zeros_like(fine_dis.pressures)
        z = torch.zeros_like(fine_dis.pressures)

        # q = torch.zeros_like(x)
        p = self.multiplyWithA(x, p)
        self.projectToZero(p)
        r = b - p
        convergence_norm = utils.norm(r)
        residual_list.append(convergence_norm.item())

        if (convergence_norm < self.max_convergence_norm):
            # print(colored("Convergence Norm less than threshold: "+str(convergence_norm)+ " after 0 iterations", 'blue'))
            return

        self.run_v_cycle(p, r, 1)
        rho = torch.sum(p*r)
        rho_new = None
        for i in range(0, self.max_cg_iters):
            z = self.multiplyWithA(p, z)
            self.projectToZero(z)
            sigma = torch.sum(p*z)
            alpha = rho/sigma
            r = r.sub(alpha*z)
            convergence_norm = utils.norm(r)
            residual_list.append(convergence_norm.item())
            # print(colored("Convergence norm = {} Iteration = {}".format(convergence_norm, i), 'blue'))
            if i > self.max_cg_iters:
                print("Ideally should not have come here")
                break
            if convergence_norm < self.max_convergence_norm:
                x[:, :, :] += alpha * p[:, :, :]
                print(colored("Convergence Norm less than threshold: "+str(convergence_norm)+ " after " + str(i) + " iterations", 'blue'))
                utils.show_convergence_rate(residual_list)
                return
            self.run_v_cycle(z, r, 1)
            rho_new = torch.sum(z*r)
            beta = rho_new/rho
            rho = rho_new
            x[:, :, :] += alpha * p[:, :, :]
            p[:, :, :] = z[:, :, :] + beta * p[:, :, :]
            
            if (viz):
                utils.show_3D_plot(x, i+1)            

        print("Ended after "+str(i)+ " iterations")
        print(convergence_norm)
        return
