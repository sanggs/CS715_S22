import torch
from termcolor import colored
from poisson_refinement import Poisson_Refinement
from poisson_discretization import Poisson_Discretization
import utils

class Multigrid_Solver:
    def __init__(self, lvls, domain_properties, device) -> None:
        assert(lvls >= 1)
        self.levels = lvls
        self.discretizations = []
        self.refinements = []
        for i in range(self.levels):
            self.discretizations.append(Poisson_Discretization(device))
        for i in range(self.levels-1):
            self.refinements.append(Poisson_Refinement())
        
        self.discretizations[0].initialize_domain(domain_properties)
        self.discretizations[0].initialize_pressures()
        self.discretizations[0].initialize_auxillary()
        self.discretizations[0].set_domain_flags(domain_properties["flags"])
        
        # initialize coarse domains
        for i in range(self.levels-1):
            refinement = self.refinements[i]
            fine_dis = self.discretizations[i]
            coarse_dis = self.discretizations[i+1]

            # domain properties
            coarse_dom_props = {}
            coarse_dom_props["width"] = int(fine_dis.width/2)
            coarse_dom_props["height"] = int(fine_dis.height/2)
            coarse_dom_props["depth"] = int(fine_dis.depth/2)
            coarse_dis.initialize_domain(coarse_dom_props)
            coarse_dis.initialize_pressures()
            coarse_dis.initialize_auxillary()

            # domain flags
            coarse_dom_flags = refinement.coarsen_domain_flags(fine_dis.domain_flags, 
                [fine_dis.width, fine_dis.height, fine_dis.depth])
            coarse_dis.set_domain_flags(coarse_dom_flags)

        self.viz_vcycle_output = False

    def apply_perturbation(self):
        self.discretizations[0].apply_perturbation()

    def JacobiSmooth(self, discretization, num_smoother_iters):
        discretization.JacobiSolve(num_smoother_iters)

    def GetSolvedState(self):
        fine_dis = self.discretizations[0]
        return fine_dis.pressures[1:fine_dis.width+1, 1:fine_dis.height+1, 1:fine_dis.depth+1]

    def Solve(self, num_v_cycles):
        for v in range(num_v_cycles):
            for i in range(0, self.levels-1, 1):
                print(colored("Downward stroke: fine_lvl = {} coarse_lvl = {}".format(i, i+1), 'blue'))
                fine_dis = self.discretizations[i]
                coarse_dis = self.discretizations[i+1]
                self.JacobiSmooth(fine_dis, 10)
                fine_dis.compute_residual()
                # print(fine_dis.pressures.shape)
                print("Smoothing at level = {} residual norm = {}".format(i, utils.norm(fine_dis.residual)))
                refine = self.refinements[i]
                coarse_dis.rhs = refine.restrict(fine_dis.residual, fine_dis.width, fine_dis.height, fine_dis.depth)
                coarse_dis.pressures[:, :, :] = 0.0
            
            cdis = self.discretizations[self.levels-1]
            cdis.solve()
            cdis.compute_residual()
            print("Full Solve at level {}. Residual = {}".format(self.levels-1, utils.norm(cdis.residual)))

            for i in range(self.levels-2, -1, -1):
                print(colored("Upward stroke: fine_lvl = {} coarse_lvl = {}".format(i, i+1), 'blue'))
                fine_dis = self.discretizations[i]
                coarse_dis = self.discretizations[i+1]
                refine = self.refinements[i]
                prolongated_res = refine.prolongate(coarse_dis.pressures, coarse_dis.width, coarse_dis.height, coarse_dis.depth)
                fine_dis.pressures += prolongated_res
                self.JacobiSmooth(fine_dis, 10)
                fine_dis.compute_residual()
                print("Smoothing at level = {} residual norm = {}".format(i, utils.norm(fine_dis.residual)))
            
            fine_dis = self.discretizations[0]
            fine_dis.compute_residual()
            print(colored("V-cycle num={} Residual Norm at finest level={}".format(v, utils.norm(fine_dis.residual)), 'yellow'))

            if (self.viz_vcycle_output):
                utils.visualize_slice(self.GetSolvedState(), 8, "z")

