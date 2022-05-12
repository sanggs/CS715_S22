import torch
import sys
import math
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
from colorama import init
from termcolor import colored
import timeit
# custom import
from poisson_discretization import Poisson_Discretization
from multigrid_solver import Multigrid_Solver
from mgpcg_solver import MGPCGSolver
import utils

class Poisson_Solver:
    def __init__(self, domain_properties, device='cuda', solver=2, levels=1):
        # No need for galerkin coarsening!
        self.device = device
        self.discretizations = []
        self.solver = solver
        if (self.solver == 0 or self.solver == 1):
            self.discretizations.append(Poisson_Discretization(device))
            self.discretizations[0].initialize_domain(domain_properties)
            self.discretizations[0].initialize_pressures()
            self.discretizations[0].initialize_auxillary()
            self.discretizations[0].set_domain_flags(domain_properties["flags"])
        elif (self.solver == 2):
            self.mg_solver = Multigrid_Solver(levels, domain_properties, device)
        elif (self.solver == 3):
            self.mgpcg_solver = MGPCGSolver(domain_properties, levels, device)
        else:
            print("Unknown solver mode {}".format(self.solver))
            exit(1)
        self.num_v_cycles = 50
        self.num_smoother_iters = 1500

    def ApplyPerturbation(self):
        if (self.solver == 2):
            self.mg_solver.apply_perturbation()
        elif (self.solver == 3):
            self.mgpcg_solver.apply_perturbation()
        else:
            self.discretizations[0].apply_perturbation()

    def GetSolvedState(self):
        fine_dis = None
        if (self.solver == 0 or self.solver == 1):
            fine_dis = self.discretizations[0]
        elif (self.solver == 2):
            fine_dis = self.mg_solver.discretizations[0]
        elif (self.solver == 3):
            fine_dis = self.mgpcg_solver.discretizations[0]
        assert(fine_dis is not None)
        return fine_dis.pressures[1:fine_dis.width+1, 1:fine_dis.height+1, 1:fine_dis.depth+1]

    def Solve(self):
        if (self.solver == 0):
            self.CGSolve()
        elif(self.solver == 1):
            self.JacobiSolve(self.num_smoother_iters)
        elif (self.solver == 2):
            self.MGSolve(self.num_v_cycles)
        elif (self.solver == 3):
            self.MGPCGSolve()

    def CGSolve(self):
        start = timeit.default_timer()
        self.discretizations[0].compute_rhs(self.discretizations[0].rhs)
        self.discretizations[0].solve()
        print("Time taken for CG solve = {}s".format(timeit.default_timer()-start))

    def JacobiSolve(self, num_smoother_iters):
        self.discretizations[0].compute_rhs(self.discretizations[0].rhs)
        self.discretizations[0].JacobiSolve(num_smoother_iters, True)

    def MGSolve(self, num_v_cycles):
        self.mg_solver.Solve(num_v_cycles)

    def MGPCGSolve(self):
        start = timeit.default_timer()
        #with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        #    with record_function("solve"):
        #        self.mgpcg_solver.solve()
        #print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_memory_usage", row_limit=10))
        self.mgpcg_solver.solve()
        print("Time taken for MGPCG solve = {}s".format(timeit.default_timer()-start))

if __name__ == "__main__":
    init()
    print ('Number of arguments:', len(sys.argv), 'arguments.')
    print ('Argument List:', str(sys.argv))
    if (len(sys.argv) != 4 and len(sys.argv) != 5):
        print("Please specify the following to run the program")
        print("python3 poisson_solver_3d.py <device> <grid size> <solver mode> <levels (optional)>")
        print("Example: python3 poisson_solver_3d.py 0 64 3 4")
        print("<device> = 0 for CPU, 1 for GPU")
        print("<solver mode> = 0 for CG, 3 for MGPCG, 1 for Jacobi, 2 for Multigrid")
        print("<levels> is optional and used when solver mode is 3 (MGPCG) or 2 (Multigrid)")
        exit(1)
    dev = 'cpu'
    if int(sys.argv[1]) == 0:
        print("Using CPU")
        dev = 'cpu'
    elif int(sys.argv[1]) == 1:
        print("Using GPU")
        dev = 'cuda'
    else:
        print("Unknown device option. Defaulting to CPU {}".format(dev))
    
    solver = int(sys.argv[3]) # 0 -> CG Solve, 1 -> Jacobi Solve, 2 -> Multigrid Solve, 3 -> MGPCG Solver
    if (solver != 0 and solver != 3 and solver != 2 and solver != 1):
        print("Invalid solver mode = {}".format(solver))
        print("<solver mode> = 0 for CG, 3 for MGPCG, 1 for Jacobi, 2 for Multigrid")
        exit(1)

    ngs = 64
    if solver == 3 or solver == 2:
        gs = float(sys.argv[2])
        ngs = math.pow(2, math.ceil(math.log(gs, 2)))
        if (int(ngs) != int(gs)):
            print("Using nearest grid size to {} which is a power of 2: {}".format( int(gs), int(ngs)))
    else:
        ngs = int(sys.argv[2])

    grid_sizes = []
    grid_sizes.append(int(ngs))

    levels = 1
    if (solver == 3):
        levels = int(sys.argv[4])

    print("Config specified is as follows:")
    print("Device = {}".format(dev))
    print("Grid size = {}".format(grid_sizes[0]))
    print("Solver = {}".format(solver))
    print("Levels = {}".format(levels))

    domain_props = {}
    for grid_size in grid_sizes:
        domain_props["width"] = grid_size
        domain_props["height"] = grid_size
        domain_props["depth"] = grid_size
        
        print("Domain size = [{}, {}, {}]".format(domain_props["width"], domain_props["height"], domain_props["depth"]))

        flags = np.zeros([domain_props["width"], domain_props["height"], domain_props["depth"]], dtype=np.ubyte)
        # left face and right faces are dirichlet
        flags[0, :, :] = 1
        flags[domain_props["width"]-1, :, :] = 1
        domain_props["flags"] = flags
        
        start = timeit.default_timer()
        ps = Poisson_Solver(domain_properties=domain_props, device=dev, solver=solver, levels=levels)
        print("Time taken for Initialization = {}s".format(timeit.default_timer()-start))

        frames = 1
        for f in range(frames):
            ps.ApplyPerturbation()
            # utils.visualize_slice(ps.GetSolvedState(), 8, "z")
            # utils.show_3D_plot(ps.mgpcg_solver.discretizations[0].pressures)
            ps.Solve()
            # utils.visualize_slice(ps.GetSolvedState(), 8, "z")
            # utils.show_3D_plot(ps.mgpcg_solver.discretizations[0].pressures)

