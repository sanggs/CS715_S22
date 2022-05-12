import torch
import utils
from termcolor import colored

class JacobiSolver():
    def __init__(self, poissonObject) -> None:
        self.poissonObject = poissonObject

    def multiplyWithA(self, p, q):
        q = self.poissonObject.apply_3d_laplacian(p, q)
        return q 

    def projectToZero(self, p):
        self.poissonObject.reset_constrained_particles(p, 0.0)
        return

    def solve(self, rhs, lhs, damping_factor, num_smoother_iters, print_residual=False):
        q = torch.empty_like(lhs)
        for i in range(num_smoother_iters):
            q = self.multiplyWithA(lhs, q)
            res = rhs - q
            self.projectToZero(res)
            lhs += damping_factor * res * (1./6.)
        if (print_residual):
            print(colored("Residual norm after {} smoother iters = {}".format(num_smoother_iters, utils.norm(res)), 'red'))
