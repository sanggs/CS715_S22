import torch
import numpy as np
import sys
from termcolor import colored
import utils

class CGSolver:
    def __init__(self, numIteration, minConvergenceNorm, width, height, depth, poissonObject):
        self.width = width
        self.height = height
        self.depth = depth
        self.maxIterations = numIteration
        self.minConvergenceNorm = torch.tensor(minConvergenceNorm, dtype = torch.float32)
        self.poissonObject = poissonObject

    def multiplyWithA(self, p, q):
        q = self.poissonObject.apply_3d_laplacian(p, q)
        return q 

    def projectToZero(self, v):
        self.poissonObject.reset_constrained_particles(v, 0.0)
        return

    def solve(self, x, rhs):
        # residual_list = []

        q = torch.zeros_like(x)
        s = torch.zeros_like(x)
        r = torch.zeros_like(x)

        q = self.multiplyWithA(x, q)
        r = rhs.sub(q)
        self.projectToZero(r)
        convergenceNorm = 0
        for i in range(0, self.maxIterations):
            convergenceNorm = utils.norm(r)
            # residual_list.append(convergenceNorm)
            # print("printing convergence norm "+str(convergenceNorm))
            if convergenceNorm < self.minConvergenceNorm:
                # print(colored("Convergence Norm less than threshold: "+str(convergenceNorm)+ " after " + str(i) + " iterations", 'green'))
                # utils.show_convergence_rate(residual_list)
                return
            if i > self.maxIterations:
                print("Ideally should not have come here")
                break
            rho = torch.sum(r*r)
            if i == 0:
                s[:, :, :] = r[:, :, :]
            else:
                s[:, :, :] = ((rho/rhoOld) * s[:, :, :]) + r[:, :, :]
            q = self.multiplyWithA(s, q)
            self.projectToZero(q)

            sDotq = torch.sum(s[:, :, :]*q[:, :, :])
            if sDotq <= 0:
                print("CG matrix appears indefinite or singular, s_dot_q/s_dot_s="
                    +str(sDotq/(torch.sum(s*s))))
            alpha = rho/sDotq
            x[:, :, :] = alpha * s[:, :, :] + x[:, :, :]
            r = -alpha * q + r
            rhoOld = rho

        print("CG Ended after "+str(i)+ " iterations")
        print(convergenceNorm)
        return
