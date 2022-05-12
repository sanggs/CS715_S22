This project contains the code for a GPU-based multigrid poisson solver for large scale fluid simulations
An example problem is solved and the pressure is visualized for every MGPCG iteration in the file out.mp4

Steps to run the code:

0. Install the dependencies by running:
    cd Project/code
    pip3 install requirements.txt

1. To run the code use the following command:
    python3 poisson_solver_3d.py <device> <grid size> <solver mode> <levels (optional)>
    <device> = 0 for CPU, 1 for GPU
    <grid size> = integer indicating grid size (only supporting cubic domains for now)
    <solver mode> = 0 for CG, 3 for MGPCG, 1 for Jacobi, 2 for Multigrid
    <levels> is optional and used when solver mode is 3 (MGPCG) or 2 (Multigrid)

    Example: python3 poisson_solver_3d.py 0 64 3 4

NOTE: solver mode 2 (Multigrid) is not tested thoroughly and is not used in the MGPCG solver implementation.

2. Uncomment lines 160 and 163 in poisson_solver_3d.py to visualize the input and output of the solve step.

Author: Sangeetha Grama Srinivasan
Email: sgsrinivasa2@wisc.edu
Please contact the author if any problems with running the code.
