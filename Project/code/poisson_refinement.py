import torch
import copy
import numpy as np

class Poisson_Refinement:
    def coarsen_domain_flags(self, fine_flags, fine_grid_dim):
        fine_flags.unsqueeze_(0)
        fine_flags.unsqueeze_(0)
        maxpool = torch.nn.MaxPool3d((2, 2, 2), stride=2)
        coarse_grid_flags = maxpool(fine_flags.type(torch.float)).type(torch.uint8)
        coarse_grid_dim = []
        for i in fine_grid_dim:
            coarse_grid_dim.append(int(i/2))
        
        coarse_grid_flags.squeeze_(0).squeeze_(0)
        assert(list(coarse_grid_flags.shape) == coarse_grid_dim)
        fine_flags.squeeze_()
        fine_flags.squeeze_()
        r = torch.tensor(self.get_restriction_kernel(), dtype=torch.float32)
        if (fine_flags.is_cuda):
            r = r.cuda()
        r.resize_([4, 4, 4])
        r.unsqueeze_(0).unsqueeze_(0)
        r.requires_grad = False

        self.conv = torch.nn.Conv3d(1, 1, kernel_size=(4, 4, 4), stride=2, padding=(1, 1, 1), bias=False)
        with torch.no_grad():
            self.conv.weight = torch.nn.Parameter(r)
        return coarse_grid_flags.cpu().numpy()

    def restrict(self, fine_var, fine_width, fine_height, fine_depth):
        # return self.conv_restrict(fine_var, fine_width, fine_height, fine_depth)
        return self.interp_restrict(fine_var, fine_width, fine_height, fine_depth)

    def interp_restrict(self, fine_var, fine_width, fine_height, fine_depth):
        var = copy.deepcopy(fine_var)
        var = var[:, :, :]
        var.unsqueeze_(0)
        var.unsqueeze_(0)
        var = torch.nn.functional.interpolate(var, scale_factor=0.5, mode='trilinear', align_corners=True, recompute_scale_factor=True)
        var.squeeze_().squeeze_()
        # var = torch.nn.functional.pad(var, (1,1,1,1,1,1), "constant", 0)
        return var
    
    def get_restriction_kernel(self):
        b = np.array([1./8., 3./8., 3./8., 1./8.])
        r = np.kron(np.kron(b, b), b)
        return r

    def conv_restrict(self, fine_var, fine_width, fine_height, fine_depth):
        var = fine_var[:, :, :]
        var.unsqueeze_(0)
        var.unsqueeze_(0)
        with torch.no_grad():
            var = self.conv(var)
        var.squeeze_().squeeze_()
        # var = torch.nn.functional.pad(var, (1,1,1,1,1,1), "constant", 0)
        return var

    def prolongate(self, coarse_var, coarse_width, coarse_height, coarse_depth):
        var = coarse_var[:, :, :]
        var.unsqueeze_(0)
        var.unsqueeze_(0)
        with torch.no_grad():
            var = torch.nn.functional.interpolate(var, scale_factor=2.0, mode='trilinear', align_corners=True, recompute_scale_factor=True)
        var.squeeze_().squeeze_()
        # var = torch.nn.functional.pad(var, (1,1,1,1,1,1), "constant", 0)
        return var
