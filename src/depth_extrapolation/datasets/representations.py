import torch
import torch.nn as nn

class EventRepresentation:
    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        raise NotImplementedError

class VoxelGrid_down(EventRepresentation):
    def __init__(self, channels: int, height: int, width: int, normalize: bool, size: int):
        self.voxel_grid_cal = torch.zeros((channels, height, width), dtype=torch.float, requires_grad=False)
        self.voxel_grid = torch.zeros((channels, height//4, width//4), dtype=torch.float, requires_grad=False)
        self.nb_channels = channels
        self.normalize = normalize
        self.size = size

    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        assert x.shape == y.shape == pol.shape == time.shape
        assert x.ndim == 1

        C, H, W = self.voxel_grid_cal.shape
        with torch.no_grad():

            self.voxel_grid = self.voxel_grid.to(pol.device)
            self.voxel_grid_cal = self.voxel_grid_cal.to(pol.device)
            # voxel_grid = self.voxel_grid.clone()
            voxel_grid_time = self.voxel_grid.clone()
            voxel_grid_pol = self.voxel_grid_cal.clone()

            t_norm = time
            t_norm = (C - 1) * (t_norm-t_norm[0]) / (t_norm[-1]-t_norm[0])
            # t0 = time

            x0 = x.int()
            y0 = y.int()
            t0 = t_norm.int()
            if int(pol.min()) == -1:
                value = pol
            else:
                value = 2*pol-1

            mask = (x0 < W) & (x0 >= 0) & (y0 < H) & (y0 >= 0) & (t0 >= 0) & (t0 < self.nb_channels)
            # interp_weights = value * (1 - (x0-x).abs()) * (1 - (y0-y).abs()) * (1 - (0 - t_norm).abs())
            # interp_weights = value * t_norm

            interp_weights = value
            time_weights = time
            index = H * W * t0.long() + \
                    W * y0.long() + \
                    x0.long()
            voxel_grid_pol.put_(index[mask], interp_weights[mask], accumulate=False)

            index = H//4 * W//4 * t0.long() + \
                    W//4 * (y0//4).long() + \
                    (x0//4).long()
            voxel_grid_time.put_(index[mask], time_weights[mask], accumulate=False)

            # unfold = nn.Unfold(kernel_size=(4,4), dilation=1, padding=0, stride=4)(voxel_grid_pol.unsqueeze(0))
            m = nn.AvgPool2d(self.size, stride=self.size)
            voxel_grid_pol = m(voxel_grid_pol)
            voxel_grid_pol[voxel_grid_pol>0] = 1
            voxel_grid_pol[voxel_grid_pol<0] = -1
            # voxel_grid_time = m(voxel_grid_time)
            voxel_grid = voxel_grid_time * voxel_grid_pol
            # .squeeze(0).reshape(C, -1, H*4, W*4).transpose(0, 1)
            voxel_grid = voxel_grid.chunk(15)
            voxel_grid = torch.sum(torch.stack(voxel_grid, dim=0), dim=1)
        return voxel_grid
