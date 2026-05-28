"""Event6DReaderScaled: build event voxels at a configurable resolution.

The base `Event6DReader` produces voxels at the sensor resolution (1280×720). This
subclass rescales raw event (x, y) coords to the requested grid size before voxelizing,
which is faster than resizing the voxel afterwards. No cache.
"""
import torch

from .my_reader_online import Event6DReader
from .representations import ReconVoxelGrid

class Event6DReaderScaled(Event6DReader):
    def __init__(self, video_dir, ev_width=1280, ev_height=720):
        super().__init__(video_dir)
        self.ev_width = ev_width
        self.ev_height = ev_height
        self.ev_scale_x = ev_width / 1280.0
        self.ev_scale_y = ev_height / 720.0
        self.voxel_grid = ReconVoxelGrid(self.num_bins, ev_height, ev_width, normalize=False)
        self.recon_voxel_grid = ReconVoxelGrid(self.num_bins, ev_height, ev_width, normalize=False)

    def get_event_voxels(self, i, j):
        """Build (start_vox, recon_vox) at ev_height×ev_width.

        `start_vox` covers [t0, t0 + t_len*(j/4)] (cumulative);
        `recon_vox`  covers [t0 + t_len*((j-1)/4), t0 + t_len*(j/4)] (window).
        """
        x, y, t, p = self._load_raw_events(i)
        h, w = self.ev_height, self.ev_width
        if len(x) == 0:
            zeros = torch.zeros(self.num_bins, h, w)
            return zeros, zeros

        if self.ev_scale_x != 1.0 or self.ev_scale_y != 1.0:
            x = (x * self.ev_scale_x).astype('float32')
            y = (y * self.ev_scale_y).astype('float32')

        t_len = float(t[-1] - t[0])
        t_end = t[0] + t_len * j / 4

        mask_cum = t <= t_end
        xc, yc, tc, pc = x[mask_cum], y[mask_cum], t[mask_cum], p[mask_cum]
        if len(xc) > 0:
            tc_n = ((tc - tc[0]) / (tc[-1] - tc[0] + 1e-9)).astype('float32')
            start_vox = self.voxel_grid.convert(
                torch.from_numpy(xc.astype('float32')),
                torch.from_numpy(yc.astype('float32')),
                torch.from_numpy(pc.astype('float32')),
                torch.from_numpy(tc_n),
            )
        else:
            start_vox = torch.zeros(self.num_bins, h, w)

        t_start_r = t[0] + t_len * (j - 1) / 4
        mask_rec = (t >= t_start_r) & (t <= t_end)
        xr, yr, tr, pr = x[mask_rec], y[mask_rec], t[mask_rec], p[mask_rec]
        if len(xr) > 0:
            tr_n = ((tr - tr[0]) / (tr[-1] - tr[0] + 1e-9)).astype('float32')
            recon_vox = self.recon_voxel_grid.convert(
                torch.from_numpy(xr.astype('float32')),
                torch.from_numpy(yr.astype('float32')),
                torch.from_numpy(pr.astype('float32')),
                torch.from_numpy(tr_n),
            )
        else:
            recon_vox = torch.zeros(self.num_bins, h, w)
        return start_vox, recon_vox
