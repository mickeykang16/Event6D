from .cbmnet_submodule import *
# from cbmnet_submodule import *
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules import conv
try:
    from correlation_package.correlation import Correlation
except:
    from .CorrelationLayer.correlation_torch import CorrTorch as Correlation
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, trunc_normal_, to_2tuple
from functools import reduce, lru_cache
import torch.nn.functional as tf
from torch.autograd import Variable
from einops import rearrange
import math
import numbers
import collections
from .models import register

def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, isReLU=True):
    if isReLU:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True)
        )

def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)

class encoder_event_flow(nn.Module):
    def __init__(self, num_chs):
        super(encoder_event_flow, self).__init__()
        self.conv1 = conv_resblock_one_small(num_chs[0], num_chs[1], stride=1)
        self.conv2 = conv_resblock_one_small(num_chs[1], num_chs[2], stride=1)
        self.conv3 = conv_resblock_one_small(num_chs[2], num_chs[3], stride=2)
        self.conv4 = conv_resblock_one_small(num_chs[3], num_chs[4], stride=2)

    def forward(self, im):
        x = self.conv1(im)
        c11 = self.conv2(x)
        c12 = self.conv3(c11)
        c13 = self.conv4(c12)
        return c11, c12, c13

class encoder_event_for_image_flow(nn.Module):
    def __init__(self, num_chs):
        super(encoder_event_for_image_flow, self).__init__()
        self.conv1 = conv_resblock_one_small(num_chs[0], num_chs[1], stride=1)
        self.conv2 = conv_resblock_one_small(num_chs[1], num_chs[2], stride=2)
        self.conv3 = conv_resblock_one_small(num_chs[2], num_chs[3], stride=2)
        self.conv4 = conv_resblock_one_small(num_chs[3], num_chs[4], stride=2)

    def forward(self, im):
        x = self.conv1(im)
        c11 = self.conv2(x)
        c12 = self.conv3(c11)
        c13 = self.conv4(c12)
        return c11, c12, c13

class encoder_image_for_image_flow(nn.Module):
    def __init__(self, num_chs):
        super(encoder_image_for_image_flow, self).__init__()
        self.conv1 = conv_resblock_one_small(num_chs[0], num_chs[1], stride=1)
        self.conv2 = conv_resblock_one_small(num_chs[1], num_chs[2], stride=2)
        self.conv3 = conv_resblock_one_small(num_chs[2], num_chs[3], stride=2)
        self.conv4 = conv_resblock_one_small(num_chs[3], num_chs[4], stride=2)

    def forward(self, image):
        x = self.conv1(image)
        f1 = self.conv2(x)
        f2 = self.conv3(f1)
        f3 = self.conv4(f2)
        return f1, f2, f3

def upsample2d(inputs, target_as, mode="bilinear"):
    _, _, h, w = target_as.size()
    return tf.interpolate(inputs, [h, w], mode=mode, align_corners=True)

def upsample2d_hw(inputs, h, w, mode="bilinear"):
    return tf.interpolate(inputs, [h, w], mode=mode, align_corners=True)

class DenseBlock(nn.Module):
    def __init__(self, ch_in):
        super(DenseBlock, self).__init__()
        self.conv1 = conv(ch_in, 32)
        self.conv2 = conv(ch_in + 32, 32)
        # self.conv3 = conv(ch_in + 256, 96)
        # self.conv4 = conv(ch_in + 352, 64)
        # self.conv5 = conv(ch_in + 416, 32)
        self.conv_last = conv(ch_in + 64, 2, isReLU=False)

    def forward(self, x):
        x1 = torch.cat([self.conv1(x), x], dim=1)
        x2 = torch.cat([self.conv2(x1), x1], dim=1)
        # x3 = torch.cat([self.conv3(x2), x2], dim=1)
        # x4 = torch.cat([self.conv4(x3), x3], dim=1)
        # x5 = torch.cat([self.conv5(x4), x4], dim=1)
        x_out = self.conv_last(x2)
        return x2, x_out

class FlowEstimatorDense(nn.Module):
    def __init__(self, ch_in=64, f_channels=(128, 128, 96, 64, 32, 32), ch_out=2):
        super(FlowEstimatorDense, self).__init__()
        N = 0
        ind = 0
        N += ch_in
        self.conv1 = conv(N, f_channels[ind])
        N += f_channels[ind]
        ind += 1
        self.conv2 = conv(N, f_channels[ind])
        N += f_channels[ind]
        ind += 1
        self.conv3 = conv(N, f_channels[ind])
        N += f_channels[ind]
        ind += 1
        self.conv4 = conv(N, f_channels[ind])
        N += f_channels[ind]
        ind += 1
        self.conv5 = conv(N, f_channels[ind])
        N += f_channels[ind]
        self.num_feature_channel = N
        ind += 1
        self.conv_last = conv(N, ch_out, isReLU=False)

    def forward(self, x):
        x1 = torch.cat([self.conv1(x), x], axis=1)
        x2 = torch.cat([self.conv2(x1), x1], axis=1)
        x3 = torch.cat([self.conv3(x2), x2], axis=1)
        x4 = torch.cat([self.conv4(x3), x3], axis=1)
        x5 = torch.cat([self.conv5(x4), x4], axis=1)
        x_out = self.conv_last(x5)
        return x5, x_out

class Tfeat_RefineBlock(nn.Module):
    def __init__(self, ch_in_frame, ch_in_event, ch_in_frame_prev, prev_scale=False):
        super(Tfeat_RefineBlock, self).__init__()
        if prev_scale:
            nf = int((ch_in_frame+ch_in_event+ch_in_frame_prev)/4)
        else:
            nf = int((ch_in_frame+ch_in_event)/4)

        self.conv_refine = nn.Sequential(conv1x1(4*nf, nf), nn.ReLU(), conv3x3(nf, ch_in_frame))

    def forward(self, x):
        x1 = self.conv_refine(x)
        return x1

def rescale_flow(flow, width_im, height_im):
    u_scale = float(width_im / flow.size(3))
    v_scale = float(height_im / flow.size(2))
    u, v = flow.chunk(2, dim=1)
    u = u_scale*u
    v = v_scale*v
    return torch.cat([u, v], dim=1)

#         if x.is_cuda:
#             grid = grid.cuda()
#         vgrid = Variable(grid) + flo

#         # scale grid to [-1,1]
#         vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
#         vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

#         if moments_across_images:
#             statistics['mean'] = ([torch.mean(F.stack(statistics['mean'], axis=0), axis=(0, ))] * len(feature_list))
#             statistics['var'] = ([torch.var(F.stack(statistics['var'], axis=0), axis=(0, ))] * len(feature_list))

#         statistics['std'] = [torch.sqrt(v + 1e-16) for v in statistics['var']]

#         # Center and normalize features.
#         if center:
#             feature_list = [f - mean for f, mean in zip(feature_list, statistics['mean'])]
#         if normalize:
#             feature_list = [f / std for f, std in zip(feature_list, statistics['std'])]
#         return feature_list

#         E_0t_pyramid = self.encoder_image_flow_event(batch['event_input_0t'])[::-1]

#         # encoder event
#         E_t0_pyramid_flow = self.encoder_event(batch['event_input_0t'])[::-1]

#         ### decoding optical flow
#         ## level 0
#         flow_t0_out_dict, flow_t0_dict = [], []

#             # flow rescale
#             down_flow_fusion_t0 = rescale_flow(upsample2d(flow_fusion_t0, F0), F0.size(3), F0.size(2))
#             # warping with optical flow
#             feat10 = self.warp(F0, self.flow_scale*down_flow_fusion_t0)
#             # feature normalization
#             feat_t_norm, feat10_norm = self.normalize_features([feat_t, feat10], normalize=True, center=True, moments_across_channels=False, moments_across_images=False)
#             # correlation
#             corr_t0 = self.leakyRELU(self.corr(feat_t_norm, feat10_norm))
#             # correlation refienement
#             _, res_flow_t0 = self.corr_refinement[level](torch.cat((corr_t0, feat_t, down_flow_fusion_t0), dim=1))
#             # frame-based optical flow generation
#             flow_t0_frame = down_flow_fusion_t0 + res_flow_t0
#             ## upsampling frame-based optical flow
#             upflow_t0_frame = rescale_flow(upsample2d(flow_t0_frame, flow_fusion_t0), flow_fusion_t0.size(3), flow_fusion_t0.size(2))
#             ### output
#             flow_t0_out_dict.append(upflow_t0_frame)
#         flow_t0_dict.append(self.flow_scale*upflow_t0_frame)
#         flow_t0_dict = flow_t0_dict[::-1]
#         ## final output return
#         if self.tb_debug:
#             return flow_t0_dict, event_flow_dict, fusion_flow_dict, image_flow_dict, mask_dict
#         else:
#             return flow_t0_dict

class FlowNet(nn.Module):
    def __init__(self, md=4, ev_ch=8, tb_debug=False, fast=True):
        super(FlowNet, self).__init__()
        # --- options ---
        self.tb_debug = tb_debug
        self.fast = fast
        self.flow_scale = 20

        # --- channel configs (smaller in fast mode) ---
        if self.fast:
            # narrower channels for speed
            num_chs_frame = [1, 4, 8, 12, 16]
            num_chs_event = [ev_ch, 4, 8, 12, 16]
            num_chs_event_image = [ev_ch, 4, 8, 12, 16]
            md = min(md, 2)  # shrink correlation radius (e.g., 4->2; 81ch->25ch)
            self.use_levels = 2          # use top-2 pyramid levels only
        else:
            num_chs_frame = [1, 4, 8, 16, 24]
            num_chs_event = [ev_ch, 4, 8, 16, 24]
            num_chs_event_image = [ev_ch, 4, 8, 16, 24]
            self.use_levels = 4          # use all 4 levels (as in original loop)

        # --- encoders/blocks (same API as original code) ---
        self.encoder_event = encoder_event_flow(num_chs_event)  # event-level flow
        self.encoder_image_flow = encoder_image_for_image_flow(num_chs_frame)  # image features
        self.encoder_image_flow_event = encoder_event_for_image_flow(num_chs_event_image)

        self.leakyRELU = nn.LeakyReLU(0.1)
        self.corr = Correlation(pad_size=md, kernel_size=1, max_displacement=md,
                                stride1=1, stride2=1, corr_multiply=1)
        nd = (2 * md + 1) ** 2

        self.corr_refinement = nn.ModuleList([
            DenseBlock(nd + num_chs_frame[-1] + 2),
            DenseBlock(nd + num_chs_frame[-2] + 2),
            DenseBlock(nd + num_chs_frame[-3] + 2),
            DenseBlock(nd + num_chs_frame[-4] + 2),
        ])

        self.decoder_event = nn.ModuleList([
            conv_resblock_one_small(num_chs_event[-1], num_chs_event[-1]),
            conv_resblock_one_small(num_chs_event[-2] + num_chs_event[-1] + 2, num_chs_event[-2]),
            conv_resblock_one_small(num_chs_event[-3] + num_chs_event[-2] + 2, num_chs_event[-3]),
        ])

        self.predict_flow = nn.ModuleList([
            conv3x3_leaky_relu(num_chs_event[-1], 2),
            conv3x3_leaky_relu(num_chs_event[-2], 2),
            conv3x3_leaky_relu(num_chs_event[-3], 2),
        ])

        self.conv_frame = nn.ModuleList([
            conv3x3_leaky_relu(num_chs_frame[-2], 16),
            conv3x3_leaky_relu(num_chs_frame[-3], 16),
        ])
        self.conv_frame_t = nn.ModuleList([
            conv3x3_leaky_relu(num_chs_frame[-2], 16),
            conv3x3_leaky_relu(num_chs_frame[-3], 16),
        ])

        self.flow_fusion_block = FlowEstimatorDense(16 * 3 + 4, (16, 16, 16, 8, 4), 1)

        self.feat_t_refinement = nn.ModuleList([
            Tfeat_RefineBlock(num_chs_frame[-1], num_chs_event_image[-1], None, prev_scale=False),
            Tfeat_RefineBlock(num_chs_frame[-2], num_chs_event_image[-2], num_chs_frame[-1], prev_scale=True),
            Tfeat_RefineBlock(num_chs_frame[-3], num_chs_event_image[-3], num_chs_frame[-2], prev_scale=True),
        ])

        # --- cached warp buffers (per-size) ---
        self.register_buffer("_grid_hw", torch.empty(0), persistent=False)   # [1,H,W,2] in [-1,1]
        self.register_buffer("_ones_mask", torch.empty(0), persistent=False) # [1,1,H,W]

    # ===== utils =====
    def _ensure_grid_and_mask(self, H, W, device):
        """
        Create (and cache) a base grid in [-1,1] for grid_sample and a ones mask.
        Works on old PyTorch versions that don't support meshgrid(indexing=...).
        """
        need_new = (self._grid_hw.numel() == 0) or (self._grid_hw.shape[1] != H) or (self._grid_hw.shape[2] != W)
        if need_new:
            ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=torch.float32)
            xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=torch.float32)
            # Old PyTorch doesn't support the 'indexing' kwarg; default is 'ij' in PyTorch.
            try:
                yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # PyTorch >= 1.10
            except TypeError:
                yy, xx = torch.meshgrid(ys, xs)                 # Older PyTorch

            grid = torch.stack([xx, yy], dim=-1)[None, ...]  # [1, H, W, 2], float32
            ones = torch.ones(1, 1, H, W, device=device, dtype=torch.float32)
            self._grid_hw = grid
            self._ones_mask = ones

        return self._grid_hw, self._ones_mask

    def warp(self, x, flo):
        """
        Warp x by flo (pixel units).
        x:   [B, C, H, W]
        flo: [B, 2, H, W]
        """
        B, C, H, W = x.shape
        device = x.device

        base_grid, ones_mask = self._ensure_grid_and_mask(H, W, device)  # float32
        # convert pixel offsets to [-1,1] offsets
        norm_dx = 2.0 * flo[:, 0:1] / max(W - 1, 1)
        norm_dy = 2.0 * flo[:, 1:2] / max(H - 1, 1)
        offsets = torch.stack(
            (norm_dx.squeeze(1), norm_dy.squeeze(1)),  # squeeze channel dim -> [B,H,W]
            dim=-1
        ).to(base_grid.dtype)

        # base_grid: [1, H, W, 2] (broadcast to B)
        vgrid = base_grid + offsets

        out = F.grid_sample(x, vgrid, mode='bilinear', padding_mode='zeros', align_corners=True)
        msk = F.grid_sample(ones_mask.expand(B, -1, H, W), vgrid, mode='nearest',
                            padding_mode='zeros', align_corners=True)
        msk = (msk > 0.9999).to(out.dtype)
        return out * msk

    def normalize_features(self, feature_list, normalize=True, center=True,
                           moments_across_channels=True, moments_across_images=False):
        """
        Vectorized, stable normalization (no per-image stacking unless requested).
        """
        dims = (1, 2, 3) if moments_across_channels else (2, 3)
        if center or normalize:
            means = [f.mean(dim=dims, keepdim=True) for f in feature_list]
            if center:
                feature_list = [f - m for f, m in zip(feature_list, means)]
            if normalize:
                stds = [(f.float().var(dim=dims, keepdim=True) + 1e-16).sqrt().to(f.dtype)
                        for f in feature_list]
                feature_list = [f / s for f, s in zip(feature_list, stds)]
        return feature_list

    # ===== forward =====
    def forward(self, batch):
        # Encoders: build pyramids and reverse for coarse->fine
        F0_pyramid = self.encoder_image_flow(batch['image_input0'])[::-1]
        E_0t_pyramid = self.encoder_image_flow_event(batch['event_input_0t'])[::-1]
        E_t0_pyramid_flow = self.encoder_event(batch['event_input_0t'])[::-1]

        flow_t0_out_dict, flow_t0_dict = [], []

        # iterate pyramid levels (coarse->fine), but limit by fast mode
        for level, (E_t0_flow, E_0t, F0) in enumerate(zip(E_t0_pyramid_flow, E_0t_pyramid, F0_pyramid)):
            if level >= self.use_levels:  # fast mode: early exit
                break

            if level == 0:
                # event flow (coarsest level)
                feat_t0_ev = self.decoder_event[level](E_t0_flow)
                flow_event_t0 = self.predict_flow[level](feat_t0_ev)

                # fusion flow (init with event flow)
                flow_fusion_t0 = flow_event_t0

                # t feature refinement
                feat_t_in = torch.cat((F0, E_0t), dim=1)
                feat_t = self.feat_t_refinement[level](feat_t_in)
            else:
                # refine t features with upsampled prev + current
                upfeat0_t = upsample2d(feat_t, F0)
                feat_t_in = torch.cat((upfeat0_t, F0, E_0t), dim=1)
                feat_t = self.feat_t_refinement[level](feat_t_in)

                # upsample previous fusion flow to current event feature res
                upflow_t0 = rescale_flow(upsample2d(flow_t0_out_dict[level - 1], E_t0_flow),
                                         E_t0_flow.size(3), E_t0_flow.size(2))

                # upsample previous event features/flows
                feat_t0_ev_up = upsample2d(feat_t0_ev, E_t0_flow)
                flow_t0_ev_up = rescale_flow(upsample2d(flow_event_t0, E_t0_flow),
                                             E_t0_flow.size(3), E_t0_flow.size(2))

                # decode event at this level
                feat_t0_ev = self.decoder_event[level](torch.cat((E_t0_flow, feat_t0_ev_up, flow_t0_ev_up), dim=1))
                flow_event_t0_ = self.predict_flow[level](feat_t0_ev)
                flow_event_t0 = flow_t0_ev_up + flow_event_t0_

                # downscale flows to frame feature res
                down_evflow_t0 = rescale_flow(upsample2d(flow_event_t0, F0), F0.size(3), F0.size(2))
                down_upflow_t0 = rescale_flow(upsample2d(flow_t0_out_dict[level - 1], F0), F0.size(3), F0.size(2))

                # fusion via warping and estimator
                F0_re = self.conv_frame[level - 1](F0)
                F0_up_warp_ev = self.warp(F0_re, self.flow_scale * down_evflow_t0)
                F0_up_warp_frame = self.warp(F0_re, self.flow_scale * down_upflow_t0)
                Ft_up = self.conv_frame_t[level - 1](feat_t)

                _, out_fusion_t0 = self.flow_fusion_block(
                    torch.cat((F0_up_warp_ev, F0_up_warp_frame, Ft_up, down_evflow_t0, down_upflow_t0), dim=1)
                )
                mask_t0 = upsample2d(torch.sigmoid(out_fusion_t0[:, -1, :, :])[:, None, :, :], E_t0_flow)
                flow_fusion_t0 = (1 - mask_t0) * upflow_t0 + mask_t0 * flow_event_t0

            # frame-based refinement with correlation
            down_flow_fusion_t0 = rescale_flow(upsample2d(flow_fusion_t0, F0), F0.size(3), F0.size(2))
            feat10 = self.warp(F0, self.flow_scale * down_flow_fusion_t0)

            # cheaper normalization (no cross-image moments)
            feat_t_norm, feat10_norm = self.normalize_features(
                [feat_t, feat10], normalize=True, center=True,
                moments_across_channels=False, moments_across_images=False
            )

            corr_t0 = self.leakyRELU(self.corr(feat_t_norm, feat10_norm))

            _, res_flow_t0 = self.corr_refinement[level](torch.cat((corr_t0, feat_t, down_flow_fusion_t0), dim=1))

            flow_t0_frame = down_flow_fusion_t0 + res_flow_t0

            # upsample to fusion resolution for next level usage
            upflow_t0_frame = rescale_flow(
                upsample2d(flow_t0_frame, flow_fusion_t0), flow_fusion_t0.size(3), flow_fusion_t0.size(2)
            )
            flow_t0_out_dict.append(upflow_t0_frame)

        # final output (same API as before: list with one flow)
        flow_t0_dict.append(self.flow_scale * upflow_t0_frame)
        flow_t0_dict = flow_t0_dict[::-1]

        if self.tb_debug:
            # NOTE: original snippet referenced event_flow_dict, etc. which were undefined.
            # To keep compatibility without extra bookkeeping, return primary output only in debug as well.
            return flow_t0_dict
        else:
            return flow_t0_dict

class frame_encoder(nn.Module):
    def __init__(self, in_dims, nf):
        super(frame_encoder, self).__init__()
        self.conv0 = conv3x3_leaky_relu(in_dims, nf)
        self.conv1 = conv_resblock_two_small(nf, nf)
        self.conv2 = conv_resblock_two_small(nf, 2*nf, stride=2)
        self.conv3 = conv_resblock_two_small(2*nf, 4*nf, stride=2)

    def forward(self, x):
        x_ = self.conv0(x)
        f1 = self.conv1(x_)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        return [f1, f2, f3]

class event_encoder(nn.Module):
    def __init__(self, in_dims, nf):
        super(event_encoder, self).__init__()
        self.conv0 = conv3x3_leaky_relu(in_dims, nf)
        self.conv1 = conv_resblock_two_small(nf, nf)
        self.conv2 = conv_resblock_two_small(nf, 2*nf, stride=2)
        self.conv3 = conv_resblock_two_small(2*nf, 4*nf, stride=2)

    def forward(self, x):
        x_ =  self.conv0(x)
        f1 = self.conv1(x_)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        return [f1, f2, f3]

##################################################
################# Restormer #####################

##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)

    def forward(self, x):
        out = self.deconv(x)
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*2*W*2*self.in_channel*self.out_channel*2*2
        print("Upsample:{%.2f}"%(flops/1e9))
        return flops

class Transformer(nn.Module):
    def __init__(self, unit_dim):
        super(Transformer, self).__init__()
        ## init qurey networks
        self.init_decoder(unit_dim)
        ## last conv
        self.last_conv0 = conv3x3(unit_dim*2, 1)
        self.last_conv1 = conv3x3(unit_dim*1, 1)
        self.last_conv2 = conv3x3(unit_dim, 1)

    def init_decoder(self, unit_dim):
        ### decoder
        ### attention k,v building (synthesis)
        self.build_kv0_syn = conv3x3_leaky_relu(unit_dim*3, unit_dim*2)
        self.build_kv1_syn = conv3x3_leaky_relu(int(unit_dim*1.5), unit_dim)
        self.build_kv2_syn = conv3x3_leaky_relu(int(unit_dim*0.75), unit_dim//2)
        ### attention k, v building (warping)
        self.build_kv0_warp = conv3x3_leaky_relu(unit_dim*3+1, unit_dim*2)
        self.build_kv1_warp = conv3x3_leaky_relu(int(unit_dim*1.5)+1, unit_dim)
        self.build_kv2_warp = conv3x3_leaky_relu(int(unit_dim*0.75)+1, unit_dim//2)
        ## level 1
        self.decoder1_1 = conv1x1(4*unit_dim, 2*unit_dim)

        ## level 2
        self.decoder2_1 = conv1x1(3*unit_dim, 1*unit_dim)

        ## level 3
        self.decoder3_1 = conv1x1(2*unit_dim, unit_dim)

        ## upsample
        self.upsample0 = Upsample(unit_dim*2, unit_dim*1)
        self.upsample1 = Upsample(unit_dim*1, unit_dim)

    def forward_decoder(self, warped_feature, frame_feature, event_feature):
        ## syntheis kv building
        cat_in0_syn = torch.cat((frame_feature[2], event_feature[2]), dim=1)
        attn_kv0_syn = self.build_kv0_syn(cat_in0_syn)
        cat_in1_syn = torch.cat((frame_feature[1], event_feature[1]), dim=1)
        attn_kv1_syn = self.build_kv1_syn(cat_in1_syn)
        cat_in2_syn = torch.cat((frame_feature[0], event_feature[0]), dim=1)
        attn_kv2_syn = self.build_kv2_syn(cat_in2_syn)
        ## warping kv building
        cat_in0_warp = torch.cat((warped_feature[2], event_feature[2]), dim=1)
        attn_kv0_warp = self.build_kv0_warp(cat_in0_warp)
        cat_in1_warp = torch.cat((warped_feature[1], event_feature[1]), dim=1)
        attn_kv1_warp = self.build_kv1_warp(cat_in1_warp)
        cat_in2_warp = torch.cat((warped_feature[0], event_feature[0]), dim=1)
        attn_kv2_warp = self.build_kv2_warp(cat_in2_warp)

        ## out 0
        out0 = self.decoder1_1(torch.cat([attn_kv0_syn, attn_kv0_warp], dim=1))
        up_out0 = self.upsample0(out0)
        ## out 1
        out1 = self.decoder2_1(torch.cat([up_out0, attn_kv1_syn, attn_kv1_warp], dim=1))
        up_out1 = self.upsample1(out1)
        ## out2
        out2 = self.decoder3_1(torch.cat([up_out1, attn_kv2_syn, attn_kv2_warp], dim=1))
        return [out0, out1, out2]

    def forward(self, event_feature, frame_feature, warped_feature):
        ### forward decoder
        out_decoder = self.forward_decoder(warped_feature, frame_feature, event_feature)
        ### synthesis frame
        img0 = self.last_conv0(out_decoder[0])
        img1 = self.last_conv1(out_decoder[1])
        img2 = self.last_conv2(out_decoder[2])
        return [img2, img1, img0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backwarp_tenGrid = {}
@register('cbmnet_light_extrapolation_small_e2vid')
class EventInterpNet(nn.Module):
    def __init__(self, encoder_spec, imnet_spec=None, in_ch=1, num_bins=5, flow_debug=False):
        super(EventInterpNet, self).__init__()
        unit_dim = 16
        # scale
        self.scale = 3
        # flownet
        self.flownet = FlowNet(md=4, ev_ch=num_bins, tb_debug=flow_debug)
        self.flow_debug = flow_debug
        # encoder
        self.encoder_f = frame_encoder(in_ch, unit_dim//4)
        self.encoder_e = event_encoder(num_bins, unit_dim//2)
        # decoder
        self.transformer = Transformer(unit_dim)
        # channel scaling convolution
        self.conv_list = nn.ModuleList([conv1x1(unit_dim, unit_dim), conv1x1(unit_dim, unit_dim), conv1x1(unit_dim, unit_dim)])

        self.fuse_conv0 = conv3x3(40, 8)
        self.fuse_conv1 = conv3x3(80, 16)
        self.fuse_conv2 = conv3x3(160, 32)
        self.fuse_list = [self.fuse_conv0, self.fuse_conv1, self.fuse_conv2]

        # --- cached warp buffers (per-size) ---
        self.register_buffer("_grid_hw", torch.empty(0), persistent=False)   # [1,H,W,2] in [-1,1]
        self.register_buffer("_ones_mask", torch.empty(0), persistent=False) # [1,1,H,W]

    def bwarp(self, x, flo):
        """
        Backward warp: x를 flo(픽셀 단위)으로 워핑.
        x:   [B, C, H, W]
        flo: [B, 2, h, w]  (h,w가 H,W와 다를 수 있음)
        """
        import torch.nn.functional as F
        B, C, H, W = x.shape
        _, _, h, w = flo.shape
        device = x.device

        # 1) flow 해상도가 다르면 x 해상도로 업샘플 + 리스케일
        if (h != H) or (w != W):
            # upsample2d는 보통 feature 해상도에 맞춰 보간만 하므로,
            # rescale_flow로 벡터 크기까지 (w,h → W,H) 보정합니다.
            flo = rescale_flow(upsample2d(flo, x), W, H)

        # 2) 그리드 캐시 확보 ([-1,1] 정규화 좌표)
        base_grid, _ = self._ensure_grid_and_mask(H, W, device)  # float32 [1,H,W,2]

        # 3) flow(픽셀)를 [-1,1] 오프셋으로 변환
        norm_dx = 2.0 * flo[:, 0:1] / max(W - 1, 1)
        norm_dy = 2.0 * flo[:, 1:2] / max(H - 1, 1)
        # offsets: [B, H, W, 2]
        offsets = torch.stack(
            (norm_dx.squeeze(1), norm_dy.squeeze(1)), dim=-1
        ).to(base_grid.dtype)

        # 4) 최종 grid
        vgrid = base_grid + offsets  # [B,H,W,2] (base_grid는 [1,H,W,2]라 브로드캐스트)

        # 5) 샘플링 (align_corners=True는 원래 코드와 일치)
        out = F.grid_sample(x, vgrid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return out

    # ===== utils =====
    def _ensure_grid_and_mask(self, H, W, device):
        """
        Create (and cache) a base grid in [-1,1] for grid_sample and a ones mask.
        Works on old PyTorch versions that don't support meshgrid(indexing=...).
        """
        need_new = (self._grid_hw.numel() == 0) or (self._grid_hw.shape[1] != H) or (self._grid_hw.shape[2] != W)
        if need_new:
            ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=torch.float32)
            xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=torch.float32)
            # Old PyTorch doesn't support the 'indexing' kwarg; default is 'ij' in PyTorch.
            try:
                yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # PyTorch >= 1.10
            except TypeError:
                yy, xx = torch.meshgrid(ys, xs)                 # Older PyTorch

            grid = torch.stack([xx, yy], dim=-1)[None, ...]  # [1, H, W, 2], float32
            ones = torch.ones(1, 1, H, W, device=device, dtype=torch.float32)
            self._grid_hw = grid
            self._ones_mask = ones

        return self._grid_hw, self._ones_mask

    def Flow_pyramid(self, flow):
        flow_pyr = []
        flow_pyr.append(flow)
        for i in range(1, 3):
            flow_pyr.append(F.interpolate(flow, scale_factor=0.5 ** i, mode='bilinear') * (0.5 ** i))
        return flow_pyr

    def Img_pyramid(self, Img):
        img_pyr = []
        img_pyr.append(Img)
        for i in range(1, 3):
            img_pyr.append(F.interpolate(Img, scale_factor=0.5 ** i, mode='bilinear'))
        return img_pyr

    def synthesis(self, batch, OF_t0, latent):
        ## frame encoding
        f_frame0 = self.encoder_f(batch['image_input0'])
        ## OF pyramid
        OF_t0_pyramid = self.Flow_pyramid(OF_t0[0])
        ## image pyramid
        I0_pyramid = self.Img_pyramid(batch['image_input0'])
        # frame0_warped, frame1_warped = [], []
        warped_feature, frame_feature = [], []
        for idx in range(self.scale):
            frame0_warped = self.bwarp(torch.cat((f_frame0[idx], I0_pyramid[idx]),dim=1), OF_t0_pyramid[idx])
            warped_feature.append(frame0_warped)
            frame_feature.append(f_frame0[idx])
        # after_tmp_feature = self.conv_list[idx](tmp_feature)
        event_feature = []
        # event encoding for frame interpolation
        f_event_0t = self.encoder_e(batch['event_input_0t'])

        for idx in range(self.scale):
            event_feature.append(self.fuse_list[idx](torch.cat([f_event_0t[idx], latent[idx].detach()], dim=1)))

        img_out = self.transformer(event_feature, frame_feature, warped_feature)
        output_clean = []
        for i in range(self.scale):
            # output_clean.append(torch.clamp(img_out[i], 0, 1))
            output_clean.append(img_out[i])
        return output_clean

    def forward(self, x, event, latent, mode='joint'):
        batch = {}

        I0 = x
        E0 = event

        #     for i, lat in enumerate(latent):
        #         lat = paddingInput(lat)
        #         latent[i] = lat

        batch['image_input0'] = I0
        batch['event_input_0t'] = E0

        OF_t0 = self.flownet(batch)

        output_clean = self.synthesis(batch, OF_t0, latent)

        output_clean = output_clean[0]
        of_t0 = OF_t0[0]

        warped_I0 = self.backwarp(x, of_t0)

        return output_clean, of_t0
        # return output_clean[:, 0].unsqueeze(1), warped_I0[:, 0].unsqueeze(1)

    def backwarp(self, tenInput, tenFlow):
        k = (str(tenFlow.device), str(tenFlow.size()))
        if k not in backwarp_tenGrid: ## youngho
            tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
                1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
            tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device).view(
                1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
            backwarp_tenGrid[k] = torch.cat(
                [tenHorizontal, tenVertical], 1).to(device)

        tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                            tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

        g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
        return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)

if __name__ =="__main__":
    model = EventInterpNet(encoder_spec=None, imnet_spec=None, in_ch=1, num_bins=8, flow_debug=False).cuda()
    x_dummy = torch.randn((1, 1, 160, 160)).cuda()
    event_dummy = torch.randn((1, 8, 160, 160)).cuda()

    test_n = 300
    elapsed_list = []
    with torch.no_grad():
        for i in range(50+test_n):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(x_dummy, event_dummy)
            end.record()
            torch.cuda.synchronize()
            e_time = start.elapsed_time(end)
            if i>=50:
                elapsed_list.append(e_time)

    print(f"elapse: {sum(elapsed_list)/test_n}m seconds")

