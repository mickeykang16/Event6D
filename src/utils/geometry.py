import os
import cv2
import sys
import time
import json
import math
import scipy
import random
import trimesh
import itertools
import pickle
import kornia
import zipfile
import datetime
import imageio, gzip, logging
import joblib, importlib

from tqdm import tqdm
import math, glob, re, copy
from typing import Tuple, List, Optional, Dict

from PIL import Image
import open3d as o3d
import numpy as np
import pandas as pd
from functools import partial
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from sklearn.decomposition import TruncatedSVD
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from pytorch3d.transforms import so3_log_map, so3_exp_map, se3_exp_map, se3_log_map, matrix_to_axis_angle, \
    matrix_to_euler_angles, euler_angles_to_matrix, rotation_6d_to_matrix
from pytorch3d.renderer import FoVPerspectiveCameras, PerspectiveCameras, look_at_view_transform, look_at_rotation, \
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams, SoftSilhouetteShader, HardPhongShader, \
    PointLights, TexturesVertex
from pytorch3d.renderer.mesh.rasterize_meshes import barycentric_coordinates
from pytorch3d.renderer.mesh.shader import SoftDepthShader, HardFlatShader
from pytorch3d.renderer.mesh.textures import Textures
from pytorch3d.structures import Meshes

import kaolin
import warp as wp
import nvdiffrast.torch as dr
from transformations import *

import mycpp.build.mycpp as mycpp
try:
  from bundlesdf.mycuda import common
except:
  common = None

# import utils.mycpp.build.mycpp as mycpp
# from utils.bundlesdf.mycuda import common

glcam_in_cvcam = np.array([[1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, -1, 0],
                           [0, 0, 0, 1]]).astype(float)

COLOR_MAP = np.array([[0, 0, 0],  # Ignore
                      [128, 0, 0],  # Background
                      [0, 128, 0],  # Wall
                      [128, 128, 0],  # Floor
                      [0, 0, 128],  # Ceiling
                      [128, 0, 128],  # Table
                      [0, 128, 128],  # Chair
                      [128, 128, 128],  # Window
                      [64, 0, 0],  # Door
                      [192, 0, 0],  # Monitor
                      [64, 128, 0],  # 11th
                      [192, 0, 128],
                      [64, 128, 128],
                      [192, 128, 128],
                      [0, 64, 0],
                      [128, 64, 0],
                      [0, 192, 0],
                      [128, 192, 0],
                      ])


def set_logging_format(level=logging.INFO):
    importlib.reload(logging)
    FORMAT = '[%(funcName)s()] %(message)s'
    logging.basicConfig(level=level, format=FORMAT)


set_logging_format()


@torch.no_grad()
def sample_random_poses_fix_ratio(
    base_T: torch.Tensor,   # (B,4,4)
    K: int,
    max_trans: float,
    max_rot_deg: float,
    thresh_trans: float,
    thresh_rot_deg: float,
    p_within: float = 0.10, # 근접 샘플 비율 (~10%)
):
    """
    Return: (B, K, 4, 4)
      각 샘플은 base_T[b] @ ΔT 형태.
      근접(≈10%): ||Δt|| ≤ thresh_trans, angle(ΔR) ≤ thresh_rot_deg
      원거리(≈90%): thresh < (trans, rot) ≤ max 범위에서 균등
    """
    assert base_T.ndim == 3 and base_T.shape[-2:] == (4,4)
    assert 0 <= thresh_trans <= max_trans
    assert 0 <= thresh_rot_deg <= max_rot_deg
    assert 0.0 < p_within < 1.0

    B = base_T.shape[0]
    device = base_T.device
    dtype  = base_T.dtype

    # 개수 배분 (근접 ≈10%)
    n_within = max(1, int(round(K * p_within)))
    n_far    = K - n_within

    def rand_unit(n):
        v = torch.randn(n, 3, device=device, dtype=dtype)
        return v / (v.norm(dim=-1, keepdim=True) + 1e-12)

    def axis_angle_to_R(axis, angle):  # angle: (...,1) radians
        ax = axis
        th = angle[..., 0]
        Kmat = torch.zeros((*ax.shape[:-1], 3, 3), device=device, dtype=dtype)
        Kmat[..., 0, 1] = -ax[..., 2]; Kmat[..., 0, 2] =  ax[..., 1]
        Kmat[..., 1, 0] =  ax[..., 2]; Kmat[..., 1, 2] = -ax[..., 0]
        Kmat[..., 2, 0] = -ax[..., 1]; Kmat[..., 2, 1] =  ax[..., 0]
        I = torch.eye(3, device=device, dtype=dtype).expand(Kmat.shape)
        sin_t = torch.sin(th).unsqueeze(-1).unsqueeze(-1)
        cos_t = torch.cos(th).unsqueeze(-1).unsqueeze(-1)
        return I + sin_t * Kmat + (1 - cos_t) * (Kmat @ Kmat)

    def uniform_radius_ball(n, r_max):
        u = torch.rand(n, device=device, dtype=dtype)
        return (u ** (1/3)) * r_max

    def uniform_radius_shell(n, r_min, r_max):
        u = torch.rand(n, device=device, dtype=dtype)
        r3 = u * (r_max**3 - r_min**3) + r_min**3
        return r3 ** (1/3)

    # ---- 근접(Within) Δt, ΔR 생성 ----
    Nw = B * n_within
    if Nw > 0:
        dir_w = rand_unit(Nw)
        rad_w = uniform_radius_ball(Nw, thresh_trans)
        dt_w  = dir_w * rad_w.unsqueeze(-1)

        axes_w = rand_unit(Nw)
        ang_w  = torch.rand(Nw, 1, device=device, dtype=dtype) * math.radians(thresh_rot_deg)
        dR_w   = axis_angle_to_R(axes_w, ang_w)

        dT_w = torch.zeros(Nw, 4, 4, device=device, dtype=dtype)
        dT_w[:, :3, :3] = dR_w
        dT_w[:, :3, 3]  = dt_w
        dT_w[:, 3, 3]   = 1.0
    else:
        dT_w = torch.empty(0, 4, 4, device=device, dtype=dtype)

    # ---- 원거리(Far) Δt, ΔR 생성 ----
    Nf = B * n_far
    if Nf > 0:
        dir_f = rand_unit(Nf)
        rad_f = uniform_radius_shell(Nf, thresh_trans, max_trans)
        dt_f  = dir_f * rad_f.unsqueeze(-1)

        axes_f = rand_unit(Nf)
        ang_f  = torch.rand(Nf, 1, device=device, dtype=dtype) * (
            math.radians(max_rot_deg) - math.radians(thresh_rot_deg)
        ) + math.radians(thresh_rot_deg)
        dR_f   = axis_angle_to_R(axes_f, ang_f)

        dT_f = torch.zeros(Nf, 4, 4, device=device, dtype=dtype)
        dT_f[:, :3, :3] = dR_f
        dT_f[:, :3, 3]  = dt_f
        dT_f[:, 3, 3]   = 1.0
    else:
        dT_f = torch.empty(0, 4, 4, device=device, dtype=dtype)

    # ---- 배치별로 섞고 적용 ----
    out = torch.zeros(B, K, 4, 4, device=device, dtype=dtype)
    for b in range(B):
        parts = []
        if n_within > 0:
            parts.append(dT_w[b*n_within:(b+1)*n_within])
        if n_far > 0:
            parts.append(dT_f[b*n_far:(b+1)*n_far])
        dTs = torch.cat(parts, dim=0) if parts else torch.empty(0,4,4,device=device,dtype=dtype)

        if K > 1:
            perm = torch.randperm(K, device=device)
            dTs = dTs[perm]

        out[b] = base_T[b].unsqueeze(0) @ dTs

    return out.reshape(B*K, 4, 4)  # (B,K,4,4)

def random_rotation_matrices(B, max_angle_deg, mode: str = "uniform", sigma: float = 5.0):
    """
    B개의 랜덤 회전 행렬을 생성 (axis-angle 기반).
    
    Args:
        B: 생성할 회전 행렬 개수
        max_angle_deg: 최대 회전 각도 (deg), uniform 모드에서만 사용
        mode: "uniform" (기존 방식) 또는 "gaussian" (정규분포 샘플링)
        sigma: gaussian 모드에서 표준편차 (deg 단위)
    
    Returns:
        (B, 3, 3) 회전 행렬 텐서
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- 회전 각도 샘플링 ---
    if mode == "uniform":
        max_angle_rad = math.radians(max_angle_deg)
        angles = torch.rand(B, device=device) * max_angle_rad
    elif mode == "gaussian":
        # 평균 0, 표준편차 sigma에서 샘플 (라디안 단위 변환)
        angles = torch.randn(B, device=device) * math.radians(sigma)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'uniform' or 'gaussian'.")

    # --- 랜덤 회전 축 (단위 벡터) ---
    axes = torch.randn(B, 3, device=device)
    axes = axes / axes.norm(dim=1, keepdim=True)

    # --- skew-symmetric 행렬 ---
    K = torch.zeros(B, 3, 3, device=device)
    K[:, 0, 1] = -axes[:, 2]
    K[:, 0, 2] =  axes[:, 1]
    K[:, 1, 0] =  axes[:, 2]
    K[:, 1, 2] = -axes[:, 0]
    K[:, 2, 0] = -axes[:, 1]
    K[:, 2, 1] =  axes[:, 0]

    # --- Rodrigues' formula ---
    I = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)
    sin_angles = torch.sin(angles).view(B, 1, 1)
    cos_angles = torch.cos(angles).view(B, 1, 1)

    K2 = torch.bmm(K, K)
    R = I + sin_angles * K + (1 - cos_angles) * K2
    return R

@torch.no_grad()
@torch.cuda.amp.autocast()
def recover_aug_inverse(object_pose, object_pose_aug, eps: float = 1e-6):
    """
    Given original pose (object_pose) and augmented pose (object_pose_aug),
    recover delta_R_inv and delta_t_inv that match the definition used in augment_pose:
      R_aug = delta_R @ R
      t_aug = t + delta_t
      return delta_t_inv = -delta_t, delta_R_inv = delta_R^{-1}

    Args:
        object_pose:     (B, 4, 4) or (B, 3/4, 4) pose matrices
        object_pose_aug: (B, 4, 4) or (B, 3/4, 4) pose matrices
        eps: small number for optional orthonormalization guard

    Returns:
        delta_t_inv: (B, 3)
        delta_R_inv: (B, 3, 3)
    """
    assert object_pose.shape[0] == object_pose_aug.shape[0], "Batch size mismatch"
    B = object_pose.shape[0]
    device = object_pose.device

    # Extract R, t
    R     = object_pose[:, :3, :3]         # (B,3,3)
    t     = object_pose[:, :3, 3]          # (B,3)
    R_aug = object_pose_aug[:, :3, :3]     # (B,3,3)
    t_aug = object_pose_aug[:, :3, 3]      # (B,3)

    # Recover delta_R and delta_t
    # For rotation matrices: R^{-1} = R^T
    delta_R = R_aug @ R.transpose(-2, -1)  # (B,3,3)
    delta_t = t_aug - t                    # (B,3)

    # Inverse consistent with your augment_pose return
    delta_R_inv = delta_R.transpose(-2, -1)  # == inv(delta_R) for rotation matrix
    delta_t_inv = -delta_t

    return object_pose_aug, delta_t_inv, delta_R_inv

@torch.no_grad()
@torch.cuda.amp.autocast()
def augment_pose(object_pose, mesh_diameter=0.05, rot_range_deg=20):
    B = object_pose.shape[0]
    device = object_pose.device

    # 1. Translation noise: uniform [-0.05, 0.05]
    delta_t = torch.empty(B, 3, device=device).uniform_(-mesh_diameter / 2, mesh_diameter / 2)

    # Limit max rotation angle (optional): project to angle < 20 deg
    max_angle_rad = math.radians(rot_range_deg)
    cos_max = math.cos(max_angle_rad)

    # New random rotation formula (JY)
    delta_R = random_rotation_matrices(B, rot_range_deg).to(device)
    
    # 3. Apply augmentation to object_pose
    object_pose_aug = object_pose.clone()
    R = object_pose[:, :3, :3]  # (B, 3, 3)
    t = object_pose[:, :3, 3]  # (B, 3)

    R_aug = delta_R @ R  # (B, 3, 3)
    t_aug = t + delta_t  # (B, 3)

    object_pose_aug[:, :3, :3] = R_aug
    object_pose_aug[:, :3, 3] = t_aug

    delta_R_inv = torch.linalg.inv(delta_R)  # or delta_R.transpose(-2, -1) if rotation matrix
    delta_t_inv = -delta_t

    return object_pose_aug, delta_t_inv, delta_R_inv

@torch.no_grad()
@torch.cuda.amp.autocast()
def augment_pose_scorer(gt_poses: torch.Tensor,
                 K: int = 64,
                 max_angle_deg: float = 20.0,
                 trans_noise_scale: float = 0.05,
                 rot_noise_sigma: float = 5.0) -> torch.Tensor:
    """
    GT pose 주변에서 K개 랜덤 perturbation 생성 (Gaussian translation, bounded rotation).

    Args:
        gt_poses: (B,4,4) homogeneous GT pose
        K: 후보 개수
        max_angle_deg: 회전 perturbation 최대 각도 (deg)
        trans_noise_scale: translation noise 표준편차 (Gaussian N(0, σ^2))

    Returns:
        (B*K, 4, 4) perturbation된 포즈
    """
    device = gt_poses.device
    B = gt_poses.shape[0]

    # GT 확장
    gt_exp = gt_poses.unsqueeze(1).expand(B, K, 4, 4).contiguous()

    # Rotation noise
    R_delta = random_rotation_matrices(B * K, max_angle_deg, mode="gaussian", sigma=rot_noise_sigma).to(device)

    # Translation noise (Gaussian)
    t_noise = torch.randn(B * K, 3, device=device) * trans_noise_scale

    # Apply noise
    R_gt = gt_exp[..., :3, :3].reshape(B * K, 3, 3)
    t_gt = gt_exp[..., :3, 3].reshape(B * K, 3)

    R_out = R_delta @ R_gt
    t_out = t_gt + t_noise

    # Homogeneous pose 구성
    out = torch.eye(4, device=device).expand(B * K, 4, 4).clone()
    out[:, :3, :3] = R_out
    out[:, :3, 3] = t_out
    return out

def is_valid_rotation_matrices(R):
    # 1. 회전 행렬이 정규 직교 행렬인지 확인: R^T * R = I (배치 처리)
    RtR = torch.bmm(R.transpose(1, 2), R)  # R^T * R (배치 행렬 곱셈)
    identity_matrix = torch.eye(3, device=R.device).expand_as(RtR[0])  # 단위 행렬

    # R^T * R이 단위 행렬에 가까운지 확인 (수치적 오차 고려)
    if not torch.allclose(RtR, identity_matrix, atol=1e-6):
        return False

    # 2. 행렬의 결정식이 +1인지 확인 (배치 처리)
    det_R = torch.det(R)  # 각 회전 행렬에 대해 결정식 계산
    if not torch.allclose(det_R, torch.ones_like(det_R), atol=1e-6):
        return False

    return True


def normalize_vector(v):
    return torch.nn.functional.normalize(v, dim=-1, eps=1e-8)

def cross_product(u, v):
    return torch.cross(u, v, dim=-1)

def Ortho6d2Mat(x_raw, y_raw):
    y = normalize_vector(y_raw)
    z = cross_product(x_raw, y)
    z = normalize_vector(z)
    x = cross_product(y, z)

    x = x.unsqueeze(2)
    y = y.unsqueeze(2)
    z = z.unsqueeze(2)
    matrix = torch.cat((x, y, z), dim=2)  # (B, 3, 3)
    return matrix

def make_mesh_tensors(mesh, max_tex_size=None, device='cuda:0'):
    mesh_tensors = {}
    if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        img = np.array(mesh.visual.material.image.convert('RGB'))
        img = img[..., :3]
        if max_tex_size is not None:
            max_size = max(img.shape[0], img.shape[1])
            if max_size > max_tex_size:
                scale = 1 / max_size * max_tex_size
                img = cv2.resize(img, fx=scale, fy=scale, dsize=None)
        mesh_tensors['tex'] = torch.as_tensor(img, device=device, dtype=torch.float)[None] / 255.0
        mesh_tensors['uv_idx'] = torch.as_tensor(mesh.faces, device=device, dtype=torch.int)
        uv = torch.as_tensor(mesh.visual.uv, device=device, dtype=torch.float)
        uv[:, 1] = 1 - uv[:, 1]
        mesh_tensors['uv'] = uv
    else:
        if mesh.visual.vertex_colors is None:
            logging.info(f"WARN: mesh doesn't have vertex_colors, assigning a pure color")
            mesh.visual.vertex_colors = np.tile(np.array([128, 128, 128]).reshape(1, 3), (len(mesh.vertices), 1))
        mesh_tensors['vertex_color'] = torch.as_tensor(mesh.visual.vertex_colors[..., :3], device=device,
                                                       dtype=torch.float) / 255.0

    mesh_tensors.update({
        'pos': torch.tensor(mesh.vertices, device=device, dtype=torch.float),
        'faces': torch.tensor(mesh.faces, device=device, dtype=torch.int),
        'vnormals': torch.tensor(mesh.vertex_normals, device=device, dtype=torch.float),
        })
    return mesh_tensors


def nvdiffrast_render(K=None, H=None, W=None, ob_in_cams=None, glctx=None, context='cuda', device='cuda:0',
                      get_normal=False, mesh_tensors=None, mesh=None, projection_mat=None, bbox2d=None,
                      output_size=None, use_light=False, light_color=None, light_dir=np.array([0, 0, 1]),
                      light_pos=np.array([0, 0, 0]), w_ambient=0.8, w_diffuse=0.5, extra={}):
    '''Just plain rendering, not support any gradient
    @K: (3,3) np array
    @ob_in_cams: (N,4,4) torch tensor, openCV camera
    @projection_mat: np array (4,4)
    @output_size: (height, width)
    @bbox2d: (N,4) (umin,vmin,umax,vmax) if only roi need to render.
    @light_dir: in cam space
    @light_pos: in cam space
    '''
    if glctx is None:
        if context == 'gl':
            glctx = dr.RasterizeGLContext()
        elif context == 'cuda':
            glctx = dr.RasterizeCudaContext()
        else:
            raise NotImplementedError

    if mesh_tensors is None:
        mesh_tensors = make_mesh_tensors(mesh)
    pos = mesh_tensors['pos']
    vnormals = mesh_tensors['vnormals']
    pos_idx = mesh_tensors['faces']
    has_tex = 'tex' in mesh_tensors

    ob_in_glcams = torch.tensor(glcam_in_cvcam, device=device, dtype=torch.float)[None] @ ob_in_cams
    if projection_mat is None:
        projection_mat = projection_matrix_from_intrinsics(K, height=H, width=W, znear=0.001, zfar=100)
    projection_mat = torch.as_tensor(projection_mat.reshape(-1, 4, 4), device=device, dtype=torch.float)
    mtx = projection_mat @ ob_in_glcams

    if output_size is None:
        output_size = np.asarray([H, W])

    pts_cam = transform_pts(pos, ob_in_cams)
    pos_homo = to_homo_torch(pos)
    pos_clip = (mtx[:, None] @ pos_homo[None, ..., None])[..., 0]
    if bbox2d is not None:
        l = bbox2d[:, 0]
        t = H - bbox2d[:, 1]
        r = bbox2d[:, 2]
        b = H - bbox2d[:, 3]
        tf = torch.eye(4, dtype=torch.float, device=device).reshape(1, 4, 4).expand(len(ob_in_cams), 4, 4).contiguous()
        tf[:, 0, 0] = W / (r - l)
        tf[:, 1, 1] = H / (t - b)
        tf[:, 3, 0] = (W - r - l) / (r - l)
        tf[:, 3, 1] = (H - t - b) / (t - b)
        pos_clip = pos_clip @ tf
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=np.asarray(output_size))
    xyz_map, _ = dr.interpolate(pts_cam, rast_out, pos_idx)
    depth = xyz_map[..., 2]
    if has_tex:
        texc, _ = dr.interpolate(mesh_tensors['uv'], rast_out, mesh_tensors['uv_idx'])
        color = dr.texture(mesh_tensors['tex'], texc, filter_mode='linear')
    else:
        color, _ = dr.interpolate(mesh_tensors['vertex_color'], rast_out, pos_idx)

    if use_light:
        get_normal = True
    if get_normal:
        vnormals_cam = transform_dirs(vnormals, ob_in_cams)
        normal_map, _ = dr.interpolate(vnormals_cam, rast_out, pos_idx)
        normal_map = F.normalize(normal_map, dim=-1)
        normal_map = torch.flip(normal_map, dims=[1])
    else:
        normal_map = None

    if use_light:
        if light_dir is not None:
            light_dir_neg = -torch.as_tensor(light_dir, dtype=torch.float, device=device)
        else:
            light_dir_neg = torch.as_tensor(light_pos, dtype=torch.float, device=device).reshape(1, 1, 3) - pts_cam
        diffuse_intensity = \
        (F.normalize(vnormals_cam, dim=-1) * F.normalize(light_dir_neg, dim=-1)).sum(dim=-1).clip(0, 1)[..., None]
        diffuse_intensity_map, _ = dr.interpolate(diffuse_intensity, rast_out, pos_idx)  # (N_pose, H, W, 1)
        if light_color is None:
            light_color = color
        else:
            light_color = torch.as_tensor(light_color, device=device, dtype=torch.float)
        color = color * w_ambient + diffuse_intensity_map * light_color * w_diffuse

    color = color.clip(0, 1)
    color = color * torch.clamp(rast_out[..., -1:], 0, 1)  # Mask out background using alpha
    color = torch.flip(color, dims=[1])  # Flip Y coordinates
    depth = torch.flip(depth, dims=[1])
    extra['xyz_map'] = torch.flip(xyz_map, dims=[1])
    return color, depth, normal_map


def toOpen3dCloud(points, colors=None, normals=None):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        if colors.max() > 1:
            colors = colors / 255.0
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    return cloud


def make_grid_image(imgs, nrow, padding=5, pad_value=255):
    '''
    @imgs: (B,H,W,C) np array
    @nrow: num of images per row
    '''
    grid = torchvision.utils.make_grid(torch.as_tensor(np.asarray(imgs)).permute(0, 3, 1, 2), nrow=nrow,
                                       padding=padding, pad_value=pad_value)
    grid = grid.permute(1, 2, 0).contiguous().data.cpu().numpy().astype(np.uint8)
    return grid


@wp.kernel(enable_backward=False)
def bilateral_filter_depth_kernel(depth: wp.array(dtype=float, ndim=2), out: wp.array(dtype=float, ndim=2), radius: int,
                                  zfar: float, sigmaD: float, sigmaR: float):
    h, w = wp.tid()
    H = depth.shape[0]
    W = depth.shape[1]
    if w >= W or h >= H:
        return
    out[h, w] = 0.0
    mean_depth = float(0.0)
    num_valid = int(0)
    for u in range(w - radius, w + radius + 1):
        if u < 0 or u >= W:
            continue
        for v in range(h - radius, h + radius + 1):
            if v < 0 or v >= H:
                continue
            cur_depth = depth[v, u]
            if cur_depth >= 0.001 and cur_depth < zfar:
                num_valid += 1
                mean_depth += cur_depth
    if num_valid == 0:
        return
    mean_depth /= float(num_valid)

    depthCenter = depth[h, w]
    sum_weight = float(0.0)
    sum = float(0.0)
    for u in range(w - radius, w + radius + 1):
        if u < 0 or u >= W:
            continue
        for v in range(h - radius, h + radius + 1):
            if v < 0 or v >= H:
                continue
            cur_depth = depth[v, u]
            if cur_depth >= 0.001 and cur_depth < zfar and abs(cur_depth - mean_depth) < 0.01:
                weight = wp.exp(-float((u - w) * (u - w) + (h - v) * (h - v)) / (2.0 * sigmaD * sigmaD) - (
                            depthCenter - cur_depth) * (depthCenter - cur_depth) / (2.0 * sigmaR * sigmaR))
                sum_weight += weight
                sum += weight * cur_depth
    if sum_weight > 0 and num_valid > 0:
        out[h, w] = sum / sum_weight


def bilateral_filter_depth(depth, radius=2, zfar=100, sigmaD=2, sigmaR=100000, device='cuda'):
    if isinstance(depth, np.ndarray):
        depth_wp = wp.array(depth, dtype=float, device=device)
    else:
        depth_wp = wp.from_torch(depth)
    out_wp = wp.zeros(depth.shape, dtype=float, device=device)
    wp.launch(kernel=bilateral_filter_depth_kernel, device=device, dim=[depth.shape[0], depth.shape[1]],
              inputs=[depth_wp, out_wp, radius, zfar, sigmaD, sigmaR])
    depth_out = wp.to_torch(out_wp)

    if isinstance(depth, np.ndarray):
        depth_out = depth_out.data.cpu().numpy()
    return depth_out


@wp.kernel(enable_backward=False)
def erode_depth_kernel(depth: wp.array(dtype=float, ndim=2), out: wp.array(dtype=float, ndim=2), radius: int,
                       depth_diff_thres: float, ratio_thres: float, zfar: float):
    h, w = wp.tid()
    H = depth.shape[0]
    W = depth.shape[1]
    if w >= W or h >= H:
        return
    d_ori = depth[h, w]
    if d_ori < 0.001 or d_ori >= zfar:
        out[h, w] = 0.0
    bad_cnt = float(0)
    total = float(0)
    for u in range(w - radius, w + radius + 1):
        if u < 0 or u >= W:
            continue
        for v in range(h - radius, h + radius + 1):
            if v < 0 or v >= H:
                continue
            cur_depth = depth[v, u]
            total += 1.0
            if cur_depth < 0.001 or cur_depth >= zfar or abs(cur_depth - d_ori) > depth_diff_thres:
                bad_cnt += 1.0
    if bad_cnt / total > ratio_thres:
        out[h, w] = 0.0
    else:
        out[h, w] = d_ori


def erode_depth(depth, radius=2, depth_diff_thres=0.001, ratio_thres=0.8, zfar=100, device='cuda'):
    depth_wp = wp.from_torch(torch.as_tensor(depth, dtype=torch.float, device=device))
    out_wp = wp.zeros(depth.shape, dtype=float, device=device)
    wp.launch(kernel=erode_depth_kernel, device=device, dim=[depth.shape[0], depth.shape[1]],
              inputs=[depth_wp, out_wp, radius, depth_diff_thres, ratio_thres, zfar], )
    depth_out = wp.to_torch(out_wp)
    if isinstance(depth, np.ndarray):
        depth_out = depth_out.data.cpu().numpy()
    return depth_out


def depth2xyzmap(depth, K, uvs=None):
    invalid_mask = (depth < 0.001)
    H, W = depth.shape[:2]
    if uvs is None:
        vs, us = np.meshgrid(np.arange(0, H), np.arange(0, W), sparse=False, indexing='ij')
        vs = vs.reshape(-1)
        us = us.reshape(-1)
    else:
        uvs = uvs.round().astype(int)
        us = uvs[:, 0]
        vs = uvs[:, 1]
    zs = depth[vs, us]
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    pts = np.stack((xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)), 1)  # (N,3)
    xyz_map = np.zeros((H, W, 3), dtype=np.float32)
    xyz_map[vs, us] = pts
    xyz_map[invalid_mask] = 0
    return xyz_map


def depth2xyzmap_batch(depths, Ks, zfar):
    '''
    @depths: torch tensor (B,H,W)
    @Ks: torch tensor (B,3,3)
    '''
    bs = depths.shape[0]
    invalid_mask = (depths < 0.001) | (depths > zfar)
    H, W = depths.shape[-2:]
    vs, us = torch.meshgrid(torch.arange(0, H), torch.arange(0, W), indexing='ij')
    vs = vs.reshape(-1).float().cuda()[None].expand(bs, -1)
    us = us.reshape(-1).float().cuda()[None].expand(bs, -1)
    zs = depths.reshape(bs, -1)
    Ks = Ks[:, None].expand(bs, zs.shape[-1], 3, 3)
    xs = (us - Ks[..., 0, 2]) * zs / Ks[..., 0, 0]  # (B,N)
    ys = (vs - Ks[..., 1, 2]) * zs / Ks[..., 1, 1]
    pts = torch.stack([xs, ys, zs], dim=-1)  # (B,N,3)
    xyz_maps = pts.reshape(bs, H, W, 3)
    xyz_maps[invalid_mask] = 0
    return xyz_maps


def sample_views_icosphere(n_views, subdivisions=None, radius=1):
    if subdivisions is not None:
        mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    else:
        subdivision = 1
        while 1:
            mesh = trimesh.creation.icosphere(subdivisions=subdivision, radius=radius)
            if mesh.vertices.shape[0] >= n_views:
                break
            subdivision += 1
    cam_in_obs = np.tile(np.eye(4)[None], (len(mesh.vertices), 1, 1))
    cam_in_obs[:, :3, 3] = mesh.vertices
    up = np.array([0, 0, 1])
    z_axis = -cam_in_obs[:, :3, 3]  # (N,3)
    z_axis /= np.linalg.norm(z_axis, axis=-1).reshape(-1, 1)
    x_axis = np.cross(up.reshape(1, 3), z_axis)
    invalid = (x_axis == 0).all(axis=-1)
    x_axis[invalid] = [1, 0, 0]
    x_axis /= np.linalg.norm(x_axis, axis=-1).reshape(-1, 1)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis, axis=-1).reshape(-1, 1)
    cam_in_obs[:, :3, 0] = x_axis
    cam_in_obs[:, :3, 1] = y_axis
    cam_in_obs[:, :3, 2] = z_axis
    return cam_in_obs


def to_homo(pts):
    '''
    @pts: (N,3 or 2) will homogeneliaze the last dimension
    '''
    assert len(pts.shape) == 2, f'pts.shape: {pts.shape}'
    homo = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=-1)
    return homo


def to_homo_torch(pts):
    '''
    @pts: shape can be (...,N,3 or 2) or (N,3) will homogeneliaze the last dimension
    '''
    ones = torch.ones((*pts.shape[:-1], 1), dtype=torch.float, device=pts.device)
    homo = torch.cat((pts, ones), dim=-1)
    return homo


def transform_pts(pts, tf):
    """Transform 2d or 3d points
    @pts: (...,N_pts,3)
    @tf: (...,4,4)
    """
    if len(tf.shape) >= 3 and tf.shape[-3] != pts.shape[-2]:
        tf = tf[..., None, :, :]
    return (tf[..., :-1, :-1] @ pts[..., None] + tf[..., :-1, -1:])[..., 0]


def transform_dirs(dirs, tf):
    """
    @dirs: (...,3)
    @tf: (...,4,4)
    """
    if len(tf.shape) >= 3 and tf.shape[-3] != dirs.shape[-2]:
        tf = tf[..., None, :, :]
    return (tf[..., :3, :3] @ dirs[..., None])[..., 0]


def random_direction():
    '''https://stackoverflow.com/questions/33976911/generate-a-random-sample-of-points-distributed-on-the-surface-of-a-unit-sphere
    '''
    vec = np.random.randn(3).reshape(3)
    vec /= np.linalg.norm(vec)
    return vec


def project_3d_to_2d(pt, K, ob_in_cam):
    pt = pt.reshape(4, 1)
    projected = K @ ((ob_in_cam @ pt)[:3, :])
    projected = projected.reshape(-1)
    projected = projected / projected[2]
    return projected.reshape(-1)[:2].round().astype(int)


def compute_mesh_diameter(model_pts=None, mesh=None, n_sample=1000):
    if mesh is not None:
        u, s, vh = scipy.linalg.svd(mesh.vertices, full_matrices=False)
        pts = u @ s
        diameter = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
        return float(diameter)

    if n_sample is None:
        pts = model_pts
    else:
        ids = np.random.choice(len(model_pts), size=min(n_sample, len(model_pts)), replace=False)
        pts = model_pts[ids]
    dists = np.linalg.norm(pts[None] - pts[:, None], axis=-1)
    diameter = dists.max()
    return diameter


def compute_crop_window_tf_batch(pts=None, H=None, W=None, poses=None, K=None, crop_ratio=1.2, out_size=None, rgb=None,
                                 uvs=None, method='min_box', mesh_diameter=None):
    '''Project the points and find the cropping transform
    @pts: (N,3)
    @poses: (B,4,4) tensor
    @min_box: min_box/min_circle
    @scale: scale to apply to the tightly enclosing roi
    '''

    def compute_tf_batch(left, right, top, bottom):
        B = len(left)
        left = left.round()
        right = right.round()
        top = top.round()
        bottom = bottom.round()

        tf = torch.eye(3)[None].expand(B, -1, -1).contiguous()
        tf[:, 0, 2] = -left
        tf[:, 1, 2] = -top
        new_tf = torch.eye(3)[None].expand(B, -1, -1).contiguous()
        new_tf[:, 0, 0] = out_size[0] / (right - left)
        new_tf[:, 1, 1] = out_size[1] / (bottom - top)
        tf = new_tf @ tf
        return tf

    B = len(poses)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if method == 'box_3d':
        radius = mesh_diameter * crop_ratio / 2
        offsets = torch.tensor([0, 0, 0,
                                radius, 0, 0,
                                -radius, 0, 0,
                                0, radius, 0,
                                0, -radius, 0]).reshape(-1, 3)
        pts = (poses[:, :3, 3].reshape(-1, 1, 3) + offsets.reshape(1, -1, 3)).float()
        K = torch.Tensor(K).float().cuda(device=pts.device)
        projected = (K @ pts.reshape(-1, 3).T).T
        uvs = projected[:, :2] / projected[:, 2:3]
        uvs = uvs.reshape(B, -1, 2)
        center = uvs[:, 0]  # (B,2)
        radius = torch.abs(uvs - center.reshape(-1, 1, 2)).reshape(B, -1).max(axis=-1)[0].reshape(-1)  # (B)
        left = center[:, 0] - radius
        right = center[:, 0] + radius
        top = center[:, 1] - radius
        bottom = center[:, 1] + radius
        tfs = compute_tf_batch(left, right, top, bottom)
        return tfs
    else:
        raise RuntimeError

    return tf


def trimesh_add_pure_colored_texture(mesh, color=np.array([255, 255, 255]), resolution=5):
    tex_img = np.tile(color.reshape(1, 1, 3), (resolution, resolution, 1)).astype(np.uint8)
    mesh = mesh.unwrap()
    mesh.visual = trimesh.visual.texture.TextureVisuals(uv=mesh.visual.uv, image=Image.fromarray(tex_img))
    return mesh


def projection_matrix_from_intrinsics(K, height, width, znear, zfar, window_coords='y_down'):
    """Conversion of Hartley-Zisserman intrinsic matrix to OpenGL proj. matrix.

    Ref:
    1) https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL
    2) https://github.com/strawlab/opengl-hz/blob/master/src/calib_test_utils.py

    :param K: 3x3 ndarray with the intrinsic camera matrix.
    :param x0 The X coordinate of the camera image origin (typically 0).
    :param y0: The Y coordinate of the camera image origin (typically 0).
    :param w: Image width.
    :param h: Image height.
    :param nc: Near clipping plane.
    :param fc: Far clipping plane.
    :param window_coords: 'y_up' or 'y_down'.
    :return: 4x4 ndarray with the OpenGL projection matrix.
    """
    x0 = 0
    y0 = 0
    w = width
    h = height
    nc = znear
    fc = zfar

    depth = float(fc - nc)
    q = -(fc + nc) / depth
    qn = -2 * (fc * nc) / depth

    # Draw our images upside down, so that all the pixel-based coordinate
    # systems are the same.
    if window_coords == 'y_up':
        proj = np.array([
            [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
            [0, -2 * K[1, 1] / h, (-2 * K[1, 2] + h + 2 * y0) / h, 0],
            [0, 0, q, qn],  # Sets near and far planes (glPerspective).
            [0, 0, -1, 0]
            ])

    # Draw the images upright and modify the projection matrix so that OpenGL
    # will generate window coords that compensate for the flipped image coords.
    elif window_coords == 'y_down':
        proj = np.array([
            [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
            [0, 2 * K[1, 1] / h, (2 * K[1, 2] - h + 2 * y0) / h, 0],
            [0, 0, q, qn],  # Sets near and far planes (glPerspective).
            [0, 0, -1, 0]
            ])
    else:
        raise NotImplementedError

    return proj


def symmetry_tfs_from_info(info, rot_angle_discrete=5):
    symmetry_tfs = [np.eye(4)]
    if 'symmetries_discrete' in info:
        tfs = np.array(info['symmetries_discrete']).reshape(-1, 4, 4)
        tfs[..., :3, 3] *= 0.001
        symmetry_tfs = [np.eye(4)]
        symmetry_tfs += list(tfs)
    if 'symmetries_continuous' in info:
        axis = np.array(info['symmetries_continuous'][0]['axis']).reshape(3)
        offset = info['symmetries_continuous'][0]['offset']
        rxs = [0]
        rys = [0]
        rzs = [0]
        if axis[0] > 0:
            rxs = np.arange(0, 360, rot_angle_discrete) / 180.0 * np.pi
        elif axis[1] > 0:
            rys = np.arange(0, 360, rot_angle_discrete) / 180.0 * np.pi
        elif axis[2] > 0:
            rzs = np.arange(0, 360, rot_angle_discrete) / 180.0 * np.pi
        for rx in rxs:
            for ry in rys:
                for rz in rzs:
                    tf = euler_matrix(rx, ry, rz)
                    tf[:3, 3] = offset
                    symmetry_tfs.append(tf)
    if len(symmetry_tfs) == 0:
        symmetry_tfs = [np.eye(4)]
    symmetry_tfs = np.array(symmetry_tfs)
    return symmetry_tfs


def pose_to_egocentric_delta_pose(A_in_cam, B_in_cam):
    '''Used for Pose Refinement. Given the object's two poses in camera, convert them to relative poses in camera's egocentric view
    @A_in_cam: (B,4,4) torch tensor
    '''
    trans_delta = B_in_cam[:, :3, 3] - A_in_cam[:, :3, 3]
    rot_mat_delta = B_in_cam[:, :3, :3] @ A_in_cam[:, :3, :3].permute(0, 2, 1)
    return trans_delta, rot_mat_delta


def egocentric_delta_pose_to_pose(A_in_cam, trans_delta, rot_mat_delta):
    '''Used for Pose Refinement. Given the object's two poses in camera, convert them to relative poses in camera's egocentric view
    @A_in_cam: (B,4,4) torch tensor
    '''
    B_in_cam = torch.eye(4, dtype=torch.float, device=A_in_cam.device)[None].expand(len(A_in_cam), -1, -1).contiguous()
    B_in_cam[:, :3, 3] = A_in_cam[:, :3, 3] + trans_delta
    B_in_cam[:, :3, :3] = rot_mat_delta @ A_in_cam[:, :3, :3]
    return B_in_cam


def sdg_load_bounding_box(file_path: str):
    """Load bounding boxes.
    Args:
        file_path: Path of the bounding box.

    Returns:
        A dictionary of the bounding boxes.
    """
    bbox_dict = {}
    bbox_array = np.load(file_path)
    for id, x_min, y_min, x_max, y_max, occlusion_ratio in zip(
            bbox_array["semanticId"],
            bbox_array["x_min"],
            bbox_array["y_min"],
            bbox_array["x_max"],
            bbox_array["y_max"],
            bbox_array["occlusionRatio"],
            ):
        bbox_dict[id] = {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "occlusion_ratio": occlusion_ratio,
            }
    return bbox_dict


def texture_map_interpolation(tex_image_numpy):
    all_channels = []
    mask = np.all(tex_image_numpy == 0, axis=2)
    x = np.arange(0, tex_image_numpy.shape[1])
    y = np.arange(0, tex_image_numpy.shape[0])
    xx, yy = np.meshgrid(x, y)
    for each_channel in range(tex_image_numpy.shape[2]):
        curr_channel = tex_image_numpy[:, :, each_channel]
        x1 = xx[~mask]
        y1 = yy[~mask]
        newarr = curr_channel[~mask]
        GD1 = griddata((x1, y1), newarr.ravel(), (xx, yy), method='nearest')
        all_channels.append(GD1[:, :, np.newaxis].round().astype(np.uint8))
    final_image = np.concatenate(all_channels, axis=-1)
    return final_image


class OctreeManager:
    def __init__(self, pts=None, max_level=None, octree=None):
        if octree is None:
            pts_quantized = kaolin.ops.spc.quantize_points(pts.contiguous(), level=max_level)
            self.octree = kaolin.ops.spc.unbatched_points_to_octree(pts_quantized, max_level, sorted=False)
        else:
            self.octree = octree
        lengths = torch.tensor([len(self.octree)], dtype=torch.int32).cpu()
        self.max_level, self.pyramids, self.exsum = kaolin.ops.spc.scan_octrees(self.octree, lengths)
        self.finest_vox_size = 2.0 / (2 ** self.max_level)
        self.n_level = self.max_level + 1
        self.vox_point_all_levels = kaolin.ops.spc.generate_points(self.octree, self.pyramids, self.exsum)
        self.point_hierarchy_dual, self.pyramid_dual = kaolin.ops.spc.unbatched_make_dual(self.vox_point_all_levels,
                                                                                          self.pyramids[0])
        self.trinkets, self.pointers_to_parent = kaolin.ops.spc.unbatched_make_trinkets(self.vox_point_all_levels,
                                                                                        self.pyramids[0],
                                                                                        self.point_hierarchy_dual,
                                                                                        self.pyramid_dual)
        self.n_vox = len(self.vox_point_all_levels)
        self.n_corners = len(self.point_hierarchy_dual)

        for level in range(self.n_level):
            vox_pts = self.get_level_quantized_points(level)
            corner_pts = self.get_level_corner_quantized_points(level)

    def get_level_corner_quantized_points(self, level):
        start = self.pyramid_dual[..., 1, level]
        num = self.pyramid_dual[..., 0, level]
        return self.point_hierarchy_dual[start:start + num]

    def get_level_quantized_points(self, level):
        start = self.pyramids[..., 1, level]
        num = self.pyramids[..., 0, level]
        return self.vox_point_all_levels[start:start + num]

    def get_center_ids(self, x, level):
        '''Get ids with 0 starting from current level's first point
        '''
        pidx = kaolin.ops.spc.unbatched_query(self.octree, self.exsum, x.float(), level, with_parents=False)
        return pidx

    def get_vox_size_at_level(self, level):
        return 2.0 / (2 ** level)

    def draw(self, level, method='point'):
        vox_size = self.get_vox_size_at_level(level)

        if method == 'point':
            corner_coords = self.get_level_corner_quantized_points(level)
            pts = corner_coords * vox_size - 1
            mesh = trimesh.points.PointCloud(pts.data.cpu().numpy().reshape(-1, 3))
            return mesh

    def ray_trace(self, rays_o, rays_d, level, debug=False):
        """Octree is in normalized [-1,1] world coordinate frame
        'rays_o': ray origin in normalized world coordinate system
        'rays_d': (N,3) unit length ray direction in normalized world coordinate system
        'octree': spc
        @voxel_size: in the scale of [-1,1] space
        Return:
            ray_depths_in_out: traveling times, NOT the Z value; invalid will be zeros
        """
        ray_index, rays_pid, depth_in_out = kaolin.render.spc.unbatched_raytrace(self.octree, self.vox_point_all_levels,
                                                                                 self.pyramids[0], self.exsum, rays_o,
                                                                                 rays_d, level=level, return_depth=True,
                                                                                 with_exit=True)
        if ray_index.size()[0] == 0:
            print("[WARNING] batch has 0 intersections!!")
            ray_depths_in_out = torch.zeros((rays_o.shape[0], 1, 2))
            rays_pid = -torch.ones_like(rays_o[:, :1])
            rays_near = torch.zeros_like(rays_o[:, :1])
            rays_far = torch.zeros_like(rays_o[:, :1])
            return rays_near, rays_far, rays_pid, ray_depths_in_out

        intersected_ray_ids, counts = torch.unique_consecutive(ray_index, return_counts=True)
        max_intersections = counts.max().item()
        start_poss = torch.cat([torch.tensor([0], device=counts.device), torch.cumsum(counts[:-1], dim=0)], dim=0)

        ray_depths_in_out = common.postprocessOctreeRayTracing(ray_index.long().contiguous(), depth_in_out.contiguous(),
                                                               intersected_ray_ids.long().contiguous(),
                                                               start_poss.long().contiguous(), max_intersections,
                                                               rays_o.shape[0])

        rays_far = ray_depths_in_out[:, :, 1].max(dim=-1)[0].reshape(-1, 1)
        rays_near = ray_depths_in_out[:, 0, 0].reshape(-1, 1)

        return rays_near, rays_far, rays_pid, ray_depths_in_out


def scale_to_reference(source_pts, reference_pts):
    # 두 점군의 bounding box 크기 계산
    source_min = np.min(source_pts, axis=0)
    source_max = np.max(source_pts, axis=0)
    source_size = source_max - source_min

    ref_min = np.min(reference_pts, axis=0)
    ref_max = np.max(reference_pts, axis=0)
    ref_size = ref_max - ref_min

    # 스케일 factor 계산 (각 축별)
    scale_factors = ref_size / source_size

    # 점군의 중심을 원점으로 이동
    source_center = (source_max + source_min) / 2
    centered_source = source_pts - source_center

    # 스케일링 적용
    scaled_pts = centered_source * scale_factors

    # reference 점군의 중심으로 이동
    ref_center = (ref_max + ref_min) / 2
    scaled_pts = scaled_pts + ref_center

    return scaled_pts


## Trimesh utils ##
def trimesh_clean(mesh):
    mesh.merge_vertices()
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_infinite_values()
    mesh.remove_unreferenced_vertices()
    return mesh


def trimesh_split(mesh, min_edge=1000):
    '''!NOTE mesh.split takes too much memory for large mesh. That's why we have this function
    '''
    components = trimesh.graph.connected_components(mesh.edges, min_len=min_edge, nodes=None, engine=None)
    meshes = []
    for i, c in enumerate(components):
        mask = np.zeros(len(mesh.vertices), dtype=bool)
        mask[c] = 1
        cur_mesh = mesh.copy()
        cur_mesh.update_vertices(mask=mask.astype(bool))
        meshes.append(cur_mesh)
    return meshes


def chamfer_distance_between_clouds_mutual(pts1, pts2):
    kdtree1 = cKDTree(pts1)
    dists1, indices1 = kdtree1.query(pts2)
    kdtree2 = cKDTree(pts2)
    dists2, indices2 = kdtree2.query(pts1)
    return 0.5 * (
                dists1.mean() + dists2.mean())  # !NOTE should not be mean of all, see https://pdal.io/en/stable/apps/chamfer.html


def calculate_chamfer_distance(reader, dir, pred_pose_init_old, gt_poses, input_mesh=None, consdier_size=False):
    """
    Calculate chamfer distance between predicted and ground truth point clouds

    Args:
        reader: Reader object containing video directory info
        dir: Directory containing predicted mesh files
        pred_pose_init_old: Initial prediction pose matrix
        gt_poses: Ground truth pose matrices

    Returns:
        cd: Chamfer distance in centimeters
    """
    # Read ground truth point cloud
    pcd = o3d.io.read_point_cloud(f'{reader.video_dir}/visible_mesh.ply')
    pcd = pcd.voxel_down_sample(0.005)
    gt_pts = np.asarray(pcd.points).copy()

    # Load predicted mesh
    if input_mesh is None:
        tmp = sorted(glob.glob(f"{dir}/**/*mesh_real_world.obj", recursive=True))
        pred_mesh = trimesh.load(tmp[-1])
        pred_mesh.apply_transform(pred_pose_init_old)
        pred_mesh.apply_transform(np.linalg.inv(gt_poses))

        # Filter out vertices outside bounding box to prevent memory issues
        max_coord = gt_pts.max(axis=0).reshape(1, 3) + 0.3
        min_coord = gt_pts.min(axis=0).reshape(1, 3) - 0.3
        bad_mask = (pred_mesh.vertices > max_coord).any(axis=-1) | (pred_mesh.vertices < min_coord).any(axis=-1)
        pred_mesh.vertices[bad_mask] = np.inf

        pred_mesh = trimesh_clean(pred_mesh)

        # Get largest valid component
        components = trimesh_split(pred_mesh, min_edge=1000)
        if len(components) == 0:
            components = trimesh_split(pred_mesh, min_edge=3)

        best_component = None
        best_size = 0
        for component in components:
            dists = np.linalg.norm(component.vertices, axis=-1)
            if dists.min() > 0.1:
                continue
            if len(component.vertices) > best_size:
                best_size = len(component.vertices)
                best_component = component
        pred_mesh = best_component
    else:
        pred_mesh = input_mesh.copy()
        pred_mesh.apply_transform(pred_pose_init_old)
        pred_mesh.apply_transform(np.linalg.inv(gt_poses))

    # Sample points from predicted mesh
    pred_pts, _ = trimesh.sample.sample_surface(pred_mesh, 99999, face_weight=None, sample_color=False)

    # Align predicted points to ground truth using ICP
    pcd_pred = toOpen3dCloud(pred_pts)
    pcd_pred = pcd_pred.voxel_down_sample(0.005)
    pcd_gt = toOpen3dCloud(gt_pts)
    thres = 0.02
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_pred, pcd_gt, thres, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
    pred_pts_icp = (reg_p2p.transformation @ to_homo(pred_pts).T).T[:, :3]

    # Calculate chamfer distance
    chamfer_dists = chamfer_distance_between_clouds_mutual(pred_pts_icp, gt_pts)
    cd = chamfer_dists.mean() * 100
    if consdier_size:
        resize_pred_pts_icp = scale_to_reference(pred_pts_icp, gt_pts)
        resize_chamfer_dists = chamfer_distance_between_clouds_mutual(resize_pred_pts_icp, gt_pts)
        resize_cd = resize_chamfer_dists.mean() * 100

        return cd, resize_cd

    # 포인트 클라우드 객체 생성
    # pred_pcd = o3d.geometry.PointCloud()
    # gt_pcd = o3d.geometry.PointCloud()

    # 포인트 설정
    # pred_pcd.points = o3d.utility.Vector3dVector(pred_pts_icp)
    # gt_pcd.points = o3d.utility.Vector3dVector(gt_pts)
    #
    # # 색상 설정 (예측값은 빨간색, Ground Truth는 초록색)
    # pred_pcd.paint_uniform_color([1, 0, 0])  # 빨간색
    # gt_pcd.paint_uniform_color([0, 1, 0])  # 초록색
    #
    # o3d.visualization.draw_geometries([pred_pcd,gt_pcd])

    return cd


def decompose_pose_matrix(pose):
    """Decompose 4x4 pose matrix into rotation and translation.

    :param pose: 4x4 ndarray with the pose matrix
    :return: tuple (R, t) where R is 3x3 rotation matrix and t is 3x1 translation vector
    """
    if pose.shape == (4, 4):
        R = pose[:3, :3]
        t = pose[:3, 3:4]  # Keep as 2D array with shape (3,1)
        return R, t
    else:
        raise ValueError("Pose matrix must be 4x4")


def calculate_3d_iou(pts_est, pts_gt):
    """
    Calculate 3D IOU between two point clouds using their bounding boxes

    Args:
        pts_est: nx3 ndarray of estimated 3D points
        pts_gt: mx3 ndarray of ground truth 3D points

    Returns:
        float: IOU score between 0 and 1
    """
    # Calculate min/max bounds for each point cloud
    min_est = np.min(pts_est, axis=0)
    max_est = np.max(pts_est, axis=0)
    min_gt = np.min(pts_gt, axis=0)
    max_gt = np.max(pts_gt, axis=0)

    # Calculate volumes
    vol_est = np.prod(max_est - min_est)
    vol_gt = np.prod(max_gt - min_gt)

    # Calculate intersection bounds
    min_intersection = np.maximum(min_est, min_gt)
    max_intersection = np.minimum(max_est, max_gt)

    # Check if boxes overlap
    if np.any(max_intersection < min_intersection):
        return 0.0

    # Calculate intersection volume
    vol_intersection = np.prod(np.maximum(0, max_intersection - min_intersection))

    # Calculate union volume
    vol_union = vol_est + vol_gt - vol_intersection

    # Calculate IOU
    iou = vol_intersection / vol_union

    return (iou * 100.0)


def calculate_3d_iou_with_pose(pose1=None, pose2=None, R1=None, t1=None, R2=None, t2=None, pts=None):
    """Calculate 3D IoU between two transformed point clouds.

    Accepts either 4x4 pose matrices or separate R, t inputs.

    :param pose1: 4x4 ndarray with the first pose matrix (optional)
    :param pose2: 4x4 ndarray with the second pose matrix (optional)
    :param R1: 3x3 ndarray with the first rotation matrix (optional)
    :param t1: 3x1 ndarray with the first translation vector (optional)
    :param R2: 3x3 ndarray with the second rotation matrix (optional)
    :param t2: 3x1 ndarray with the second translation vector (optional)
    :param pts1: nx3 ndarray with first set of original 3D points
    :param pts2: mx3 ndarray with second set of original 3D points
    :return: IoU score between 0 and 1
    """
    # Handle pose matrix inputs
    if pose1 is not None:
        R1, t1 = decompose_pose_matrix(pose1)
    if pose2 is not None:
        R2, t2 = decompose_pose_matrix(pose2)

    # Verify we have all needed transforms
    if R1 is None or t1 is None or R2 is None or t2 is None:
        raise ValueError("Must provide either pose matrices or R, t pairs")

    # Transform points using respective poses
    transformed_pts1 = misc.transform_pts_Rt(pts, R1, t1)
    transformed_pts2 = misc.transform_pts_Rt(pts, R2, t2)

    return calculate_3d_iou(transformed_pts1, transformed_pts2)