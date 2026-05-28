# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import functools
import os,sys,kornia
import time
import numpy as np
import torch
from omegaconf import OmegaConf
from src.refiner.foundationpose.models.refine_network import *
from src.refiner.foundationpose.datasets.h5_dataset import *
from Utils import *

#@torch.inference_mode()
def make_crop_data_batch(render_size, ob_in_cams, mesh, rgb, depth, K, crop_ratio, xyz_map, normal_map=None, mesh_diameter=None, cfg=None, glctx=None, mesh_tensors=None, dataset:PoseRefinePairH5Dataset=None):
  # logging.info("Welcome make_crop_data_batch")

  H,W = depth.shape[-2:]
  args = []
  method = 'box_3d'
  device = ob_in_cams.device
  dtype = ob_in_cams.dtype

  try:
    tf_to_crops = compute_crop_window_tf_batch(pts=None, H=H, W=W, poses=ob_in_cams, K=K, crop_ratio=crop_ratio, out_size=(render_size[1], render_size[0]), method=method, mesh_diameter=mesh_diameter).to(device)
  except:
    tf_to_crops = compute_crop_window_tf_batch_test(pts=None, H=H, W=W, poses=ob_in_cams, K=K, crop_ratio=crop_ratio, out_size=(render_size[1], render_size[0]), method=method, mesh_diameter=mesh_diameter).to(device)
# logging.info("make tf_to_crops done")

  B = len(ob_in_cams)
  poseA = torch.as_tensor(ob_in_cams, dtype=torch.float, device='cuda')

  # bs = 512

  if not isinstance(mesh_tensors, list):
    mesh_tensors = [mesh_tensors]
    bs = B
  else:
    bs = 1
  rgb_rs = []
  depth_rs = []
  normal_rs = []
  xyz_map_rs = []

  bbox2d_crop = torch.tensor(np.array([0, 0, cfg['input_resize'][0]-1, cfg['input_resize'][1]-1]).reshape(2,1,2), device=device, dtype=dtype).repeat(1, B, 1)
  # (2 B 2)
  bbox2d_ori = transform_pts(bbox2d_crop.reshape(-1, 2), tf_to_crops.inverse().repeat(2, 1, 1)).reshape(2, B, 2).permute(1, 0, 2)

  for b in range(0,len(poseA),bs):
    extra = {}
    if isinstance(K, torch.Tensor) and len(K) == len(poseA):
      K_np = K[b].detach().cpu().numpy() if K.is_cuda else K.numpy()
    elif isinstance(K, np.ndarray):
      # 이미 NumPy
      K_np = K

    rgb_r, depth_r, normal_r = nvdiffrast_render(K=K_np, H=H, W=W, ob_in_cams=poseA[b:b+bs], context='cuda',
                                                 get_normal=cfg['use_normal'], glctx=glctx, mesh_tensors=mesh_tensors[b],
                                                 output_size=cfg['input_resize'], bbox2d=bbox2d_ori[b:b+bs].reshape(-1, 4), use_light=True, extra=extra)

    if 'cv_gray' in cfg and cfg['cv_gray']:
      gray = 0.299 * rgb_r[..., 0] + 0.587 * rgb_r[..., 1] + 0.114 * rgb_r[..., 2]  # (1, H, W)
      rgb_r[..., 0] = gray
      rgb_r[..., 1] = gray
      rgb_r[..., 2] = gray
      rgb_r = rgb_r.clip(0, 1)

    rgb_rs.append(rgb_r)
    depth_rs.append(depth_r[...,None])
    normal_rs.append(normal_r)
    xyz_map_rs.append(extra['xyz_map'])
  rgb_rs = torch.cat(rgb_rs, dim=0).permute(0,3,1,2) * 255
  depth_rs = torch.cat(depth_rs, dim=0).permute(0,3,1,2)  #(B,1,H,W)
  xyz_map_rs = torch.cat(xyz_map_rs, dim=0).permute(0,3,1,2)  #(B,3,H,W)

  if isinstance(K, torch.Tensor) and len(K) == len(poseA):
    Ks = K
    rgb = rgb
    xyz_map = xyz_map.permute(0, 3, 1, 2)
  else:
    Ks = torch.as_tensor(K, device='cuda', dtype=torch.float).reshape(1,3,3)
    rgb = torch.as_tensor(rgb, dtype=torch.float, device='cuda').permute(2,0,1)[None].expand(B,-1,-1,-1)
    xyz_map = torch.as_tensor(xyz_map, device='cuda', dtype=torch.float).permute(2,0,1)[None].expand(B,-1,-1,-1)
  if cfg['use_normal']:
    normal_rs = torch.cat(normal_rs, dim=0).permute(0,3,1,2)  #(B,3,H,W)

  # logging.info("render done")
  rgbBs = kornia.geometry.transform.warp_perspective(rgb, tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
  if rgb_rs.shape[-2:]!=cfg['input_resize']:
    rgbAs = kornia.geometry.transform.warp_perspective(rgb_rs, tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
  else:
    rgbAs = rgb_rs

  if xyz_map_rs.shape[-2:]!=cfg['input_resize']:
    xyz_mapAs = kornia.geometry.transform.warp_perspective(xyz_map_rs, tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
  else:
    xyz_mapAs = xyz_map_rs
  xyz_mapBs = kornia.geometry.transform.warp_perspective(xyz_map, tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)  #(B,3,H,W)

  if cfg['use_normal']:
    normalAs = kornia.geometry.transform.warp_perspective(normal_rs, tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
    normalBs = kornia.geometry.transform.warp_perspective(torch.as_tensor(normal_map, dtype=torch.float, device='cuda').permute(2,0,1)[None].expand(B,-1,-1,-1), tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
  else:
    normalAs = None
    normalBs = None

  # logging.info("warp done")
  md_device = mesh_diameter.device if isinstance(mesh_diameter, torch.Tensor) else 'cpu'
  mesh_diameters = (torch.ones((len(rgbAs)), dtype=torch.float, device=md_device)*mesh_diameter).to(rgbAs.device)
  # mesh_diameters = mesh_diameter
  pose_data = BatchPoseData(rgbAs=rgbAs, rgbBs=rgbBs, depthAs=None, depthBs=None, normalAs=normalAs, normalBs=normalBs, poseA=poseA, poseB=None, xyz_mapAs=xyz_mapAs, xyz_mapBs=xyz_mapBs, tf_to_crops=tf_to_crops, Ks=Ks, mesh_diameters=mesh_diameters)
  pose_data = dataset.transform_batch(batch=pose_data, H_ori=H, W_ori=W, bound=1)

  # logging.info("pose batch data done")

  return pose_data

class PoseRefinePredictor:
  def __init__(self, load_pretrained='default', for_ev=False, config_path=None):
    self.amp = True
    code_dir = os.path.dirname(os.path.realpath(__file__))
    default_dir = f'{code_dir}/../../../../weights/foundationpose_refiner'
    ckpt_dir = f'{default_dir}/model.pth'
    cfg_path = config_path or f'{default_dir}/config.yml'

    # If a custom ckpt path is provided, look for a sibling `config.yml` next to it.
    if load_pretrained and load_pretrained != 'default' and config_path is None:
      sibling_cfg = os.path.join(os.path.dirname(load_pretrained), 'config.yml')
      if os.path.isfile(sibling_cfg):
        cfg_path = sibling_cfg

    self.cfg = OmegaConf.load(cfg_path)

    self.cfg['ckpt_dir'] = ckpt_dir
    self.cfg['enable_amp'] = True

    ########## Defaults, to be backward compatible
    if 'use_normal' not in self.cfg:
      self.cfg['use_normal'] = False
    if 'use_mask' not in self.cfg:
      self.cfg['use_mask'] = False
    if 'use_BN' not in self.cfg:
      self.cfg['use_BN'] = False
    if 'c_in' not in self.cfg:
      self.cfg['c_in'] = 4
    if 'crop_ratio' not in self.cfg or self.cfg['crop_ratio'] is None:
      self.cfg['crop_ratio'] = 1.2
    if 'n_view' not in self.cfg:
      self.cfg['n_view'] = 1
    if 'trans_rep' not in self.cfg:
      self.cfg['trans_rep'] = 'tracknet'
    if 'rot_rep' not in self.cfg:
      self.cfg['rot_rep'] = 'axis_angle'
    if 'zfar' not in self.cfg:
      self.cfg['zfar'] = 3
    if 'normalize_xyz' not in self.cfg:
      self.cfg['normalize_xyz'] = False
    if isinstance(self.cfg['zfar'], str) and 'inf' in self.cfg['zfar'].lower():
      self.cfg['zfar'] = np.inf
    if 'normal_uint8' not in self.cfg:
      self.cfg['normal_uint8'] = False
    # logging.info(f"self.cfg: \n {OmegaConf.to_yaml(self.cfg)}")

    self.dataset = PoseRefinePairH5Dataset(cfg=self.cfg, h5_file='', mode='test')

    if for_ev:
      self.model = RefineNetEv(cfg=self.cfg, c_in=self.cfg['c_in'], c_in_ev=13).cuda()
      # self.model = RefineNetEv_v2(cfg=self.cfg, c_in=self.cfg['c_in'], c_in_ev=13).cuda()
    else:
      self.model = RefineNet(cfg=self.cfg, c_in=self.cfg['c_in']).cuda()

    if load_pretrained:
      if load_pretrained != 'default':
        ckpt_dir = load_pretrained
      logging.info(f"Using pretrained model from {ckpt_dir}")
      ckpt = torch.load(ckpt_dir, map_location='cpu')
      if 'model' in ckpt:
        ckpt = ckpt['model']
      self.model.load_state_dict(ckpt)

      self.model.cuda().eval()
      logging.info("init done")
    self.last_trans_update = None
    self.last_rot_update = None

  def train(self, rgb, depth, K, ob_in_cams, xyz_map, normal_map=None, get_vis=False, mesh=None, mesh_tensors=None, glctx=None, mesh_diameter=None, iteration=5):
    '''
    @rgb: np array (H,W,3)
    @ob_in_cams: np array (N,4,4)
    '''
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # logging.info(f'ob_in_cams:{ob_in_cams.shape}')
    tf_to_center = np.eye(4)
    ob_centered_in_cams = ob_in_cams
    mesh_centered = mesh

    # logging.info(f'self.cfg.use_normal:{self.cfg.use_normal}')
    if not self.cfg.use_normal:
      normal_map = None

    crop_ratio = self.cfg['crop_ratio']
    # logging.info(f"trans_normalizer:{self.cfg['trans_normalizer']}, rot_normalizer:{self.cfg['rot_normalizer']}")
    bs = 1024

    B_in_cams = torch.as_tensor(ob_centered_in_cams, device='cuda', dtype=torch.float)

    if mesh_tensors is None:
      mesh_tensors = make_mesh_tensors(mesh_centered)

    rgb_tensor = torch.as_tensor(rgb, device='cuda', dtype=torch.float)
    depth_tensor = torch.as_tensor(depth, device='cuda', dtype=torch.float)
    xyz_map_tensor = torch.as_tensor(xyz_map, device='cuda', dtype=torch.float)
    trans_normalizer = self.cfg['trans_normalizer']
    if not isinstance(trans_normalizer, float):
      trans_normalizer = torch.as_tensor(list(trans_normalizer), device='cuda', dtype=torch.float).reshape(1,3)

    for _ in range(iteration):
      # logging.info("making cropped data")
      pose_data = make_crop_data_batch(self.cfg.input_resize, B_in_cams, mesh_centered, rgb_tensor, depth_tensor, K, crop_ratio=crop_ratio, normal_map=normal_map, xyz_map=xyz_map_tensor, cfg=self.cfg, glctx=glctx, mesh_tensors=mesh_tensors, dataset=self.dataset, mesh_diameter=mesh_diameter)
      B_in_cams = []
      for b in range(0, pose_data.rgbAs.shape[0], bs):
        A = torch.cat([pose_data.rgbAs[b:b+bs].cuda(), pose_data.xyz_mapAs[b:b+bs].cuda()], dim=1).float()
        B = torch.cat([pose_data.rgbBs[b:b+bs].cuda(), pose_data.xyz_mapBs[b:b+bs].cuda()], dim=1).float()
        # logging.info("forward start")
        with torch.cuda.amp.autocast(enabled=self.amp):
          output = self.model(A,B)
        for k in output:
          output[k] = output[k].float()
        # logging.info("forward done")
        if self.cfg['trans_rep']=='tracknet':
          if not self.cfg['normalize_xyz']:
            trans_delta = torch.tanh(output["trans"])*trans_normalizer
          else:
            trans_delta = output["trans"]

        elif self.cfg['trans_rep']=='deepim':
          def project_and_transform_to_crop(centers):
            uvs = (pose_data.Ks[b:b+bs]@centers.reshape(-1,3,1)).reshape(-1,3)
            uvs = uvs/uvs[:,2:3]
            uvs = (pose_data.tf_to_crops[b:b+bs]@uvs.reshape(-1,3,1)).reshape(-1,3)
            return uvs[:,:2]

          rot_delta = output["rot"]
          z_pred = output['trans'][:,2]*pose_data.poseA[b:b+bs][...,2,3]
          uvA_crop = project_and_transform_to_crop(pose_data.poseA[b:b+bs][...,:3,3])
          uv_pred_crop = uvA_crop + output['trans'][:,:2]*self.cfg['input_resize'][0]
          uv_pred = transform_pts(uv_pred_crop, pose_data.tf_to_crops[b:b+bs].inverse().cuda())
          center_pred = torch.cat([uv_pred, torch.ones((len(rot_delta),1), dtype=torch.float, device='cuda')], dim=-1)
          center_pred = (pose_data.Ks[b:b+bs].inverse().cuda()@center_pred.reshape(len(rot_delta),3,1)).reshape(len(rot_delta),3) * z_pred.reshape(len(rot_delta),1)
          trans_delta = center_pred-pose_data.poseA[b:b+bs][...,:3,3]

        else:
          trans_delta = output["trans"]

        if self.cfg['rot_rep']=='axis_angle':
          rot_mat_delta = torch.tanh(output["rot"])*self.cfg['rot_normalizer']
          rot_mat_delta = so3_exp_map(rot_mat_delta).permute(0,2,1)
        elif self.cfg['rot_rep']=='6d':
          rot_mat_delta = rotation_6d_to_matrix(output['rot']).permute(0,2,1)
        else:
          raise RuntimeError

        if self.cfg['normalize_xyz']:
          trans_delta *= (mesh_diameter/2)

        B_in_cam = egocentric_delta_pose_to_pose(pose_data.poseA[b:b+bs], trans_delta=trans_delta, rot_mat_delta=rot_mat_delta)
        B_in_cams.append(B_in_cam)

      B_in_cams = torch.cat(B_in_cams, dim=0).reshape(len(ob_in_cams),4,4)

    B_in_cams_out = B_in_cams@torch.tensor(tf_to_center[None], device='cuda', dtype=torch.float)
    torch.cuda.empty_cache()
    self.last_trans_update = trans_delta
    self.last_rot_update = rot_mat_delta

    if get_vis:
      # logging.info("get_vis...")
      canvas = []
      padding = 2
      pose_data = make_crop_data_batch(self.cfg.input_resize, torch.as_tensor(ob_centered_in_cams), mesh_centered, rgb, depth, K, crop_ratio=crop_ratio, normal_map=normal_map, xyz_map=xyz_map_tensor, cfg=self.cfg, glctx=glctx, mesh_tensors=mesh_tensors, dataset=self.dataset, mesh_diameter=mesh_diameter)
      for id in range(0, len(B_in_cams)):
        rgbA_vis = (pose_data.rgbAs[id]*255).permute(1,2,0).data.cpu().numpy()
        rgbB_vis = (pose_data.rgbBs[id]*255).permute(1,2,0).data.cpu().numpy()
        row = [rgbA_vis, rgbB_vis]
        H,W = rgbA_vis.shape[:2]
        if pose_data.depthAs is not None:
          depthA = pose_data.depthAs[id].data.cpu().numpy().reshape(H,W)
          depthB = pose_data.depthBs[id].data.cpu().numpy().reshape(H,W)
        elif pose_data.xyz_mapAs is not None:
          depthA = pose_data.xyz_mapAs[id][2].data.cpu().numpy().reshape(H,W)
          depthB = pose_data.xyz_mapBs[id][2].data.cpu().numpy().reshape(H,W)
        zmin = min(depthA.min(), depthB.min())
        zmax = max(depthA.max(), depthB.max())
        depthA_vis = depth_to_vis(depthA, zmin=zmin, zmax=zmax, inverse=False)
        depthB_vis = depth_to_vis(depthB, zmin=zmin, zmax=zmax, inverse=False)
        row += [depthA_vis, depthB_vis]
        if pose_data.normalAs is not None:
          pass
        row = make_grid_image(row, nrow=len(row), padding=padding, pad_value=255)
        row = cv_draw_text(row, text=f'id:{id}', uv_top_left=(10,10), color=(0,255,0), fontScale=0.5)
        canvas.append(row)
      canvas = make_grid_image(canvas, nrow=1, padding=padding, pad_value=255)

      pose_data = make_crop_data_batch(self.cfg.input_resize, B_in_cams, mesh_centered, rgb, depth, K, crop_ratio=crop_ratio, normal_map=normal_map, xyz_map=xyz_map_tensor, cfg=self.cfg, glctx=glctx, mesh_tensors=mesh_tensors, dataset=self.dataset, mesh_diameter=mesh_diameter)
      canvas_refined = []
      for id in range(0, len(B_in_cams)):
        rgbA_vis = (pose_data.rgbAs[id]*255).permute(1,2,0).data.cpu().numpy()
        rgbB_vis = (pose_data.rgbBs[id]*255).permute(1,2,0).data.cpu().numpy()
        row = [rgbA_vis, rgbB_vis]
        H,W = rgbA_vis.shape[:2]
        if pose_data.depthAs is not None:
          depthA = pose_data.depthAs[id].data.cpu().numpy().reshape(H,W)
          depthB = pose_data.depthBs[id].data.cpu().numpy().reshape(H,W)
        elif pose_data.xyz_mapAs is not None:
          depthA = pose_data.xyz_mapAs[id][2].data.cpu().numpy().reshape(H,W)
          depthB = pose_data.xyz_mapBs[id][2].data.cpu().numpy().reshape(H,W)
        zmin = min(depthA.min(), depthB.min())
        zmax = max(depthA.max(), depthB.max())
        depthA_vis = depth_to_vis(depthA, zmin=zmin, zmax=zmax, inverse=False)
        depthB_vis = depth_to_vis(depthB, zmin=zmin, zmax=zmax, inverse=False)
        row += [depthA_vis, depthB_vis]
        row = make_grid_image(row, nrow=len(row), padding=padding, pad_value=255)
        canvas_refined.append(row)

      canvas_refined = make_grid_image(canvas_refined, nrow=1, padding=padding, pad_value=255)
      canvas = make_grid_image([canvas, canvas_refined], nrow=2, padding=padding, pad_value=255)
      torch.cuda.empty_cache()
      return B_in_cams_out, canvas

    return B_in_cams_out, None

  @torch.inference_mode()
  def predict(self, rgb, depth, K, ob_in_cams, xyz_map, normal_map=None, get_vis=False, mesh=None, mesh_tensors=None, glctx=None, mesh_diameter=None, iteration=5):
    '''
    @rgb: np array (H,W,3)
    @ob_in_cams: np array (N,4,4)
    '''
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # logging.info(f'ob_in_cams:{ob_in_cams.shape}')
    tf_to_center = np.eye(4)
    ob_centered_in_cams = ob_in_cams
    mesh_centered = mesh

    # logging.info(f'self.cfg.use_normal:{self.cfg.use_normal}')
    if not self.cfg.use_normal:
      normal_map = None

    crop_ratio = self.cfg['crop_ratio']
    # logging.info(f"trans_normalizer:{self.cfg['trans_normalizer']}, rot_normalizer:{self.cfg['rot_normalizer']}")
    bs = 1024

    B_in_cams = torch.as_tensor(ob_centered_in_cams, device='cuda', dtype=torch.float)

    if mesh_tensors is None:
      mesh_tensors = make_mesh_tensors(mesh_centered)

    rgb_tensor = torch.as_tensor(rgb, device='cuda', dtype=torch.float)
    depth_tensor = torch.as_tensor(depth, device='cuda', dtype=torch.float)
    xyz_map_tensor = torch.as_tensor(xyz_map, device='cuda', dtype=torch.float)
    trans_normalizer = self.cfg['trans_normalizer']
    if not isinstance(trans_normalizer, float):
      trans_normalizer = torch.as_tensor(list(trans_normalizer), device='cuda', dtype=torch.float).reshape(1,3)
    # K_tensor = torch.as_tensor(K, device='cuda', dtype=torch.float)

    for _ in range(iteration):
      # logging.info("making cropped data")
      pose_data = make_crop_data_batch(self.cfg.input_resize, B_in_cams, mesh_centered, rgb_tensor, depth_tensor, K, crop_ratio=crop_ratio, normal_map=normal_map, xyz_map=xyz_map_tensor, cfg=self.cfg, glctx=glctx, mesh_tensors=mesh_tensors, dataset=self.dataset, mesh_diameter=mesh_diameter)
      B_in_cams = []
      for b in range(0, pose_data.rgbAs.shape[0], bs):
        A = torch.cat([pose_data.rgbAs[b:b+bs].cuda(), pose_data.xyz_mapAs[b:b+bs].cuda()], dim=1).float()
        B = torch.cat([pose_data.rgbBs[b:b+bs].cuda(), pose_data.xyz_mapBs[b:b+bs].cuda()], dim=1).float()
        # logging.info("forward start")
        with torch.cuda.amp.autocast(enabled=self.amp):
          output = self.model(A,B)
        for k in output:
          output[k] = output[k].float()
        # logging.info("forward done")
        if self.cfg['trans_rep']=='tracknet':
          if not self.cfg['normalize_xyz']:
            trans_delta = torch.tanh(output["trans"])*trans_normalizer
          else:
            trans_delta = output["trans"]

        elif self.cfg['trans_rep']=='deepim':
          def project_and_transform_to_crop(centers):
            uvs = (pose_data.Ks[b:b+bs]@centers.reshape(-1,3,1)).reshape(-1,3)
            uvs = uvs/uvs[:,2:3]
            uvs = (pose_data.tf_to_crops[b:b+bs]@uvs.reshape(-1,3,1)).reshape(-1,3)
            return uvs[:,:2]

          rot_delta = output["rot"]
          z_pred = output['trans'][:,2]*pose_data.poseA[b:b+bs][...,2,3]
          uvA_crop = project_and_transform_to_crop(pose_data.poseA[b:b+bs][...,:3,3])
          uv_pred_crop = uvA_crop + output['trans'][:,:2]*self.cfg['input_resize'][0]
          uv_pred = transform_pts(uv_pred_crop, pose_data.tf_to_crops[b:b+bs].inverse().cuda())
          center_pred = torch.cat([uv_pred, torch.ones((len(rot_delta),1), dtype=torch.float, device='cuda')], dim=-1)
          center_pred = (pose_data.Ks[b:b+bs].inverse().cuda()@center_pred.reshape(len(rot_delta),3,1)).reshape(len(rot_delta),3) * z_pred.reshape(len(rot_delta),1)
          trans_delta = center_pred-pose_data.poseA[b:b+bs][...,:3,3]

        else:
          trans_delta = output["trans"]

        if self.cfg['rot_rep']=='axis_angle':
          rot_mat_delta = torch.tanh(output["rot"])*self.cfg['rot_normalizer']
          rot_mat_delta = so3_exp_map(rot_mat_delta).permute(0,2,1)
        elif self.cfg['rot_rep']=='6d':
          rot_mat_delta = rotation_6d_to_matrix(output['rot']).permute(0,2,1)
        else:
          raise RuntimeError

        if self.cfg['normalize_xyz']:
          trans_delta *= (mesh_diameter/2)

        B_in_cam = egocentric_delta_pose_to_pose(pose_data.poseA[b:b+bs], trans_delta=trans_delta, rot_mat_delta=rot_mat_delta)
        B_in_cams.append(B_in_cam)

      B_in_cams = torch.cat(B_in_cams, dim=0).reshape(len(ob_in_cams),4,4)

    B_in_cams_out = B_in_cams@torch.tensor(tf_to_center[None], device='cuda', dtype=torch.float)
    torch.cuda.empty_cache()
    self.last_trans_update = trans_delta
    self.last_rot_update = rot_mat_delta

    if get_vis:
      # logging.info("get_vis...")
      canvas = []
      padding = 2
      pose_data = make_crop_data_batch(self.cfg.input_resize, torch.as_tensor(ob_centered_in_cams), mesh_centered, rgb, depth, K, crop_ratio=crop_ratio, normal_map=normal_map, xyz_map=xyz_map_tensor, cfg=self.cfg, glctx=glctx, mesh_tensors=mesh_tensors, dataset=self.dataset, mesh_diameter=mesh_diameter)
      for id in range(0, len(B_in_cams)):
        rgbA_vis = (pose_data.rgbAs[id]*255).permute(1,2,0).data.cpu().numpy()
        rgbB_vis = (pose_data.rgbBs[id]*255).permute(1,2,0).data.cpu().numpy()
        row = [rgbA_vis, rgbB_vis]
        H,W = rgbA_vis.shape[:2]
        if pose_data.depthAs is not None:
          depthA = pose_data.depthAs[id].data.cpu().numpy().reshape(H,W)
          depthB = pose_data.depthBs[id].data.cpu().numpy().reshape(H,W)
        elif pose_data.xyz_mapAs is not None:
          depthA = pose_data.xyz_mapAs[id][2].data.cpu().numpy().reshape(H,W)
          depthB = pose_data.xyz_mapBs[id][2].data.cpu().numpy().reshape(H,W)
        zmin = min(depthA.min(), depthB.min())
        zmax = max(depthA.max(), depthB.max())
        depthA_vis = depth_to_vis(depthA, zmin=zmin, zmax=zmax, inverse=False)
        depthB_vis = depth_to_vis(depthB, zmin=zmin, zmax=zmax, inverse=False)
        row += [depthA_vis, depthB_vis]
        if pose_data.normalAs is not None:
          pass
        row = make_grid_image(row, nrow=len(row), padding=padding, pad_value=255)
        row = cv_draw_text(row, text=f'id:{id}', uv_top_left=(10,10), color=(0,255,0), fontScale=0.5)
        canvas.append(row)
      canvas = make_grid_image(canvas, nrow=1, padding=padding, pad_value=255)

      pose_data = make_crop_data_batch(self.cfg.input_resize, B_in_cams, mesh_centered, rgb, depth, K, crop_ratio=crop_ratio, normal_map=normal_map, xyz_map=xyz_map_tensor, cfg=self.cfg, glctx=glctx, mesh_tensors=mesh_tensors, dataset=self.dataset, mesh_diameter=mesh_diameter)
      canvas_refined = []
      for id in range(0, len(B_in_cams)):
        rgbA_vis = (pose_data.rgbAs[id]*255).permute(1,2,0).data.cpu().numpy()
        rgbB_vis = (pose_data.rgbBs[id]*255).permute(1,2,0).data.cpu().numpy()
        row = [rgbA_vis, rgbB_vis]
        H,W = rgbA_vis.shape[:2]
        if pose_data.depthAs is not None:
          depthA = pose_data.depthAs[id].data.cpu().numpy().reshape(H,W)
          depthB = pose_data.depthBs[id].data.cpu().numpy().reshape(H,W)
        elif pose_data.xyz_mapAs is not None:
          depthA = pose_data.xyz_mapAs[id][2].data.cpu().numpy().reshape(H,W)
          depthB = pose_data.xyz_mapBs[id][2].data.cpu().numpy().reshape(H,W)
        zmin = min(depthA.min(), depthB.min())
        zmax = max(depthA.max(), depthB.max())
        depthA_vis = depth_to_vis(depthA, zmin=zmin, zmax=zmax, inverse=False)
        depthB_vis = depth_to_vis(depthB, zmin=zmin, zmax=zmax, inverse=False)
        row += [depthA_vis, depthB_vis]
        row = make_grid_image(row, nrow=len(row), padding=padding, pad_value=255)
        canvas_refined.append(row)

      canvas_refined = make_grid_image(canvas_refined, nrow=1, padding=padding, pad_value=255)
      canvas = make_grid_image([canvas, canvas_refined], nrow=2, padding=padding, pad_value=255)
      torch.cuda.empty_cache()
      return B_in_cams_out, canvas

    return B_in_cams_out, None

  def refine_pose_with_render_and_model(self, mesh, mesh_tensors, object_pose, K, color, depth, xyz_map, glctx, mesh_diameter,vis=False, misc={}):
    tf_to_center = np.eye(4)
    mesh_centered = mesh
    crop_ratio = self.cfg['crop_ratio']
    bs = 1024
    B_in_cams = object_pose
    device = color.device

    pose_data = make_crop_data_batch(self.cfg.input_resize, B_in_cams, mesh_centered, color, depth, K,
                                     crop_ratio=crop_ratio, normal_map=None, xyz_map=xyz_map, cfg=self.cfg,
                                     glctx=glctx, mesh_tensors=mesh_tensors, dataset=self.dataset,
                                     mesh_diameter=mesh_diameter)

    B_in_cams = []
    for b in range(0, pose_data.rgbAs.shape[0], bs):
        A = torch.cat([pose_data.rgbAs[b:b + bs].to(device), pose_data.xyz_mapAs[b:b + bs].to(device)], dim=1).float()
        B = torch.cat([pose_data.rgbBs[b:b + bs].to(device), pose_data.xyz_mapBs[b:b + bs].to(device)], dim=1).float()
        # logging.info("forward start")
        with torch.cuda.amp.autocast(enabled=misc.get('amp', False)):
            output = self.model(A, B)
        for k in output:
            output[k] = output[k].float()

        # logging.info("forward done")
        if self.cfg['trans_rep'] == 'tracknet':
            if not self.cfg['normalize_xyz']:
                # use now
                trans_delta = torch.tanh(output["trans"]) * self.cfg['trans_normalizer']
            else:
                trans_delta = output["trans"]
        else:
            trans_delta = output["trans"]

        if self.cfg['rot_rep'] == 'axis_angle':
            # use now
            rot_mat_delta = torch.tanh(output["rot"]) * self.cfg['rot_normalizer']
            rot_mat_delta = so3_exp_map(rot_mat_delta).permute(0, 2, 1)

        if self.cfg['normalize_xyz']:
            if mesh_diameter.ndim == 0: mesh_diameter.unsqueeze_(0)
            trans_delta *= (mesh_diameter[:, None].to(device) / 2)

        B_in_cam = egocentric_delta_pose_to_pose(pose_data.poseA[b:b + bs], trans_delta=trans_delta, rot_mat_delta=rot_mat_delta)
        B_in_cams.append(B_in_cam)
    B_in_cams = torch.cat(B_in_cams, dim=0).reshape(len(object_pose), 4, 4)
    B_in_cams_out = B_in_cams @ torch.tensor(tf_to_center[None], device=device, dtype=torch.float)
    torch.cuda.empty_cache()

    results = {}
    results['B_in_cams'] = B_in_cams
    results['B_in_cams_out'] = B_in_cams_out
    results['trans_delta'] = trans_delta
    results['rot_mat_delta'] = rot_mat_delta

    with torch.no_grad():
        if vis:
            # logging.info("get_vis...")
            if 'color' in misc: color = misc['color']

            canvas = []
            padding = 2
            pose_data = make_crop_data_batch(self.cfg.input_resize, torch.as_tensor(object_pose),
                                            mesh_centered, color, depth, K, crop_ratio=crop_ratio,
                                            normal_map=None, xyz_map=xyz_map, cfg=self.cfg,
                                            glctx=glctx, mesh_tensors=mesh_tensors, dataset=self.dataset,
                                            mesh_diameter=mesh_diameter)
            for id in range(0, len(B_in_cams), 8):
                rgbA_vis = (pose_data.rgbAs[id] * 255).permute(1, 2, 0).data.cpu().numpy()
                rgbB_vis = (pose_data.rgbBs[id] * 255).permute(1, 2, 0).data.cpu().numpy()

                H, W = rgbA_vis.shape[:2]
                if pose_data.depthAs is not None:
                    depthA = pose_data.depthAs[id].data.cpu().numpy().reshape(H, W)
                    depthB = pose_data.depthBs[id].data.cpu().numpy().reshape(H, W)
                elif pose_data.xyz_mapAs is not None:
                    depthA = pose_data.xyz_mapAs[id][2].data.cpu().numpy().reshape(H, W)
                    depthB = pose_data.xyz_mapBs[id][2].data.cpu().numpy().reshape(H, W)
                zmin = min(depthA.min(), depthB.min())
                zmax = max(depthA.max(), depthB.max())
                # depthA_vis = depth_to_vis(depthA, zmin=zmin, zmax=zmax, inverse=False)
                # depthB_vis = depth_to_vis(depthB, zmin=zmin, zmax=zmax, inverse=False)

                maskA = depthA != 0
                maskA_check = rgbB_vis.copy()
                maskA_check[maskA==False] = 0

                # depthA_check = depthB_vis.copy()
                # depthA_check[maskA==False] = 0

                row = [rgbA_vis, rgbB_vis, maskA_check]
                # row += [depthA_vis, depthB_vis, depthA_check]
                if pose_data.normalAs is not None:
                    pass
                # row = make_grid_image(row, nrow=len(row), padding=padding, pad_value=255)
                row = make_grid_image(row, nrow=1, padding=padding, pad_value=255)
                row = cv_draw_text(row, text=f'id:{id}', uv_top_left=(10, 10), color=(0, 255, 0), fontScale=0.5)
                canvas.append(row)
            # canvas = make_grid_image(canvas, nrow=1, padding=padding, pad_value=255)
            canvas = make_grid_image(canvas, nrow=len(canvas), padding=padding, pad_value=255)

            pose_data = make_crop_data_batch(self.cfg.input_resize, B_in_cams, mesh_centered, color, depth, K,
                                            crop_ratio=crop_ratio, normal_map=None, xyz_map=xyz_map,
                                            cfg=self.cfg, glctx=glctx, mesh_tensors=mesh_tensors,
                                            dataset=self.dataset, mesh_diameter=mesh_diameter)
            canvas_refined = []
            for id in range(0, len(B_in_cams), 8):
                rgbA_vis = (pose_data.rgbAs[id] * 255).permute(1, 2, 0).data.cpu().numpy()
                rgbB_vis = (pose_data.rgbBs[id] * 255).permute(1, 2, 0).data.cpu().numpy()
                H, W = rgbA_vis.shape[:2]
                if pose_data.depthAs is not None:
                    depthA = pose_data.depthAs[id].data.cpu().numpy().reshape(H, W)
                    depthB = pose_data.depthBs[id].data.cpu().numpy().reshape(H, W)
                elif pose_data.xyz_mapAs is not None:
                    depthA = pose_data.xyz_mapAs[id][2].data.cpu().numpy().reshape(H, W)
                    depthB = pose_data.xyz_mapBs[id][2].data.cpu().numpy().reshape(H, W)
                zmin = min(depthA.min(), depthB.min())
                zmax = max(depthA.max(), depthB.max())
                # depthA_vis = depth_to_vis(depthA, zmin=zmin, zmax=zmax, inverse=False)
                # depthB_vis = depth_to_vis(depthB, zmin=zmin, zmax=zmax, inverse=False)

                maskA = depthA != 0
                maskA_check = rgbB_vis.copy()
                maskA_check[maskA==False] = 0

                # depthA_check = depthB_vis.copy()
                # depthA_check[maskA==False] = 0

                row = [rgbA_vis, rgbB_vis, maskA_check]
                row = make_grid_image(row, nrow=1, padding=padding, pad_value=255)
                canvas_refined.append(row)

            # canvas_refined = make_grid_image(canvas_refined, nrow=1, padding=padding, pad_value=255)
            canvas_refined = make_grid_image(canvas_refined, nrow=len(canvas_refined), padding=padding, pad_value=255)
            canvas = make_grid_image([canvas, canvas_refined], nrow=1, padding=4, pad_value=255)
            # canvas = np.concatenate([canvas, canvas_refined], axis=0)  # 직접 가로 concat (height 동일해야 함)
            torch.cuda.empty_cache()
            results['vis'] = canvas[...,::-1]

    return results

    return results
