"""Event6D evaluation on the HO3D-v2 event-augmented dataset (online mode).

Iterates over the 13 HO3D-v2 evaluation sequences (AP10–AP14, MPM10–MPM14, SB11, SB13,
SM1), runs the Event6D depth-extrapolation + E2VID + FoundationPose pipeline at the
requested frame rate (30 or 120), and writes per-sequence and aggregated metrics under
`outputs/<name>/0_eval_metric/`.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import nvdiffrast.torch as dr
from bop_toolkit_lib.renderer_vispy import RendererVispy

from Utils import (
    calculate_origin_metrics,
    evaluate_pose_estimation,
    simplify,
    visualize_frame_results_gt,
)
from src.dataloader.ho3d_reader import Ho3dReader
from src.dataloader.representations import ReconVoxelGrid
from src.refiner.foundationpose import FoundationPose
from src.refiner.foundationpose.training.predict_pose_refine import PoseRefinePredictor

# Vendored EventVFI for the depth-extrapolation network and E2VID front-end.
from src.depth_extrapolation import models as evfi_models  # noqa: E402
from src.e2vid.image_reconstructor import ImageReconstructor  # noqa: E402
from src.e2vid.utils.loading_utils import load_model  # noqa: E402

HO3D_SEQUENCES = [
    'AP10', 'AP11', 'AP12', 'AP13', 'AP14',
    'MPM10', 'MPM11', 'MPM12', 'MPM13', 'MPM14',
    'SB11', 'SB13', 'SM1',
]
EV_H, EV_W = 480, 640  # HO3D event-camera resolution
HO3D_DEPTH_SCALE = 0.00012498664727900177

class OnlineEventProcessor:
    """Wraps the depth-extrapolation network and E2VID front-end for online inference."""

    def __init__(self, model_path, e2vid_path, num_bins=5, standardization=False):
        device = torch.device('cuda')
        model_spec = torch.load(model_path)['model']
        self.depth_model = evfi_models.make(model_spec, load_sd=True).to(device).eval()
        front_end, _ = load_model(e2vid_path)
        front_end.eval()
        self.reconstructor = ImageReconstructor(
            front_end.to(device), EV_H, EV_W, num_bins, device, standardization=standardization
        )
        self.device = device
        self._last_e2vid_img = None  # cached E2VID frame for next step's j=0

    def reset(self):
        self.reconstructor.last_states_for_each_channel = {'grayscale': None}
        self._last_e2vid_img = None

    @torch.no_grad()
    def process(self, start_depth_np, start_vox, recon_vox):
        H_full, W_full = start_depth_np.shape[:2]
        inp_full = torch.tensor(start_depth_np[np.newaxis, np.newaxis]).float().to(self.device) - 1.0
        sv = start_vox.unsqueeze(0).to(self.device)
        rv = recon_vox.unsqueeze(0).to(self.device)

        Hv, Wv = sv.shape[-2], sv.shape[-1]
        if Hv != H_full or Wv != W_full:
            inp = torch.nn.functional.interpolate(inp_full, size=(Hv, Wv), mode='bilinear', align_corners=False)
        else:
            inp = inp_full

        recon_img, _, latent = self.reconstructor.update_reconstruction(rv)
        pred, _ = self.depth_model(inp, sv.float(), latent)

        depth_np = (pred[0, 0] + 1.0).cpu().numpy()
        color_np = (255.0 * recon_img[0].cpu().numpy().transpose(1, 2, 0)).astype(np.uint8)
        if color_np.shape[2] == 1:
            color_np = np.repeat(color_np, 3, axis=2)
        return depth_np, color_np

def visualize_depth(depth, vmin=None, vmax=None):
    """depth (H,W) float32 → BGR colormap image (H,W,3) uint8."""
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if len(valid) == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    lo = np.nanpercentile(valid, 1.0) if vmin is None else vmin
    hi = np.nanpercentile(valid, 99.0) if vmax is None else vmax
    if hi <= lo:
        lo, hi = float(valid.min()), float(valid.max())
    normed = np.clip((depth - lo) / (hi - lo + 1e-9) * 255, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(normed, cv2.COLORMAP_TURBO)

_voxel_grid = None
_recon_voxel_grid = None

def _init_voxel_grids(num_bins=5):
    global _voxel_grid, _recon_voxel_grid
    if _voxel_grid is None:
        _voxel_grid = ReconVoxelGrid(num_bins, EV_H, EV_W, normalize=False)
        _recon_voxel_grid = ReconVoxelGrid(num_bins, EV_H, EV_W, normalize=False)

def get_ho3d_event_voxels(reader, i, j, stride, num_bins=5):
    """Build (start_vox, recon_vox) torch tensors (num_bins, EV_H, EV_W) for frame i, sub-step j.

    Aggregates raw events from `reader.event_files[i+1 : i+stride+1]`; `start_vox` covers
    `[t[0], t[j/4]]` (cumulative) and `recon_vox` covers `[t[(j-1)/4], t[j/4]]` (window).
    """
    _init_voxel_grids(num_bins)

    ev_list = []
    for f in reader.event_files[i + 1: i + stride + 1]:
        if not os.path.isfile(f):
            continue
        ev_list.append(np.load(f, mmap_mode='r')['data'])

    if not ev_list:
        zeros = torch.zeros(num_bins, EV_H, EV_W)
        return zeros, zeros

    event = np.concatenate(ev_list, axis=0)
    # HO3D npz columns: x (0~639), y (0~479), polarity (±1), timestamp.
    x = event[:, 0].astype('float32')
    y = event[:, 1].astype('float32')
    t = event[:, 3].astype('float64')
    p = event[:, 2].astype('float32')

    mask = (x >= 0) & (x < EV_W) & (y >= 0) & (y < EV_H)
    x, y, t, p = x[mask], y[mask], t[mask], p[mask]
    if len(x) == 0:
        zeros = torch.zeros(num_bins, EV_H, EV_W)
        return zeros, zeros

    t_len = float(t[-1] - t[0])
    t_end = t[0] + t_len * j / 4

    mc = t <= t_end
    xc, yc, tc, pc = x[mc], y[mc], t[mc], p[mc]
    if len(xc) > 1:
        tc_n = ((tc - tc[0]) / (tc[-1] - tc[0] + 1e-9)).astype('float32')
        start_vox = _voxel_grid.convert(
            torch.from_numpy(xc), torch.from_numpy(yc),
            torch.from_numpy(pc), torch.from_numpy(tc_n))
    else:
        start_vox = torch.zeros(num_bins, EV_H, EV_W)

    t_start_r = t[0] + t_len * (j - 1) / 4
    mr = (t >= t_start_r) & (t <= t_end)
    xr, yr, tr, pr = x[mr], y[mr], t[mr], p[mr]
    if len(xr) > 1:
        tr_n = ((tr - tr[0]) / (tr[-1] - tr[0] + 1e-9)).astype('float32')
        recon_vox = _recon_voxel_grid.convert(
            torch.from_numpy(xr), torch.from_numpy(yr),
            torch.from_numpy(pr), torch.from_numpy(tr_n))
    else:
        recon_vox = torch.zeros(num_bins, EV_H, EV_W)

    return start_vox, recon_vox

def load_ho3d_depth(color_file):
    """Decode HO3D 16-bit RGB-encoded depth from the corresponding depth/ file."""
    depth_file = color_file.replace('.jpg', '.png').replace('rgb', 'depth')
    d = cv2.imread(depth_file, -1)
    return (d[..., 2] + d[..., 1] * 256) * HO3D_DEPTH_SCALE

def build_symmetry_transforms(model_info_entry):
    """Identity + any discrete symmetries listed in BOP `models_info.json`."""
    out = [{'R': np.eye(3), 't': np.zeros((3, 1))}]
    for sym in model_info_entry.get('symmetries_discrete', []):
        sym_4x4 = np.reshape(sym, (4, 4))
        out.append({'R': sym_4x4[:3, :3], 't': sym_4x4[:3, 3].reshape((3, 1))})
    return out

def tracking(args):
    save_root = Path('./outputs') / args.name
    save_eval_path = str(save_root / args.eval_folder_name)
    os.makedirs(save_eval_path, exist_ok=True)

    with open('workspace/models_info.json') as f:
        model_info = json.load(f)

    processor = OnlineEventProcessor(args.depth_extrapolation_ckpt, args.e2vid_ckpt) if args.online else None

    metrics_all = []
    summary_metrics_all = []
    poses_all = []

    for obj_f in HO3D_SEQUENCES:
        if processor is not None:
            processor.reset()

        save_results_est_per_path = str(save_root / obj_f)
        reader = Ho3dReader(os.path.join(args.video_dirs, obj_f))

        glctx = dr.RasterizeCudaContext()
        gt_mesh = reader.get_gt_mesh()
        gt_diameter = reader.get_gt_mesh_diamter()
        ob_id = reader.get_obj_id()
        trans_disc = build_symmetry_transforms(model_info[f'{ob_id}'])

        renderer = RendererVispy(EV_W, EV_H, mode='depth')
        renderer.my_add_object({
            'pts': np.asarray(gt_mesh.vertices),
            'normals': np.asarray(gt_mesh.face_normals),
            'faces': np.asarray(gt_mesh.faces),
        }, int(ob_id))

        est = FoundationPose(
            model_pts=gt_mesh.vertices.copy(),
            model_normals=gt_mesh.vertex_normals.copy(),
            mesh=gt_mesh,
            debug_dir=save_results_est_per_path,
            debug=0,
            glctx=glctx,
            refiner=PoseRefinePredictor(args.ckpt),
        )
        est.refiner.cfg['cv_gray'] = False

        video_len = len(reader.color_files) if args.max_length == -1 else args.max_length
        metrics_obj = []

        for i in tqdm(range(0, video_len - args.stride, args.stride), desc=obj_f):
            gt_pose = reader.get_gt_pose(i)
            if gt_pose is None:
                continue

            mask = reader.get_mask(i).astype(np.bool_)

            if args.online:
                start_depth = load_ho3d_depth(reader.color_files[i])

            metrics = []
            for j in range(4):
                if args.online:
                    if j == 0:
                        depth = start_depth
                        color = processor._last_e2vid_img if processor._last_e2vid_img is not None \
                                else cv2.cvtColor(cv2.imread(reader.color_files[i]), cv2.COLOR_BGR2RGB)
                    else:
                        start_vox, recon_vox = get_ho3d_event_voxels(reader, i, j, args.stride)
                        depth, color = processor.process(start_depth, start_vox, recon_vox)
                else:
                    color = cv2.cvtColor(cv2.imread(reader.color_files[i]), cv2.COLOR_BGR2RGB)
                    depth = reader.get_depth(i, j)

                if i == 0 and j == 0:
                    est.pose_last = torch.tensor(gt_pose).float().cuda()
                    pred_pose = gt_pose
                else:
                    pred_pose = est.track_one(rgb=color, depth=depth, K=reader.K,
                                              iteration=args.iteration)

                poses_all.append(pred_pose)
                metric = calculate_origin_metrics(
                    est=est, pred_pose=pred_pose, gt_pose=gt_pose, gt_mesh=gt_mesh,
                    K=reader.K, mask=mask, gt_diameter=gt_diameter, trans_disc=trans_disc,
                    frame_idx=i, renderer=renderer, ob_id=int(ob_id), depth=depth,
                )
                metrics.append(metric)

                if j == 0 and i % args.viz_step == 0:
                    vis_img = visualize_frame_results_gt(
                        color=color, gt_mesh=gt_mesh, K=reader.K, gt_pose=gt_pose,
                        pred_pose=pred_pose, metric=metric, obj_f=obj_f, frame_idx=i,
                        save_path=save_eval_path, glctx=glctx,
                        name=f'{len(reader.color_files)}_{args.name}',
                    )
                    if args.online and vis_img is not None:
                        vis_path = os.path.join(
                            save_eval_path,
                            f'{obj_f}_img_{len(reader.color_files)}_{args.name}',
                        )
                        depth_vis = visualize_depth(depth)
                        depth_vis_resized = cv2.resize(
                            depth_vis,
                            (int(depth_vis.shape[1] * vis_img.shape[0] / depth_vis.shape[0]),
                             vis_img.shape[0]),
                        )
                        depth_vis_rgb = cv2.cvtColor(depth_vis_resized, cv2.COLOR_BGR2RGB)
                        combined = np.concatenate([vis_img, depth_vis_rgb], axis=1)
                        imageio.imwrite(os.path.join(vis_path, f'{obj_f}_img_{i:05d}.jpg'), combined)

            if args.online:
                start_vox4, recon_vox4 = get_ho3d_event_voxels(reader, i, 4, args.stride)
                _, color4 = processor.process(start_depth, start_vox4, recon_vox4)
                processor._last_e2vid_img = color4

            if args.eval_fps == 120:
                metrics_all.extend(metrics)
                metrics_obj.extend(metrics)
            else:  # 30 fps
                metrics_all.extend(metrics[:1])
                metrics_obj.extend(metrics[:1])

        summary_metrics, _ = evaluate_pose_estimation(reader, obj_f, metrics_obj,
                                                     save_eval_path, dir=None)
        summary_metrics_all.append(summary_metrics)

    summary_metrics, _ = evaluate_pose_estimation(reader, 'ALL', metrics_all,
                                                  save_eval_path, dir=None)
    summary_metrics_all.append(summary_metrics)

    for entry in summary_metrics_all:
        for key in entry:
            entry[key] = simplify(entry[key])
    df_summary = pd.DataFrame(summary_metrics_all)
    excel_path = os.path.join(save_eval_path, '0_mean_all.xlsx')
    with pd.ExcelWriter(excel_path) as writer:
        df_summary.to_excel(writer, sheet_name='Summary_Metrics', index=False)
    print(f'Metrics saved to: {excel_path}')

    poses_all = np.array(poses_all)
    pose_save = save_root / obj_f / f'est_poses_stride_{args.stride}.npy'
    pose_save.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(pose_save), poses_all)

def parse_args():
    parser = argparse.ArgumentParser(description='Event6D evaluation on the HO3D-v2 dataset')
    parser.add_argument('--name', type=str, default='eventho3d_eval', help='Experiment name')
    parser.add_argument('--video_dirs', type=str, default='./data/EventHO3D/evaluation',
                        help='Root directory containing the 13 HO3D-v2 evaluation sequences')
    parser.add_argument('--eval_folder_name', type=str, default='0_eval_metric')
    parser.add_argument('--stride', type=int, default=10, help='Frame stride')
    parser.add_argument('--max_length', type=int, default=-1, help='Max frames per sequence (-1 = all)')
    parser.add_argument('--iteration', type=int, default=2, help='Pose refiner iterations')
    parser.add_argument('--ckpt', type=str, default='./weights/foundationpose_refiner/model.pth',
                        help='FoundationPose refiner checkpoint')
    parser.add_argument('--eval_fps', type=int, default=30, choices=[30, 120],
                        help='Evaluate at 30 fps (RGB rate) or 120 fps (per sub-timestep)')
    parser.add_argument('--viz_step', type=int, default=1, help='Save visualizations every N frames')
    parser.add_argument('--online', action='store_true', default=False,
                        help='Run E2VID + depth extrapolation online')
    parser.add_argument('--depth_extrapolation_ckpt', type=str,
                        default='./weights/depth_extrapolation.pth',
                        help='Event6D depth-extrapolation checkpoint')
    parser.add_argument('--e2vid_ckpt', type=str,
                        default='./weights/e2vid.pth.tar',
                        help='E2VID front-end checkpoint')
    return parser.parse_args()

if __name__ == '__main__':
    tracking(parse_args())
