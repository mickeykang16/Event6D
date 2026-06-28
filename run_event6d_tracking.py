"""Event6D evaluation on the real-world Event6D dataset (online mode).

Loads the Event6D depth-extrapolation network + E2VID reconstruction once, then iterates
over every sequence listed in `test.txt` under `--video_dirs`, runs the FoundationPose
tracker over its frames at the requested frame rate (--eval_fps 30 or 120), and writes
per-sequence and aggregated metrics under `outputs/<name>/0_eval_metric/`.
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
from bop_toolkit_lib import transform
from bop_toolkit_lib.renderer_vispy import RendererVispy

from Utils import (
    calculate_origin_metrics,
    evaluate_pose_estimation,
    simplify,
    visualize_frame_results_gt,
)
from src.dataloader.my_reader_online_scaled import Event6DReaderScaled
from src.refiner.foundationpose import FoundationPose
from src.refiner.foundationpose.training.predict_pose_refine import PoseRefinePredictor

# Vendored EventVFI for the depth-extrapolation network and E2VID front-end.
from src.depth_extrapolation import models as evfi_models  # noqa: E402
from src.e2vid.image_reconstructor import ImageReconstructor  # noqa: E402
from src.e2vid.utils.loading_utils import load_model  # noqa: E402

class OnlineEventProcessor:
    """Wraps the depth-extrapolation network and E2VID front-end for online inference."""

    def __init__(self, model_path, e2vid_path, num_bins=5, standardization=False):
        device = torch.device('cuda')
        model_spec = torch.load(model_path)['model']
        self.depth_model = evfi_models.make(model_spec, load_sd=True).to(device).eval()
        front_end, _ = load_model(e2vid_path)
        front_end.eval()
        self.reconstructor = ImageReconstructor(
            front_end.to(device), 480, 640, num_bins, device, standardization=standardization
        )
        self.device = device
        self._last_e2vid_img = None  # cached E2VID frame for the next sequence step at j=0

    def reset(self):
        self.reconstructor.last_states_for_each_channel = {'grayscale': None}
        self._last_e2vid_img = None

    @torch.no_grad()
    def process(self, start_depth_np, start_vox, recon_vox):
        """Run one forward pass and return (depth (H,W), color (H,W,3) uint8).

        The voxel resolution may differ from the depth map's; depth is resized to match
        the voxel resolution for inference, then the prediction is consumed at that
        resolution.
        """
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

def perturb_pose(pose, rot_deg, trans_cm):
    """Apply random rotation (deg) + translation (cm) noise to the GT init pose."""
    axis = np.random.randn(3)
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    angle = np.deg2rad(rot_deg)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R_noise = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    t_noise = np.random.randn(3) * (trans_cm / 100.0)
    pose_noisy = pose.copy()
    pose_noisy[:3, :3] = R_noise @ pose[:3, :3]
    pose_noisy[:3, 3] = pose[:3, 3] + t_noise
    return pose_noisy

def build_symmetry_transforms(model_info_entry, max_sym_disc_step=0.01):
    """Return a list of {R, t} transforms encoding the object's continuous symmetries."""
    trans_cont = []
    for sym in model_info_entry.get('symmetries_continuous', []):
        axis = np.array(sym['axis'])
        offset = np.array(sym['offset']).reshape((3, 1))
        discrete_steps_count = int(np.ceil(np.pi / max_sym_disc_step))
        discrete_step = 2.0 * np.pi / discrete_steps_count
        for i in range(discrete_steps_count):
            R = transform.rotation_matrix(i * discrete_step, axis)[:3, :3]
            t = -R.dot(offset) + offset
            trans_cont.append({'R': R, 't': t})
    return trans_cont if trans_cont else [{'R': np.eye(3), 't': np.zeros((3, 1))}]

def tracking(args):
    save_root = Path('./outputs') / args.name
    save_eval_path = str(save_root / args.eval_folder_name)
    os.makedirs(save_eval_path, exist_ok=True)

    with open(os.path.join(args.video_dirs, 'test.txt')) as f:
        obj_folder = sorted(line.strip().split()[0] for line in f if line.strip())

    with open('workspace/models_info_eot.json') as f:
        model_info = json.load(f)

    processor = OnlineEventProcessor(
        args.depth_extrapolation_ckpt, args.e2vid_ckpt, standardization=args.standardization
    ) if args.online else None

    ev_w, ev_h = args.ev_width, args.ev_height
    scale_x = ev_w / 1280.0
    scale_y = ev_h / 720.0
    needs_resize = scale_x != 1.0 or scale_y != 1.0

    metrics_all = []
    summary_metrics_all = []
    poses_all = []

    for obj_f in obj_folder:
        if processor is not None:
            processor.reset()

        video_dir = os.path.join(args.video_dirs, obj_f)
        reader = Event6DReaderScaled(video_dir, ev_width=ev_w, ev_height=ev_h)

        obj_name = obj_f.replace('/', '-')
        save_results_est_per_path = str(save_root / obj_name)

        glctx = dr.RasterizeCudaContext()
        gt_mesh = reader.get_gt_mesh()
        gt_diameter = reader.get_gt_mesh_diamter()
        ob_id = reader.get_obj_id()
        trans_disc = build_symmetry_transforms(model_info[f'{ob_id}'])

        renderer = RendererVispy(ev_w, ev_h, mode='depth')
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

        for i in tqdm(range(0, video_len, args.stride), desc=obj_name):
            if args.online:
                start_depth = reader.get_start_depth_event(i)
                if needs_resize:
                    start_depth = cv2.resize(start_depth, (ev_w, ev_h), interpolation=cv2.INTER_LINEAR)

            metrics = []
            for j in range(1 if args.rgbd_baseline else 4):
                gt_pose = reader.get_gt_pose(i, j, frame='event' if args.e2vid else 'rgb')
                if gt_pose is None:
                    continue

                if args.online:
                    if j == 0:
                        depth = start_depth
                        H, W = depth.shape[:2]
                        color = processor._last_e2vid_img if processor._last_e2vid_img is not None \
                                else np.full((H, W, 3), 128, dtype=np.uint8)
                    else:
                        start_vox, recon_vox = reader.get_event_voxels(i, j)
                        depth, color = processor.process(start_depth, start_vox, recon_vox)
                else:
                    color = reader.get_event_img(i, j, False) if args.e2vid else reader.get_color(i, j)
                    depth = reader.get_depth(i, j, frame='event' if args.e2vid else 'rgb')

                mask = reader.get_mask(i).astype(np.bool_)
                if needs_resize:
                    mask = cv2.resize(mask.astype(np.uint8), (ev_w, ev_h),
                                      interpolation=cv2.INTER_NEAREST).astype(np.bool_)

                K = reader.K1.copy() if args.e2vid else reader.K.copy()
                if needs_resize:
                    K[0, :] *= scale_x
                    K[1, :] *= scale_y

                if i == 0 and j == 0:
                    init_pose = gt_pose
                    if args.init_rot_noise > 0 or args.init_trans_noise > 0:
                        init_pose = perturb_pose(gt_pose, args.init_rot_noise, args.init_trans_noise)
                    est.pose_last = torch.tensor(init_pose).float().cuda()
                    pred_pose = init_pose
                else:
                    pred_pose = est.track_one(rgb=color, depth=depth, K=K, iteration=args.iteration, index=i)

                poses_all.append(pred_pose)
                metric = calculate_origin_metrics(
                    est=est, pred_pose=pred_pose, gt_pose=gt_pose, gt_mesh=gt_mesh, K=K,
                    mask=mask, gt_diameter=gt_diameter, trans_disc=trans_disc,
                    frame_idx=i, renderer=renderer, ob_id=int(ob_id), depth=depth,
                )
                metrics.append(metric)

                if j == 0 and i % args.viz_step == 0:
                    vis_img = visualize_frame_results_gt(
                        color=color, gt_mesh=gt_mesh, K=K, gt_pose=gt_pose, pred_pose=pred_pose,
                        metric=metric, obj_f=obj_name, frame_idx=i, save_path=save_eval_path,
                        glctx=glctx, name=f'{len(reader.color_files)}_{args.name}',
                    )
                    if args.online and vis_img is not None:
                        vis_path = os.path.join(save_eval_path,
                                                f'{obj_name}_img_{len(reader.color_files)}_{args.name}')
                        depth_vis = visualize_depth(depth)
                        h = vis_img.shape[0]
                        depth_vis_resized = cv2.resize(
                            depth_vis,
                            (int(depth_vis.shape[1] * h / depth_vis.shape[0]), h),
                        )
                        depth_vis_rgb = cv2.cvtColor(depth_vis_resized, cv2.COLOR_BGR2RGB)
                        combined = np.concatenate([vis_img, depth_vis_rgb], axis=1)
                        imageio.imwrite(os.path.join(vis_path, f'{obj_name}_img_{i:05d}.jpg'), combined)

            # Cache the IFrameIndex=4 E2VID output as next-step j=0 input.
            if args.online:
                start_vox4, recon_vox4 = reader.get_event_voxels(i, 4)
                _, color4 = processor.process(start_depth, start_vox4, recon_vox4)
                processor._last_e2vid_img = color4

            if args.eval_fps == 120:
                metrics_all.extend(metrics)
                metrics_obj.extend(metrics)
            elif args.eval_fps == 30:
                metrics_all.extend(metrics[:1])
                metrics_obj.extend(metrics[:1])
            else:
                raise NotImplementedError

        summary_metrics, _ = evaluate_pose_estimation(reader, obj_name, metrics_obj,
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
    pose_save = save_root / obj_name / f'est_poses_stride_{args.stride}.npy'
    pose_save.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(pose_save), poses_all)

def parse_args():
    parser = argparse.ArgumentParser(description='Event6D evaluation on the Event6D dataset')
    parser.add_argument('--name', type=str, default='event6d_eval', help='Experiment name')
    parser.add_argument('--video_dirs', type=str, default='./data/Event6D',
                        help='Root directory containing test.txt and per-object sequences')
    parser.add_argument('--eval_folder_name', type=str, default='0_eval_metric')
    parser.add_argument('--stride', type=int, default=1, help='Frame stride')
    parser.add_argument('--max_length', type=int, default=-1, help='Max frames per sequence (-1 = all)')
    parser.add_argument('--iteration', type=int, default=2, help='Pose refiner iterations')
    parser.add_argument('--init_rot_noise', type=float, default=0.0,
                        help='Initial rotation noise (deg) added to GT init')
    parser.add_argument('--init_trans_noise', type=float, default=0.0,
                        help='Initial translation noise (cm) added to GT init')
    parser.add_argument('--ckpt', type=str, default='./weights/foundationpose_refiner/model.pth',
                        help='FoundationPose refiner checkpoint')
    parser.add_argument('--e2vid', action='store_true', default=False,
                        help='Use event-camera intrinsics + E2VID images')
    parser.add_argument('--rgbd_baseline', action='store_true', default=False,
                        help='Vanilla FoundationPose RGB-D baseline: one refinement per RGB '
                             'frame at the RGB rate (no events / E2VID / depth extrapolation)')
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
    parser.add_argument('--standardization', action='store_true', default=False,
                        help='Standardize E2VID outputs')
    parser.add_argument('--ev_width', type=int, default=1280, help='Event-camera voxel grid width')
    parser.add_argument('--ev_height', type=int, default=720, help='Event-camera voxel grid height')
    return parser.parse_args()

if __name__ == '__main__':
    tracking(parse_args())
