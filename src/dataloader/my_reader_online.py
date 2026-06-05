# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import pickle, glob, cv2, imageio, os, sys, re, json, trimesh, copy, logging, multiprocessing, subprocess, \
    joblib
import numpy as np
import ruamel.yaml
# import yaml

from .representations_og import ReconVoxelGrid
import torch

from bop_toolkit_lib.misc import calc_pts_diameter
yaml = ruamel.yaml.YAML()
from Utils import *

import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from scipy.spatial.transform import Rotation as R

from scipy.spatial import ConvexHull, distance

def euler_to_matrix_scipy(angles, order="zyx", degrees=True, homogeneous=False):
    """
    SciPy로 오일러 각 -> 회전행렬
    - angles: (a1, a2, a3). order 순서에 대응
    - order : 예) 'zyx'(yaw-pitch-roll), 'xyz', 'zxy' 등
              ※ 소문자 = intrinsic(회전 후 축), 대문자 = extrinsic(고정 축)
    - degrees: True면 도(deg), False면 라디안
    - homogeneous: True면 4x4 동차행렬 반환
    """
    rot = R.from_euler(order, angles, degrees=degrees)
    M = rot.as_matrix()  # 3x3
    if not homogeneous:
        return M
    H = np.eye(4, dtype=float)
    H[:3, :3] = M
    return H

def inverse_T(T):
    assert T.shape == (4, 4)
    R = T[:3, :3]
    R_inv = np.linalg.inv(R)
    t = T[:-1, -1]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:-1, -1] = -R_inv @ t
    return T_inv

def load_extrinsic_yaml(yaml_path: str):
    with open(yaml_path, "r") as f:
        data = yaml.load(f)

    R = np.array(data["R"], dtype=np.float64)  # (3,3)
    t = np.array(data["t"], dtype=np.float64).reshape(3, 1)  # (3,1)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()

    return T

def parse_cam0_from_kalibr(yaml_path: str, cam0_key: str = "cam0", cam1_key: str = "cam1"):
    with open(yaml_path, "r") as f:
        y = yaml.load(f)
    if cam0_key not in y:
        raise KeyError(f"{cam0_key} not found in {yaml_path}")

    node0 = y[cam0_key]
    intr = node0.get("intrinsics")
    if intr is None or len(intr) < 4:
        raise ValueError("intrinsics must be [fx, fy, cx, cy] in camchain.yaml")
    fx, fy, cx, cy = map(float, intr)
    K0 = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]], dtype=np.float64)

    dist_model = str(node0.get("distortion_model", "radtan")).lower()
    dist_coeffs = node0.get("distortion_coeffs", [])
    D0 = np.array([float(v) for v in dist_coeffs], dtype=np.float64).reshape(-1,)

    H = W = None
    if "resolution" in node0 and node0["resolution"] is not None:
        cols, rows = node0["resolution"]; W, H = int(cols), int(rows)
    elif "image_width" in node0 and "image_height" in node0:
        W = int(node0["image_width"]); H = int(node0["image_height"])
    else:
        H = int(round(2 * float(cy))); W = int(round(2 * float(cx)))
    return K0, D0, (H, W), dist_model

def pose_to_matrix(px, py, pz, qx, qy, qz, qw):
    # SciPy expects [x, y, z, w] (same 순서)
    r = R.from_quat([qx, qy, qz, qw])
    Rmat = r.as_matrix()  # 3x3 rotation

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rmat
    T[:3, 3] = [px, py, pz]
    return T

def warp_depth_rgb_to_event(
    depth_rgb,           # (H_rgb, W_rgb), depth in meters in RGB cam
    K_rgb,               # (3,3) intrinsic of RGB
    K_ev,                # (3,3) intrinsic of Event
    T_ev_rgb,            # (4,4) transform from RGB cam to Event cam (X_ev = T_ev_rgb @ X_rgb)
    out_hw=None,         # (H_ev, W_ev). None이면 K_ev에 맞춘 동일 해상도라고 가정
    invalid_depth=0.0,   # depth_rgb에서 무효값 기준
    bilinear_splat=True  # True면 부드럽게 분배(holes 줄임), False면 최근접 픽셀만
):
    """
    Return:
        depth_ev: (H_ev, W_ev) Event 프레임의 깊이(Z_ev)
        hit_mask: (H_ev, W_ev) 어떤 픽셀이 채워졌는지
    """

    H_rgb, W_rgb = depth_rgb.shape
    if out_hw is None:
        # K_ev가 대응하는 해상도가 따로 있다면 그걸 넣어 주세요.
        # 기본은 RGB와 동일 크기로 만듭니다.
        H_ev, W_ev = H_rgb, W_rgb
    else:
        H_ev, W_ev = out_hw

    # 유효 depth 마스크
    valid = np.isfinite(depth_rgb) & (depth_rgb > 0) & (depth_rgb != invalid_depth)
    if not np.any(valid):
        return np.zeros((H_ev, W_ev), np.float32), np.zeros((H_ev, W_ev), bool)

    # 픽셀 그리드 (RGB)
    jj, ii = np.meshgrid(np.arange(W_rgb), np.arange(H_rgb))  # jj=x(u), ii=y(v)
    u = jj[valid].astype(np.float64)
    v = ii[valid].astype(np.float64)
    z = depth_rgb[valid].astype(np.float64)

    # K_inv로 back-project in RGB cam
    Kinv = np.linalg.inv(K_rgb)
    ones = np.ones_like(u)
    pix_h = np.stack([u, v, ones], axis=0)                   # (3, N)
    rays = Kinv @ pix_h                                       # (3, N) normalized rays
    X_rgb = rays * z                                          # (3, N)

    # 동차 좌표로 변환하고 Event 좌표계로
    N = X_rgb.shape[1]
    X_rgb_h = np.vstack([X_rgb, np.ones((1, N))])            # (4, N)
    X_ev_h = T_ev_rgb @ X_rgb_h                               # (4, N)
    X_ev = X_ev_h[:3, :]                                      # (3, N)
    Xe, Ye, Ze = X_ev[0], X_ev[1], X_ev[2]

    # 앞쪽(z>0)만 유효
    infront = Ze > 1e-6
    if not np.any(infront):
        return np.zeros((H_ev, W_ev), np.float32), np.zeros((H_ev, W_ev), bool)

    Xe, Ye, Ze = Xe[infront], Ye[infront], Ze[infront]

    # Event 카메라로 투영
    uvw = K_ev @ np.stack([Xe, Ye, Ze], axis=0)              # (3, M)
    ue = uvw[0] / uvw[2]
    ve = uvw[1] / uvw[2]
    ze = Ze                                                  # 깊이는 Ze (Event 카메라 z)

    # 유효한 이미지 영역
    in_img = (ue >= 0) & (ue < W_ev - 1) & (ve >= 0) & (ve < H_ev - 1)
    if not np.any(in_img):
        return np.zeros((H_ev, W_ev), np.float32), np.zeros((H_ev, W_ev), bool)

    ue = ue[in_img]; ve = ve[in_img]; ze = ze[in_img]

    # Z-buffer용 초기화 (가까운 깊이만 남기기)
    depth_ev = np.full((H_ev, W_ev), np.inf, dtype=np.float64)

    if bilinear_splat:
        # bilinear splatting: 주변 4픽셀에 가중치로 분배 (holes 감소)
        u0 = np.floor(ue).astype(int); v0 = np.floor(ve).astype(int)
        du = ue - u0; dv = ve - v0

        # 4개 코너
        neighbors = [
            (u0,   v0,   (1-du)*(1-dv)),
            (u0+1, v0,     du  *(1-dv)),
            (u0,   v0+1, (1-du)*dv),
            (u0+1, v0+1,   du  *dv),
        ]

        for uu, vv, w in neighbors:
            # 화면 밖 체크 (위에서 in_img로 -1 피했으므로 이중 체크)
            mask = (uu >= 0) & (uu < W_ev) & (vv >= 0) & (vv < H_ev) & (w > 1e-9)
            if not np.any(mask):
                continue
            # z-buffer: 더 가까우면 갱신
            # 여러 포인트가 같은 픽셀로 들어오면 최소 z를 유지
            idx_u = uu[mask]; idx_v = vv[mask]; zc = ze[mask]
            # 벡터화된 최소 업데이트
            old = depth_ev[idx_v, idx_u]
            new = np.minimum(old, zc)
            depth_ev[idx_v, idx_u] = new
    else:
        # 최근접 픽셀로만 찍기 (빠르지만 holes 많을 수 있음)
        u_nn = np.rint(ue).astype(int)
        v_nn = np.rint(ve).astype(int)
        m = (u_nn >= 0) & (u_nn < W_ev) & (v_nn >= 0) & (v_nn < H_ev)
        u_nn = u_nn[m]; v_nn = v_nn[m]; zc = ze[m]
        old = depth_ev[v_nn, u_nn]
        new = np.minimum(old, zc)
        depth_ev[v_nn, u_nn] = new

    # inf → 0으로 (미채움)
    hit_mask = np.isfinite(depth_ev)
    depth_ev[~hit_mask] = 0.0
    return depth_ev.astype(np.float32), hit_mask

class Event6DReader:
    def __init__(self,
                 video_dir,
                 read_all=False,
                 load_opti_pose=False,
                 fps=30,
                 event_depth_dir=None,
                 voxel_cache_dir=None):
        self.video_dir = video_dir

        self.event_depth_dir = video_dir if event_depth_dir is None else event_depth_dir

        assert os.path.isdir(video_dir), f'Cannot find data directory {video_dir}'
        # self.ho3d_root = '/'.join(self.video_dir.split('/')[:-2])

        with open(os.path.join(self.video_dir, 'obj.txt')) as f:
            self.obj_id = f.readline().strip()

        self.mesh_root = '../EOT6D/simple_mesh'
        self.ho3d_root = 'data/ho3d'
        self.fps = fps

        # self.color_files = sorted(glob.glob(f"{self.video_dir}/rgb/*.jpg"))
        self.depth_files = sorted(glob.glob(f"{self.video_dir}/depth_aligned_to_color/*.png"))
        self.color_files = [file.replace('depth_aligned_to_color', 'rgb').replace('png', 'jpg') for file in self.depth_files]

        # meta_file = self.color_files[0].replace('.jpg', '.pkl').replace('rgb', 'meta')
        # self.K = pickle.load(open(meta_file, 'rb'))['camMat']

        calib_file ='../EOT6D/0001-camchain.yaml'

        K, D, (H_yaml, W_yaml), dist_model = parse_cam0_from_kalibr(calib_file)

        K1, D1, _, _ = parse_cam0_from_kalibr(calib_file, cam0_key="cam1")

        self.K = K
        self.K1 = K1

        # self.T_opti_to_cam = load_extrinsic_yaml('../EOT6D/calib_1014/P_feats_800.yaml')
        # self.T_opti_to_cam1 = load_extrinsic_yaml('../EOT6D/calib_1014/P1_feats_800.yaml')
        self.T_opti_to_cam = load_extrinsic_yaml('../EOT6D/P_feats.yaml')
        self.T_opti_to_cam1 = load_extrinsic_yaml('../EOT6D/P1_feats.yaml')

        # self.T_opti_to_cam = load_extrinsic_yaml('../EOT6D/calib_1013/P_feats.yaml')

        # self.T_opti_to_cam1 = load_extrinsic_yaml('../EOT6D/calib_1013/P1_feats.yaml')

        # self.rgb_to_event = self.T_opti_to_cam1 @ inverse_T(self.T_opti_to_cam)
        with open(calib_file, "r") as f:
            y = yaml.load(f)
            self.rgb_to_event = np.array(y['cam1']['T_cn_cnm1'])

        mask_file = glob.glob(os.path.join(self.video_dir, 'mask' , '*.npy'))[-1]
        self.first_mask = np.load(mask_file)
        if not read_all:
            with open(os.path.join(self.video_dir, 'startend.txt')) as f:
                start_end_index = f.readline().strip()
            self.start, self.end = map(int, start_end_index.split(' '))
        else:
            fname = os.path.basename(mask_file)
            self.start, self.end = int(os.path.splitext(fname)[0]), -4

        if load_opti_pose:
            self.opti_pose = []
            df = pd.read_csv(os.path.join(video_dir, 'aligned_depth_pose.csv'))

            if fps == 30:

                mask = ~pd.isna(df.get("depth_idx"))

                rows = df[mask].copy().drop_duplicates(subset=["depth_idx"], keep="first")

                for _, r in rows.iterrows():
                    px, py, pz = float(r["px"]), float(r["py"]), float(r["pz"])
                    qx, qy, qz, qw = float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"])
                    self.opti_pose.append(pose_to_matrix(px, py, pz, qx, qy, qz, qw))
            elif fps == 120:

                mask = ~pd.isna(df.get("depth_idx"))
                rows = df[mask].copy().sort_index()  # 순서 유지

                prev_depth_idx = None
                buf = []  # 현재 그룹을 임시로 쌓는 버퍼
                self.opti_pose = []  # 필요 시 초기화

                def pick4_indices(n: int):
                    """맨 뒤(n-1) 포함, 가능하면 0 제외. 길이 4로 균등 샘플."""
                    if n <= 4:
                        # 개수 <=4면 끝에서 최대 4개만 사용 (맨 뒤 포함)
                        return np.arange(max(0, n - 4), n, dtype=int)
                    # [0, n-1] 구간을 5등분한 뒤 첫 점(0)을 버리고 4개만 취함
                    # 예: n=8 -> 등분점 [0,1.75,3.5,5.25,7] -> 첫 점 제외 -> [1.75,3.5,5.25,7] -> [1,3,5,7]
                    return np.linspace(0, n - 1, 5, dtype=float)[1:].astype(int)

                def flush_buffer(buf):
                    """버퍼에 쌓인 rows를 길이 4로 (맨 뒤 포함, 0 회피) 균등 샘플해 pose_list로 변환"""
                    n = len(buf)
                    if n == 0:
                        return None
                    idx = pick4_indices(n)
                    chosen = [buf[i] for i in idx]

                    pose_list = []
                    for r in chosen:
                        px, py, pz = float(r["px"]), float(r["py"]), float(r["pz"])
                        qx, qy, qz, qw = float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"])
                        pose_list.append(pose_to_matrix(px, py, pz, qx, qy, qz, qw))
                    return pose_list

                for _, r in rows.iterrows():
                    cur_depth_idx = r["depth_idx"]

                    if prev_depth_idx is None:
                        # 첫 그룹 시작
                        prev_depth_idx = cur_depth_idx
                        buf = [r]
                        continue
                    if cur_depth_idx != prev_depth_idx:
                        # ★ 변경점: depth_idx가 바뀐 '현재 r'까지 포함해서 이전 그룹 flush
                        # buf.append(r)
                        pose_list = flush_buffer(buf)
                        if pose_list is not None:
                            self.opti_pose.append(pose_list)
                        # 새 그룹 시작: 현재 r은 이미 이전 flush에 포함되었으니 비움
                        buf = [r]
                        prev_depth_idx = cur_depth_idx
                    else:
                        # 동일 depth_idx 계속 쌓기
                        buf.append(r)
                # 마지막 그룹 flush
                pose_list = flush_buffer(buf)
                if pose_list is not None:
                    self.opti_pose.append(pose_list)

            else:
                raise NotImplementedError

            self.opti_pose = self.opti_pose[self.start:self.end+1]

        self.color_files = self.color_files[self.start:self.end+1]

        # Online event voxel grids (no model, preprocessing only)
        self.num_bins = 5
        self.ev_height = 720
        self.ev_width = 1280
        self.voxel_grid = ReconVoxelGrid(self.num_bins, self.ev_height, self.ev_width, normalize=False)
        self.recon_voxel_grid = ReconVoxelGrid(self.num_bins, self.ev_height, self.ev_width, normalize=False)

        # voxel cache
        if voxel_cache_dir is not None:
            seq_name = os.path.relpath(video_dir, os.path.dirname(os.path.dirname(video_dir))).replace('/', '_')
            self.voxel_cache_dir = Path(voxel_cache_dir) / seq_name
        else:
            self.voxel_cache_dir = None

    def __len__(self):
        return len(self.color_files)

    def get_video_name(self):
        return self.obj_id

    def get_mask(self, i):
        if i == 0: return self.first_mask
        else: return np.zeros_like(self.first_mask)

    def get_occ_mask(self, i):
        video_name = self.get_video_name()
        index = int(os.path.basename(self.color_files[i]).split('.')[0])
        mask = cv2.imread(f'{self.ho3d_root}/masks_XMem/{video_name}_hand/{index:04d}.png', -1)
        return mask

    def get_reference_mesh(self, ref_view=4):
        video2name = {
            'AP': 'ob_0000011',
            'MPM': 'ob_0000009',
            'SB': 'ob_0000012',
            'SM': 'ob_0000005',
            }
        video_name = self.get_video_name()

        for k in video2name:
            if video_name.startswith(k):
                ob_name = video2name[k]
                break
        mesh = trimesh.load(f'/data/dataset/foundationpose_dataset/ref_view_dir/ycbv/ref_views_{ref_view}/{ob_name}/model/model.obj')
        return mesh

    def get_obj_prompt(self):
        video2name = {
            'AP': 'blue pitcher base',
            'MPM': 'potted meat can',
            'SB': 'white bleach cleanser',
            'SM': 'yellow mustard bottle',
            }
        video_name = self.get_video_name()

        for k in video2name:
            if video_name.startswith(k):
                ob_name = video2name[k]
                break
        return ob_name

    def get_gt_mesh_diamter(self):
        def fast_diameter(pts):
            # 볼록껍질의 꼭짓점(지름은 항상 여기에 있음)
            hull = ConvexHull(pts)
            vertices = pts[hull.vertices]

            # 꼭짓점끼리만 거리 계산 (개수가 훨씬 적음)
            dists = distance.pdist(vertices, 'euclidean')
            return dists.max()

        # gt_diameter = calc_pts_diameter(np.array(self.get_gt_mesh().vertices))
        gt_diameter = fast_diameter(np.array(self.get_gt_mesh().vertices)[::10])
        return gt_diameter

    def get_obj_relative_pose(self):
        obj_list = {
            'AP': '11',  # 019_pitcher_base
            'MPM': '9',  # 010_potted_meat_can
            'SB': '12',  # 021_bleach_cleanser
            'SM': '5',  # 006_mustard_bottle
            }
        video_name = self.get_video_name()
        for k in obj_list:
            if video_name.startswith(k):
                ob_name = obj_list[k]
                break
        pose_dir = f'/data/dataset/foundationpose_dataset/ref_view_dir/ycbv/rel_pose/{ob_name}'
        pose_file = os.path.join(pose_dir, 'fp_to_ho3d_pose.txt')
        rel_fp_to_ho3d = np.loadtxt(pose_file)
        return rel_fp_to_ho3d

    def get_obj_id(self):

        video_name = self.get_video_name()

        return int(video_name[:3])

    def get_obj_name(self):
        video_name = self.get_video_name()
        for k in obj_list:
            if video_name.startswith(k):
                ob_name = obj_list[k]
                break
        return ob_name

    def get_obj_name(self):
        obj_list = {
            'AP': 'AP',  # 10 x 11 x 019_pitcher_base
            'MPM': 'MPM',  # 010_potted_meat_can
            'SB': 'SB',  # 021_bleach_cleanser
            'SM': 'SM',  # 006_mustard_bottle
            }
        video_name = self.get_video_name()
        for k in obj_list:
            if video_name.startswith(k):
                ob_name = obj_list[k]
                break
        return ob_name

    def get_event_raw_img(self, i, j=0):
        event_img_file = self.color_files[i].replace('rgb', 'event_viz4')[:-4] + '_' + str(int(j)) + '.jpg'
        color = cv2.imread(event_img_file)
        return color

    def get_gt_mesh(self):
        mesh = trimesh.load(f'{self.mesh_root}/{self.obj_id}/textured.obj')
        return mesh

    def get_start_depth_event(self, i):
        """RGB depth를 event 카메라 프레임으로 warp해서 반환 (j=0 starting depth)"""
        depth_scale = 0.001
        depth_rgb = cv2.imread(
            self.color_files[i].replace('.jpg', '.png').replace('rgb', 'depth_aligned_to_color'),
            cv2.IMREAD_ANYDEPTH
        ).astype(np.float32) * depth_scale
        depth_ev, _ = warp_depth_rgb_to_event(
            depth_rgb, self.K, self.K1, self.rgb_to_event, out_hw=None
        )
        return depth_ev

    def _load_raw_events(self, i):
        """parsed_events/{rgb_idx+1:06d}.npz 에서 raw events 로드"""
        rgb_idx = int(os.path.basename(self.color_files[i]).split('.')[0])
        event_path = os.path.join(self.video_dir, 'parsed_events', f'{rgb_idx+1:06d}.npz')
        data = np.load(event_path, allow_pickle=True)['data']
        return data['x'], data['y'], data['t'], data['p']

    def get_event_voxels(self, i, j):
        """
        j: IFrameIndex (1~4)
        returns: (start_vox, recon_vox) 각각 torch.Tensor (num_bins, H, W)
          - start_vox: cumulative voxel [t0 → t0 + t_len*(j/4)]
          - recon_vox: interval voxel  [t0 + t_len*((j-1)/4) → t0 + t_len*(j/4)]
        """
        # --- cache lookup ---
        if self.voxel_cache_dir is not None:
            rgb_idx = int(os.path.basename(self.color_files[i]).split('.')[0])
            cache_path = self.voxel_cache_dir / f'{rgb_idx+1:06d}_{j}.npz'
            if cache_path.exists():
                cached = np.load(str(cache_path))
                return torch.from_numpy(cached['start_vox']), torch.from_numpy(cached['recon_vox'])

        x, y, t, p = self._load_raw_events(i)
        h, w = self.ev_height, self.ev_width

        if len(x) == 0:
            zeros = torch.zeros(self.num_bins, h, w)
            return zeros, zeros

        t_len = float(t[-1] - t[0])

        # cumulative voxel: t[0] → t[j/4]
        t_end = t[0] + t_len * j / 4
        mask_cum = (t <= t_end)
        xc, yc, tc, pc = x[mask_cum], y[mask_cum], t[mask_cum], p[mask_cum]
        if len(xc) > 0:
            tc_n = ((tc - tc[0]) / (tc[-1] - tc[0] + 1e-9)).astype('float32')
            start_vox = self.voxel_grid.convert(
                torch.from_numpy(xc.astype('float32')),
                torch.from_numpy(yc.astype('float32')),
                torch.from_numpy(pc.astype('float32')),
                torch.from_numpy(tc_n)
            )
        else:
            start_vox = torch.zeros(self.num_bins, h, w)

        # recon voxel: t[(j-1)/4] → t[j/4]
        t_start_r = t[0] + t_len * (j - 1) / 4
        mask_rec = (t >= t_start_r) & (t <= t_end)
        xr, yr, tr, pr = x[mask_rec], y[mask_rec], t[mask_rec], p[mask_rec]
        if len(xr) > 0:
            tr_n = ((tr - tr[0]) / (tr[-1] - tr[0] + 1e-9)).astype('float32')
            recon_vox = self.recon_voxel_grid.convert(
                torch.from_numpy(xr.astype('float32')),
                torch.from_numpy(yr.astype('float32')),
                torch.from_numpy(pr.astype('float32')),
                torch.from_numpy(tr_n)
            )
        else:
            recon_vox = torch.zeros(self.num_bins, h, w)

        # --- lazy cache save ---
        if self.voxel_cache_dir is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(cache_path),
                                start_vox=start_vox.numpy(),
                                recon_vox=recon_vox.numpy())

        return start_vox, recon_vox

    def get_color(self, i, j=0):
        color = cv2.cvtColor(cv2.imread(self.color_files[i]), cv2.COLOR_BGR2RGB)
        return color

    def get_event_img(self, i, j=0, cv_gray=False):
        event_img_file = self.color_files[i].replace('rgb', 'parsed_e2vid')[:-4] + '_' + str(int(j)) + '.png'
        # event_img_file = self.color_files[i].replace('rgb', 'parsed_e2vid_base')[:-4] + '_' + str(int(j)) + '.png'

        # event_img_file = event_img_file.replace('parsed_e2vid', 'e2vid').replace('data_demo', 'data_demo_extrapolated_depth_ours_demo_8times_epoch100_trial2')
        # event_img_file = event_img_file.replace(os.path.dirname(os.path.dirname(self.video_dir)), self.event_depth_dir)
        # event_img_file = event_img_file.replace('parsed_e2vid', 'e2vid')

        color = cv2.cvtColor(cv2.imread(event_img_file), cv2.COLOR_BGR2RGB)

        if cv_gray:
            gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
            gray_img = np.stack([gray]*3, axis=-1)
            color = gray_img
        return color

    def get_depth(self, i, j=0, frame='rgb'):
        # color = imageio.imread(self.color_files[i])
        depth_scale = 0.001
        if frame == 'rgb':
            depth = cv2.imread(self.color_files[i].replace('.jpg', '.png').replace('rgb', 'depth_aligned_to_color'), cv2.IMREAD_ANYDEPTH)
            depth = depth * depth_scale
        elif frame == 'event':
            depth_aligned_to_event = self.color_files[i].replace('.jpg', '.png').replace('rgb', 'depth_aligned_to_event')
            # depth_aligned_to_event = depth_aligned_to_event.replace('data_v4', 'data_v4_extrapolated_depth')
            # depth_aligned_to_event = depth_aligned_to_event.replace('data_v4', 'data_v4_extrapolated_depth_final')

            depth_aligned_to_event = depth_aligned_to_event.replace(os.path.dirname(os.path.dirname(self.video_dir)), self.event_depth_dir)

            depth_aligned_to_event = depth_aligned_to_event[:-4] + '_' + str(int(j)) + '.png'
            # if os.path.isfile(depth_aligned_to_event):
            depth = cv2.imread(depth_aligned_to_event, cv2.IMREAD_ANYDEPTH)
            depth = depth * depth_scale
            # else:
                # depth, _ = warp_depth_rgb_to_event(depth, self.K, self.K1, self.rgb_to_event, out_hw=None,  bilinear_splat=True)
        return depth

    def get_xyz_map(self, i):
        depth = self.get_depth(i)
        xyz_map = depth2xyzmap(depth, self.K)
        return xyz_map

    def get_gt_pose(self, i):
        meta_file = self.color_files[i].replace('.jpg', '.pkl').replace('rgb', 'meta')
        meta = pickle.load(open(meta_file, 'rb'))
        ob_in_cam_gt = np.eye(4)
        if meta['objTrans'] is None:
            return None
        else:
            ob_in_cam_gt[:3, 3] = meta['objTrans']
            ob_in_cam_gt[:3, :3] = cv2.Rodrigues(meta['objRot'].reshape(3))[0]
            ob_in_cam_gt = glcam_in_cvcam @ ob_in_cam_gt
        return ob_in_cam_gt

    def save_pose(self, i, pred_pose):
        save_path = self.color_files[i].replace('/rgb/', '/pose/').replace('.jpg', '.npy')
        parent = Path(save_path).parent.as_posix()
        os.makedirs(parent, exist_ok=True)
        np.save(save_path, pred_pose)

        return

    def get_gt_pose(self, i, j=0, frame='rgb'):

        pose_path = self.color_files[i].replace('/rgb/', '/pose/').replace('.jpg', '.npy')
        pose = np.load(pose_path) # (4, 4) or (n, 4, 4)

        if len(pose.shape) == 3:
            pose = pose[j]

        if frame=='event':
            pose = self.rgb_to_event @ pose

        return pose

    def get_opti_pose(self, i, j=0):
        if self.fps == 30:
            ob_in_cam = self.T_opti_to_cam @ self.opti_pose[i]
        else:
            ob_in_cam = self.T_opti_to_cam @ self.opti_pose[i][j]
        return ob_in_cam

if __name__=="__main__":

    reader = Event6DReader('../EOT6D/demo_0916/0000')
    gt_mesh = reader.get_gt_mesh()
    gt_diameter = reader.get_gt_mesh_diamter()
    ob_id = reader.get_obj_id()
    mask = reader.get_mask(0).astype(np.bool_)

    for i in range(len(reader)):
        color_file = reader.color_files[i]
        reader.get_opti_pose(i)
        reader.get_depth(i)
        reader.K
