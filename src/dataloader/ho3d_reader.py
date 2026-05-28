"""HO3D-v2 evaluation-set reader used by `run_ho3d_tracking_120fps_online.py`.

The HO3D-v2 layout assumed under `video_dir`'s grandparent is::

    {ho3d_root}/
        evaluation/<seq>/rgb,depth,meta,...      # original HO3D-v2
        evaluation_events/<seq>/000.npz,...      # simulated events (this release)
        masks_XMem/<seq>/00000.png,...           # XMem object masks (this release)
        ycb_models/{obj}/textured_simple.obj     # YCB-Video meshes (user-provided)

`YCB_MODELS_PATH` env var overrides the `{ho3d_root}/ycb_models` lookup.
"""
import glob
import os
import pickle

import cv2
import numpy as np
import trimesh

from bop_toolkit_lib.misc import calc_pts_diameter
from Utils import glcam_in_cvcam

HO3D_VIDEO_TO_OBJECT = {
    'AP': ('11', '019_pitcher_base'),
    'MPM': ('9', '010_potted_meat_can'),
    'GPMF': ('9', '010_potted_meat_can'),
    'SB': ('12', '021_bleach_cleanser'),
    'SM': ('5', '006_mustard_bottle'),
}

def _resolve_object(video_name):
    """Map a HO3D-v2 sequence name (e.g. SM1, AP10) to (obj_id_str, ycb_folder_name)."""
    for prefix, (obj_id, ycb_name) in HO3D_VIDEO_TO_OBJECT.items():
        if video_name.startswith(prefix):
            return obj_id, ycb_name
    raise ValueError(f'Unknown HO3D-v2 sequence prefix: {video_name}')

class Ho3dReader:
    def __init__(self, video_dir):
        self.video_dir = video_dir
        self.ho3d_root = '/'.join(video_dir.split('/')[:-2])
        self.color_files = sorted(glob.glob(f'{video_dir}/rgb/*.jpg'))
        if not self.color_files:
            raise FileNotFoundError(f'No RGB frames under {video_dir}/rgb/')

        with open(self.color_files[0].replace('.jpg', '.pkl').replace('rgb', 'meta'), 'rb') as f:
            self.K = pickle.load(f)['camMat']

        self.event_files = [
            f.replace('evaluation', 'evaluation_events').replace('rgb/', '').replace('jpg', 'npz')
            for f in self.color_files
        ]
        self.ev_shape = (480, 640)
        self.id_strs = [os.path.basename(f).split('.')[0] for f in self.color_files]

    def __len__(self):
        return len(self.color_files)

    def get_video_name(self):
        return os.path.dirname(os.path.abspath(self.color_files[0])).split('/')[-2]

    def get_obj_id(self):
        return _resolve_object(self.get_video_name())[0]

    def get_gt_mesh(self):
        ycb_name = _resolve_object(self.get_video_name())[1]
        ycb_root = os.environ.get('YCB_MODELS_PATH', f'{self.ho3d_root}/ycb_models')
        return trimesh.load(f'{ycb_root}/{ycb_name}/textured_simple.obj')

    def get_gt_mesh_diamter(self):
        return calc_pts_diameter(np.array(self.get_gt_mesh().vertices))

    def get_mask(self, i):
        """XMem object mask at index `i`. Returns a boolean H×W array."""
        video_name = self.get_video_name()
        index = int(os.path.basename(self.color_files[i]).split('.')[0])
        mask_path = f'{self.ho3d_root}/masks_XMem/{video_name}/{index:05d}.png'
        mask = cv2.imread(mask_path, -1)
        if mask is None:
            raise FileNotFoundError(f'Missing mask: {mask_path}')
        return mask

    def get_gt_pose(self, i):
        """Parse GT pose from `meta/{i}.pkl`; returns None when the meta is degenerate."""
        meta_file = self.color_files[i].replace('.jpg', '.pkl').replace('rgb', 'meta')
        with open(meta_file, 'rb') as f:
            meta = pickle.load(f)
        if meta['objTrans'] is None:
            return None
        ob_in_cam_gt = np.eye(4)
        ob_in_cam_gt[:3, 3] = meta['objTrans']
        ob_in_cam_gt[:3, :3] = cv2.Rodrigues(meta['objRot'].reshape(3))[0]
        return glcam_in_cvcam @ ob_in_cam_gt
