# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import pickle, glob, cv2, imageio, os, sys, pdb, re, json, trimesh, copy, logging, multiprocessing, subprocess, \
    joblib
    
import time, random
from tqdm import tqdm
import numpy as np
import ruamel.yaml
from pathlib import Path
from torch.utils.data import Dataset, get_worker_info

from src.utils.geometry import *
from src.utils.basics import read_pickle, LRUMeshCache
# from src.utils.sampler import project_cad_points
from bop_toolkit_lib.misc import calc_pts_diameter
from src.utils.batchsampler import DistributedGroupBatchSampler
from src.dataloader.collate_fn import custom_collate_fn

from src.dataloader.representations_og import ReconVoxelGrid

def list_index(list, idx):
    return [list[i] for i in idx]

class Blender6DDataset(Dataset):
    def __init__(
        self, 
        root_dir: str, 
        split: str = 'train', 
        window_size: int = 1, 
        stride: int = 1, 
        step_size: int = 1,
        num_obj_points: int = 1024,
        max_cache_size: int = 32,
        return_tensor = False,
        event_root_dir = '',
        event_cache_dir = '',
        *args, **kwargs
    ):
        super().__init__()
        assert split in ['train', 'valid', 'test']
        
        # Required parameters.
        self.split = split
        self.window_size = window_size
        self.stride = stride
        self.step_size = step_size
        self.num_obj_points = num_obj_points
        self.max_cache_size = max_cache_size
        self.root_dir = Path(root_dir)
        self.ev = os.path.isdir(event_root_dir)
        if self.ev: 
            print("Start Event Loader!")
            self.event_root_dir = Path(event_root_dir)
            self.ev_shape = (480, 640)
            # 5-bin ReconVoxelGrid (matching inference in my_reader_online.py)
            self.converter = ReconVoxelGrid(5, *self.ev_shape, normalize=False)
            
            # # Object to convert raw event data into grid representations
            # self.converter = MixedDensityEventStack(
            #     image_shape=self.ev_shape,
            #     num_stacks=num_bins,
            #     interpolation='bilinear',
            #     channel_overlap=True,
            #     centered_channels=False
            # )
        
        self.event_cache_dir = Path(event_cache_dir) if event_cache_dir else None
        self.model_dir = self.root_dir / 'gso'
        self.return_tensor = return_tensor
        # self.gs_model_dir = self.root_dir / 'gso_gs_models'
        self.worker_cache = {}  # Mesh cache for each worker
        
        # additional parameters
        self.kwargs = kwargs
        
        start = time.time()
        # import pdb; pdb.set_trace()
        self.scene_info, self.obj_ids = self.process_subdir_fast(categories=['easy'])
        end = time.time()
        print("Elapsed Reading:", end - start, "seconds")
        
    def __len__(self):
        return len(self.scene_info)
    
    def process_subdir_fast(self, categories=['easy', 'medium', 'hard']):
        
        def gather(data, indices):
            return [data[i] for i in indices]
        
        scene_info_all, obj_ids = [], []
        
        with open(self.root_dir / f'{self.split}.txt', 'r') as f:
            scene_lists = f.readlines()
        scene_lists = [self.root_dir / s.strip() for s in scene_lists if (s.strip()).split('_')[1] in categories]
        
        for scene_dir in scene_lists:
            ev_dir = Path(str(scene_dir).replace(str(self.root_dir), str(self.event_root_dir)).replace('train_pbr/', ''))
            ev_scene_dir = ev_dir.name

            color_files = sorted(list(scene_dir.glob('rgb/*.png')))
            depth_files = sorted(list(scene_dir.glob('depth/*.png')))
            # import pdb; pdb.set_trace()
            if self.ev: 
                event_files = [None] + sorted(list(Path(str(scene_dir).replace(str(self.root_dir), str(self.event_root_dir)).replace('train_pbr/', '')).glob('*.npz')))
                if len(event_files) != len(color_files):
                    min_len = min(len(event_files), len(color_files))
                    event_files = event_files[:min_len]
                    color_files = color_files[:min_len]
                    depth_files = depth_files[:min_len]
            with open(scene_dir / 'scene_gt.json', 'r') as f:
                object_pose_info = json.load(f)  # include cam_R_m2c, cam_t_m2c, obj_id
            with open(scene_dir / 'scene_camera.json', 'r') as f:
                camera_info = json.load(f)  # include cam_K, cam_R_w2c, cam_t_w2c, depth_scale
            num_object = len(object_pose_info['0'])
            num_frames = len(color_files)
            
            cam_intrinsic, cam_extrinsic, depth_scale = {}, {}, {}
            for key in range(num_frames):
                key = str(key)
                cam_Rt = np.eye(4)
                cam_Rt[:3, :3] = np.array(camera_info[key]['cam_R_w2c']).reshape(3, 3)
                cam_Rt[:3, 3] = np.array(camera_info[key]['cam_t_w2c']).reshape(3) / 1e3  # convert to meters
                cam_intrinsic[key] = np.array(camera_info[key]['cam_K']).reshape(3, 3).astype(np.float32)
                cam_extrinsic[key] = cam_Rt.astype(np.float32)
                depth_scale[key] = camera_info[key]['depth_scale']
            
            for i in range(num_object):   
                start_idx = 1 if self.ev else 0
                # import pdb; pdb.set_trace()             
                for start in range(start_idx, len(color_files) - (self.window_size - 1) * self.stride, self.step_size):
                    scene_info = dict()
                    idxs = range(start , start + self.window_size * self.stride , self.stride)
                    scene_info['color_files'] = gather(color_files, idxs)
                    if self.ev:
                        # try:
                        ev_clip = gather(event_files, range(start+1 , start + (self.window_size-1) * self.stride + 1, 1))
                        # except: import pdb; pdb.set_trace()
                        chunks = [ev_clip[i:i+self.stride] for i in range(0, len(ev_clip), self.stride)]
                        scene_info['event_files'] = chunks
                    scene_info['ev_scene_dir'] = ev_scene_dir
                    scene_info['window_start'] = start
                        
                    scene_info['depth_files'] = gather(depth_files, idxs)
                    scene_info['mask_files'] = []
                    scene_info['object_w2c'], scene_info['cam_w2c'], scene_info['cam_K'], scene_info['depth_scale'] = [], [], [], []
                    scene_info['obj_id'] = object_pose_info['0'][i]['obj_id']
                    for key in idxs:
                        key = str(key)
                        obj_Rt = np.eye(4)
                        obj_Rt[:3, :3] = np.array(object_pose_info[key][i]['cam_R_m2c']).reshape(3, 3)
                        obj_Rt[:3, 3] = np.array(object_pose_info[key][i]['cam_t_m2c']).reshape(3) / 1e3  # convert to meters
                        scene_info['object_w2c'].append(obj_Rt.astype(np.float32))
                        scene_info['cam_w2c'].append(cam_extrinsic[key])
                        scene_info['cam_K'].append(cam_intrinsic[key])
                        scene_info['depth_scale'].append(depth_scale[key])
                        
                        mask_path = scene_dir / 'mask_visib' / '{:06d}_{:06d}.png'.format(int(key), i)
                        scene_info['mask_files'].append(mask_path)

                    scene_info_all.append(scene_info)
                    obj_ids.append(scene_info['obj_id'])

        return scene_info_all, obj_ids

    def events_to_voxel_grid(self,x, y, p, t):
        t = (t - t[0]).astype('float32')
        t = (t/t[-1])
        x = x.astype('float32')
        y = y.astype('float32')
        pol = p.astype('float32')
        
        return self.converter.convert(
                torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(pol),
                torch.from_numpy(t))
    
    def get_depth(self, file_path, depth_scale):
        # import pdb; pdb.set_trace()
        depth = cv2.imread(file_path, -1)
        depth = depth * depth_scale / 1e3  # convert to meters
        return depth.astype(np.float32)
    
    def get_object_mask(self, file_path, target_size):
        mask = cv2.imread(file_path, -1)
        mask = cv2.resize(mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
            
        # erode mask to make conservative mask
        kernel = np.ones((5, 5), np.uint8)
        eroded_mask = cv2.erode(mask, kernel, iterations=1)
            
        return mask.astype(np.bool_), eroded_mask.astype(np.bool_)
    
    def preprocess_scene(self, trimesh_obj):
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(trimesh_obj.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(trimesh_obj.faces)
        o3d_mesh.triangle_normals = o3d.utility.Vector3dVector(trimesh_obj.face_normals.copy())
        
        # for ray-tracing
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh, device=o3d.core.Device("CPU:0"))
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh_t)
        
        return scene
    
    def get_mesh_diameter(self, mesh):
        gt_diameter = calc_pts_diameter(np.array(mesh.vertices))
        # gt_diameter = torch.tensor(gt_diameter, dtype=torch.float32)
        return gt_diameter
    
    def mesh_loader(self, obj_id, return_tensor=True):
        mesh = trimesh.load(str(self.model_dir / obj_id / 'visual_geometry.obj'), process=False)
        if return_tensor:
            mesh_tensors = make_mesh_tensors(mesh, device='cpu')
        else:
            mesh_tensors = None
        mesh_diameter = self.get_mesh_diameter(mesh)
        
        return mesh, mesh_tensors, mesh_diameter

    def get_normalized_points(self, points, mesh):
        """
        Normalize points to the range [-1, 1] based on the mesh bounding box.
        """
        min_bound = mesh.vertices.min(axis=0)
        max_bound = mesh.vertices.max(axis=0)
        center = (min_bound + max_bound) / 2.0
        scale = np.linalg.norm(max_bound - min_bound) / 2.0
        
        normalized_points = (points - center) / scale
        return normalized_points
    
    
    def get_event(self, event_list, max_split=None):
        
        # import pdb; pdb.set_trace()
        # event_list = self.event_files[max(0, i-stride+1): i+1]
        
        ev_list = []
        for file in event_list:
            # if os.path.isfile(file):
            event = np.load(file ,mmap_mode='r')['data']
            split = True
            # else: 
            #     event = np.zeros((0, 4))
            #     split = False
            
            event_x = event[:, 0]
            event_y = event[:, 1]
            event_p = event[:, 2]
            event_t = event[:, 3]
            
            mask = (event_x < self.ev_shape[1]) & (event_x >= 0) & (event_y < self.ev_shape[0]) & (event_y >= 0)
            event_t = event_t[mask]
            event_x = event_x[mask].astype('uint16')
            event_y = event_y[mask].astype('uint16')
            event_p = event_p[mask]
            # events = np.stack([event_x,
            #                 event_y,
            #                     event_t,
            #                     event_p], axis=1)
            events = np.stack([event_y,
                            event_x,
                                event_t,
                                event_p], axis=1)
            ev_list.append(events)
        
        all_events = np.concatenate(ev_list)    
        
        # import pdb; pdb.set_trace()

        num_events = 60000
        # num_events = 1e9 # Force ev_split=1
        if split:
            ev_split = (len(all_events) // num_events)+1
            chunk = np.array_split(all_events, ev_split)
        else:
            ev_split = 0    
            chunk = np.array_split(all_events, 1)

        if max_split is not None:
            chunk = chunk[-max_split:]
        
        event_representations = []
        for ev_chunk in chunk:
            # i_start = max(len(ev_chunk) - num_events, 0)    
            # events = ev_chunk[i_start:]
            events = ev_chunk
            ev_repr = self.converter(events)
            event_representations.append(ev_repr)
        
        voxels = np.stack(event_representations, axis=0)
        # import pdb; pdb.set_trace()
        # return  voxels, ev_split
        return  voxels
    
    def _to_voxel(self, ev):
        # ev columns: [x, y, p, t]
        x = torch.from_numpy(ev[:, 0].astype(np.float32))
        y = torch.from_numpy(ev[:, 1].astype(np.float32))
        p = torch.from_numpy(ev[:, 2].astype(np.float32))
        t = ev[:, 3].astype(np.float64)
        t = ((t - t[0]) / (t[-1] - t[0] + 1e-9)).astype(np.float32)
        return self.converter.convert(x, y, p, torch.from_numpy(t))

    def _get_or_make_recon_voxel(self, event_file_path):
        """Single-event recon voxel, cached at EvBlenderProcReconVoxel/<scene>/<file>.npz.
        Compatible with Stage 1's cache layout — both stages share this directory."""
        cache_path = Path(str(event_file_path).replace('EvBlenderProcEv', 'EvBlenderProcReconVoxel'))
        if cache_path.exists():
            try:
                return torch.from_numpy(np.load(str(cache_path))['vox'])
            except Exception:
                cache_path.unlink(missing_ok=True)
        # miss — compute + save
        raw = np.load(str(event_file_path), mmap_mode='r')['data']
        voxel = self._to_voxel(raw)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        vox_np = voxel.cpu().numpy() if hasattr(voxel, 'cpu') else voxel
        np.savez_compressed(str(cache_path), vox=vox_np)
        return voxel

    def get_event_dual(self, files):
        event_files_flatten = [file for sub in files for file in sub]

        # recon: stride=1 → single-file voxel (per-file cache, shared with Stage 1)
        #        stride>1 → multi-file concat (compute fresh, no shared cache)
        if self.stride == 1:
            recon_voxel = self._get_or_make_recon_voxel(event_files_flatten[-1])
            # only need to mmap-load events for depth
            all_events = [np.load(f, mmap_mode='r')['data'] for f in event_files_flatten]
        else:
            all_events = [np.load(f, mmap_mode='r')['data'] for f in event_files_flatten]
            recon_voxel = self._to_voxel(np.concatenate(all_events[-self.stride:], axis=0))

        # depth: concat ALL events in window (per-window cache handled by caller)
        depth_voxel = self._to_voxel(np.concatenate(all_events, axis=0))

        return recon_voxel, depth_voxel
    
    def __getitem__(self, index):
        
        if isinstance(index, tuple) and len(index) == 2 and isinstance(index[1], (bool, np.bool_)):
            idx, is_special = index
        else:
            idx, is_special = index, False
        
        
        # cache assign (for each worker)
        worker_info = get_worker_info()
        wid = worker_info.id if worker_info else f"pid_{os.getpid()}"
        # print(wid)
        if wid not in self.worker_cache:
            self.worker_cache[wid] = LRUMeshCache(maxsize=self.max_cache_size)

        scene_info = self.scene_info[idx]
        
        start_idx = random.randint(0, self.window_size-2)
        end_idx = self.window_size-1
        se = (start_idx, end_idx)
        # read color images
        color_files = list_index(scene_info['color_files'], se)
        colors = np.stack([cv2.cvtColor(cv2.imread(str(color_file)), cv2.COLOR_BGR2RGB) for color_file in color_files])
        target_size = colors.shape[1:3]  # (H, W)
        
        # read depth images
        depth_files = list_index(scene_info['depth_files'], se)
        depth_scales = scene_info['depth_scale']
        depths = np.stack([self.get_depth(str(depth_file), depth_scale) for depth_file, depth_scale in zip(depth_files, depth_scales)])
        
        # read mask images
        mask_files = list_index(scene_info['mask_files'], se)
        masks, masks_eroded = [], []
        for mask_file in mask_files:
            mask, mask_eroded = self.get_object_mask(str(mask_file), target_size)
            masks.append(mask)
            # masks_eroded.append(mask_eroded)
        masks = np.stack(masks)
        # masks_eroded = np.stack(masks_eroded)
        # import pdb; pdb.set_trace()
        # from utils.visualizer import paint_point_track, export_to_gif
        # export_to_gif(np.uint8(colors), 'color.gif', 10)
        # export_to_gif(np.uint8(depths / depths.max() * 255), 'depth.gif', 10)
        # export_to_gif(np.uint8(masks * 255), 'masks.gif', 10)

        
        # read intrinsics and object poses
        object_w2cs = np.stack(list_index(scene_info['object_w2c'], se), axis=0)
        # camera_extrinsics = np.stack(scene_info['cam_w2c'], axis=0)
        camera_intrinsics = np.stack(list_index(scene_info['cam_K'], se), axis=0)
        
        # convert to tensors
        colors = torch.tensor(colors, dtype=torch.float32).permute(0, 3, 1, 2)  # (T, 3, H, W)
        depths = torch.tensor(depths, dtype=torch.float32)  # (T, H, W)
        masks = torch.tensor(masks, dtype=torch.bool)  # (T, H, W)
        # masks_eroded = torch.tensor(masks_eroded, dtype=torch.bool)  # (T, H, W)
        intrinsics = torch.tensor(camera_intrinsics, dtype=torch.float32)
        objPoses = torch.tensor(object_w2cs, dtype=torch.float32)
        # camPoses = torch.tensor(camera_extrinsics, dtype=torch.float32)
                
        ret = {
            'color': colors,  # (T, H, W, 3)
            'depth': depths,  # (T, H, W)
            'mask': masks,  # (T, H, W)
            # 'eroded_mask': masks_eroded,  # (T, H, W)
            # 'pts3d': points,  # (P, 3)
            # 'pts3d_norm': points_norm,  # (P, 3)
            # 'face_indices': torch.tensor(face_indices, dtype=torch.long),  # (P,)
            'camMat': intrinsics,  # (T, 3, 3)
            'objPose': objPoses,  # (T, 4, 4)
            'objMeshID': scene_info['obj_id'],
            }
        
        
        if is_special:
            # read mesh and scene
            cad_mesh, cad_mesh_tensors, cad_mesh_diameter = self.worker_cache[wid].get(scene_info['obj_id'], self.mesh_loader, return_tensor=self.return_tensor)
            
            ret.update({
                # 'objMesh': cad_mesh,  # mesh object
                'objMesh': None,  # mesh object
                'objMeshDm': torch.tensor(cad_mesh_diameter, dtype=torch.float32),  # mesh diameter
            })
            
            if self.return_tensor:
                ret.update({
                        'objMeshTensor': cad_mesh_tensors,  # mesh tensors
                        })
        
        
        # read event files and process
        if self.ev:
            event_files = list_index(scene_info['event_files'], range(*se)) # list of list of ev_path

            if self.event_cache_dir is not None:
                # depth-only cache (per-window). Recon is cached per-file inside get_event_dual
                # at EvBlenderProcReconVoxel/<scene>/<file>.npz (shared with Stage 1).
                depth_cache_path = (self.event_cache_dir
                                    / scene_info['ev_scene_dir']
                                    / f"w{scene_info['window_start']:06d}_s{start_idx}_depth.npz")
                depth_loaded = False
                if depth_cache_path.exists():
                    try:
                        depth_ev = torch.from_numpy(np.load(str(depth_cache_path))['vox'])
                        depth_loaded = True
                    except Exception:
                        depth_cache_path.unlink(missing_ok=True)

                if depth_loaded:
                    # recon still computed (or cache-hit) per-file
                    event_files_flatten = [f for sub in event_files for f in sub]
                    if self.stride == 1:
                        recon_ev = self._get_or_make_recon_voxel(event_files_flatten[-1])
                    else:
                        all_events = [np.load(f, mmap_mode='r')['data'] for f in event_files_flatten]
                        recon_ev = self._to_voxel(np.concatenate(all_events[-self.stride:], axis=0))
                else:
                    recon_ev, depth_ev = self.get_event_dual(event_files)
                    depth_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    depth_np = depth_ev.cpu().numpy() if hasattr(depth_ev, 'cpu') else depth_ev.numpy()
                    np.savez_compressed(str(depth_cache_path), vox=depth_np)
            else:
                recon_ev, depth_ev = self.get_event_dual(event_files)

            ret.update({
                        'recon_event': recon_ev,
                        'depth_event': depth_ev
                        })
            
        return ret
    
    
    
def get_dataset(name: str, *args, **kwargs):
    if name == 'blender6d':
        return Blender6DDataset(*args, **kwargs)
    else:
        raise ValueError(f"Dataset {name} not supported.")
    
    
    
if __name__ == '__main__':
    
    from src.dataloader.collate_fn import custom_collate_fn, GroupByObjIDSampler
    
    # Example usage
    g = torch.Generator(); g.manual_seed(1234)
    dataset = get_dataset('blender6d', '/mnt7/EvBlenderProc', 
                          event_root_dir='/data3/dataset/EvBlenderProcEv', 
                          split='train', window_size=2, stride=2, num_obj_points=1024, max_cache_size=0)
    sampler = DistributedGroupBatchSampler(dataset.obj_ids, local_batch_size=8, drop_last=False, pad_to_full=True, seed=1234)
    print(f"Dataset length: {len(dataset)}")
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        generator=g, 
        # pin_memory=True,
        # persistent_workers=True,
        pin_memory=False,
        persistent_workers=False,
        collate_fn=custom_collate_fn,
        batch_sampler=sampler
    )
    
    # import pdb; pdb.set_trace()
    for idx, sample in enumerate(tqdm(dataloader)):
        if idx % 100 == 0:
            print(sample['objMeshID'][0])
        continue