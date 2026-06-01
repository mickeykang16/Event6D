from pathlib import Path
import weakref

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from .representations_og import VoxelGrid, ReconVoxelGrid
from .eventslicer import EventSlicer
import PIL.Image
from . import common
import os
import yaml
import json
import random

class blenderSequence(Dataset):
    # NOTE: This is just an EXAMPLE class for convenience. Adapt it to your case.
    # In this example, we use the voxel grid representation.
    #
    # This class assumes the following structure in a sequence directory:
    #
    # seq_name (e.g. zurich_city_11_a)
    # ├── disparity
    # │   ├── event
    # │   │   ├── 000000.png
    # │   │   └── ...
    # │   └── timestamps.txt
    # └── events
    #     ├── left
    #     │   ├── events.h5
    #     │   └── rectify_map.h5
    #     └── right
    #         ├── events.h5
    #         └── rectify_map.h5

    def __init__(self, seq_path: Path, mode: str='train', delta_t_ms: int=50, num_bins: int=5, use_events: bool=True, time_interval=5,
                nr_events_data: int = 4
                ):
        assert num_bins >= 1
        assert delta_t_ms <= 100, 'adapt this code, if duration is higher than 100 ms'

        assert seq_path.is_dir()

        # NOTE: Adapt this code according to the present mode (e.g. train, val or test).
        self.mode = mode

        self.use_events = use_events

        # Save output dimensions
        # self.height = 625
        # self.width = 970
        self.height = 480
        self.width = 640
        self.num_bins = num_bins
        self.time_interval = time_interval

        self.nr_events_data = nr_events_data
        self.recon_num_bins = num_bins

        # Set event representation
        self.recon_voxel_grid = ReconVoxelGrid(self.recon_num_bins, self.height, self.width, normalize=False)

        # seq_path -> event_path

        self.seq_path = seq_path
        # Save delta timestamp in ms
        self.delta_t_us = delta_t_ms * 1000

        with open(seq_path / 'scene_camera.json', 'r') as f:
            self.camera_info = json.load(f)  # include cam_K, cam_R_w2c, cam_t_w2c, depth_scale

        with open(seq_path / 'scene_gt.json', 'r') as f:
            self.object_pose_info = json.load(f)  # include cam_R_m2c, cam_t_m2c, obj_id
        self.num_object = len(self.object_pose_info['0'])

        # depth_scale[key] = camera_info[key]['depth_scale']

        self.image_dir = Path(seq_path) / 'depth'
        self.mask_dir = Path(seq_path) / 'mask_visib'

        replace_dir = 'EvBlenderProc'
        event_dir = Path(str(seq_path).replace(replace_dir, replace_dir + 'Ev').replace('/train_pbr', ''))
        self.replace_dir = replace_dir

        assert event_dir.is_dir()
        event_pathstrings = list()

        for entry in event_dir.iterdir():
            if str(entry.name).endswith('.npz'):
                event_pathstrings.append(str(entry))
        event_pathstrings.sort()

        event_result = []

        if self.mode == 'train':
            event_result = []
            for i in range(0,len(event_pathstrings),4):
                if i + 4 < len(event_pathstrings):
                    event_result.append(event_pathstrings[i:i+4])
            self.event_pathstrings = event_result
        else:
            event_result = []
            for i in range(0,len(event_pathstrings),5):
                if i + 5 < len(event_pathstrings):
                    event_result.append(event_pathstrings[i:i+5])
            self.event_pathstrings = event_result

        # k = len(self.event_pathstrings) // 10  # 리스트 길이의 1/10 (내림)

        # self.event_pathstrings = random.sample(self.event_pathstrings, k)  # 중복 없이 k개 추출
        self.event_pathstrings = self.event_pathstrings[::10]

    def events_to_voxel_grid(self,x, y, p, t):
        t = (t - t[0]).astype('float32')
        t = (t/t[-1])
        x = x.astype('float32')
        y = y.astype('float32')
        pol = p.astype('float32')

        return self.recon_voxel_grid.convert(
                torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(pol),
                torch.from_numpy(t))

    def get_depth(self, file_path, depth_scale):
        depth = cv2.imread(file_path, -1)
        depth = depth * depth_scale / 1e3  # convert to meters
        return depth.astype(np.float32)

    def get_object_mask(self, file_path, target_size):
        mask = cv2.imread(file_path, -1)
        mask = cv2.resize(mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)

        # return mask.astype(np.float)/255.0
        return mask.astype(float)/255.0

    def get_image(self, color_file):
        # assert filepath.is_file()
        # image = imageio.imread(str(filepath), pilmode='RGB')
        image = np.array(cv2.cvtColor(cv2.imread(color_file), cv2.COLOR_BGR2RGB)).astype(np.uint8)/255.0
        return image

    def get_event(self, filepath: Path):
        assert filepath.is_file()
        # x, y, t, p = get_event
        # events = np.load(filepath)
        # x, y, t, p = events[:, 1], events[:, 2], events[:, 0], events[:, 3]
        event = np.load(filepath, allow_pickle=True)['data']

        x = event[:, 0]
        y = event[:, 1]
        p = event[:, 2]
        t = event[:, 3]

        return x, y, t, p

    def __len__(self):

        return len(self.event_pathstrings) // 2
        # return 5

    def __getitem__(self, index):
        # ts_end = self.timestamps[index]
        # ts_start should be fine (within the window as we removed the first disparity map)
        # ts_start = ts_end - self.delta_t_us

        IFrameIndexs = [1, 2, 3, 4]

        IFrameIndexs = [random.choice(IFrameIndexs)]
        outputs = []
        for IFrameIndex in IFrameIndexs:
            frameRange = [0, IFrameIndex]
            returnIndex = IFrameIndex - 1

            sample = []
            sample_img = []
            sample_mask = []
            start_path = self.image_dir / (str(int(self.event_pathstrings[index][0].split('/')[-1].split('.')[0])
                                    -1).zfill(6) + '.png')
            start_mask_path = self.mask_dir / (str(int(self.event_pathstrings[index][0].split('/')[-1].split('.')[0])
                                    -1).zfill(6) + '.png')
            start_idx = str(int(self.event_pathstrings[index][0].split('/')[-1].split('.')[0])-1)

            depth_scale = self.camera_info[start_idx]['depth_scale']
            rgb = self.get_depth(str(start_path), depth_scale)[:, :, np.newaxis]
            img = self.get_image(str(start_path).replace('depth', 'rgb'))

            target_size = rgb.shape[0:2]
            object_idx = random.randrange(self.num_object)
            mask = self.get_object_mask(str(start_mask_path).replace('.png', '_' + str(object_idx).zfill(6) + '.png'), target_size)[:, :, np.newaxis]

            sample_img.append(common.image2tensor(img))
            sample_mask.append(common.image2tensor(mask))
            sample.append(common.image2tensor(rgb))

            gt_path = self.image_dir / (str(int(self.event_pathstrings[index][0].split('/')[-1].split('.')[0])
                                    -1 + IFrameIndex).zfill(6) + '.png')
            gt_mask_path = self.mask_dir / (str(int(self.event_pathstrings[index][0].split('/')[-1].split('.')[0])
                                    -1 + IFrameIndex).zfill(6) + '.png')
            gt_idx = str(int(self.event_pathstrings[index][0].split('/')[-1].split('.')[0])-1+IFrameIndex)
            depth_scale = self.camera_info[gt_idx]['depth_scale']

            rgb = self.get_depth(str(gt_path), depth_scale)[:, :, np.newaxis]
            img = self.get_image(str(gt_path).replace('depth', 'rgb'))
            mask = self.get_object_mask(str(gt_mask_path).replace('.png', '_' + str(object_idx).zfill(6) + '.png'), target_size)[:, :, np.newaxis]

            sample_img.append(common.image2tensor(img))
            sample_mask.append(common.image2tensor(mask))
            sample.append(common.image2tensor(rgb))

            # print(start_path)
            # print(gt_path)

            voxel_path = self.event_pathstrings[index][frameRange[1]-1].replace('EvBlenderProcEv', 'EvBlenderProcDepthVoxel')
            Path(voxel_path).parent.mkdir(parents=True, exist_ok=True)

            while True:
                try:
                    # if not os.path.exists(voxel_path):
                    if True:
                        start_event_p = []
                        start_event_t = []
                        start_event_x = []
                        start_event_y = []
                        for eventIndex in range(frameRange[0], frameRange[1]):
                            # print(self.event_pathstrings[index][eventIndex])
                            x, y, t, p = self.get_event(Path(self.event_pathstrings[index][eventIndex]))
                            start_event_p.append(p)
                            start_event_t.append(t)
                            start_event_x.append(x)
                            start_event_y.append(y)
                        if len(start_event_t) > 0:
                            se_p = np.concatenate(start_event_p, axis=0)
                            se_t = np.concatenate(start_event_t, axis=0)
                            se_x = np.concatenate(start_event_x, axis=0)
                            se_y = np.concatenate(start_event_y, axis=0)
                            try:
                                start_event_representation = self.events_to_voxel_grid(se_x, se_y, se_p, se_t)
                            except:
                                print("except voxel")
                                start_event_representation = torch.zeros(self.num_bins, self.height, self.width)
                        else:
                            start_event_representation = torch.zeros(self.num_bins, self.height, self.width)
                        np.savez_compressed(voxel_path, vox=start_event_representation)
                        break
                    else:
                        try:
                            start_event_representation = torch.tensor(np.load(voxel_path, allow_pickle=True)['vox'])
                            if start_event_representation.shape[0] == self.num_bins:
                                break
                            else:
                                p = Path(voxel_path)
                                p.unlink()
                        except:
                            p = Path(voxel_path)
                            p.unlink()
                except:
                    pass

            voxel_path = self.event_pathstrings[index][frameRange[1]-1].replace('EvBlenderProcEv', 'EvBlenderProcReconVoxel')

            Path(voxel_path).parent.mkdir(parents=True, exist_ok=True)

            while True:
                try:
                    if not os.path.exists(voxel_path):
                        # for eventIndex in range(frameRange[0], frameRange[1]):

                        eventIndex = returnIndex

                        x, y, t, p = self.get_event(Path(self.event_pathstrings[index][eventIndex]))
                        # try:

                        if len(x) > 0:
                            recon_event_representation = self.events_to_voxel_grid(x, y, p, t)
                        else:
                            recon_event_representation = torch.zeros(self.num_bins, self.height, self.width)
                        np.savez_compressed(voxel_path, vox=recon_event_representation)
                        break
                    else:
                        try:
                            recon_event_representation = torch.tensor(np.load(voxel_path, allow_pickle=True)['vox'])
                            if recon_event_representation.shape[0] == self.num_bins:
                                break
                            else:
                                p = Path(voxel_path)
                                p.unlink()
                        except:
                            p = Path(voxel_path)
                            p.unlink()
                except:
                    pass

            output = {}
            output['imgs'] = sample
            output['rgbs'] = sample_img
            output['masks'] = sample_mask
            output['frameRange'] = frameRange
            output['returnIndex'] = returnIndex
            output['depth_scale'] = depth_scale

            # output['total_event'] = torch.tensor(self.get_event(Path(self.event_pathstrings[index][-1]))['total'])

            output['start_event'] = start_event_representation
            output['recon_event'] = recon_event_representation

            output['gt_path'] = str(gt_path).split(self.replace_dir)[-1]

            outputs.append(output)

        return outputs

