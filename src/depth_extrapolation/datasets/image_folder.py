"""Image-folder dataset registry for Stage 1 (EventVFI) pretraining.

Only the `blender-train-wo-mask-recon-single` register is needed for training the
released depth-extrapolation checkpoint.
"""
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .datasets import register
from .sequence_blender_no_mask_e2vid_single import blenderSequence as blenderSequenceNoMaskReconSingle

@register('blender-train-wo-mask-recon-single')
class ImageFolder(Dataset):
    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none', mode=None, use_events=None):
        self.repeat = repeat
        self.cache = cache
        dataset_path = Path(root_path)
        assert dataset_path.is_dir(), str(dataset_path)

        with open(os.path.join(root_path, f'{mode}.txt'), 'r') as f:
            scene_lists = f.readlines()
        scene_lists = [Path(os.path.join(root_path, s.strip())) for s in scene_lists]

        train_sequences = [blenderSequenceNoMaskReconSingle(child, mode) for child in scene_lists]
        self.train_dataset = torch.utils.data.ConcatDataset(train_sequences)

    def get_train_dataset(self):
        return self.train_dataset

    def __len__(self):
        return len(self.train_dataset)
