import random
import itertools
import numpy as np
from collections import defaultdict

import torch
import torch.nn.functional as F
from einops import repeat, rearrange
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import Sampler


def custom_collate_fn(batch):
    collated_batch = {}
    
    for key in batch[0]:
        values = [d.get(key, None) for d in batch if key in d]    
    
        # if isinstance(values[0], trimesh.base.Trimesh):
        if key in ('objMesh', 'objMeshTensor'):
            collated_batch[key] = [item for item in values if item is not None]
        else:
            collated_batch[key] = default_collate(values)
    
    # import pdb; pdb.set_trace()
    return collated_batch


def clustering_group(group_indices, M):
    groups = list(group_indices.keys())
    sizes = [(group, len(group_indices[group])) for group in groups]
    sizes.sort(key=lambda x: x[1], reverse=True)  # 큰 것부터
    bins = [[] for _ in range(M)]
    bin_loads = [0] * M
    for k, sz in sizes:
        j = int(np.argmin(bin_loads))
        bins[j].extend(group_indices[k])
        bin_loads[j] += sz
    return bins, bin_loads


class GroupByObjIDSampler(Sampler):
    
    def __init__(self, obj_ids, batch_size, drop_last=False, seed=None):
        
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.obj_ids = list(obj_ids)
        self.rng = random.Random(seed)
        
        self.group2idx = defaultdict(list)
        for idx, oid in enumerate(self.obj_ids):
            self.group2idx[oid].append(idx)
            
    def __iter__(self):
        # 매 epoch마다: 그룹 내부 섞기
        groups = list(self.group2idx.keys())
        self.rng.shuffle(groups)
        for group in groups:
            self.rng.shuffle(self.group2idx[group])
            
        # group clustering
        group_indices, group_element_num = clustering_group(self.group2idx, self.batch_size)
        num_min_elements = min(group_element_num)

        for _ in range(num_min_elements):
            batch_indices = []
            for indices in group_indices:
                if len(indices) > 0:
                    batch_indices.append(indices.pop())
            yield batch_indices

        rest = list(itertools.chain(*group_indices))
        if rest:
            self.rng.shuffle(rest)
            for i in range(0, len(rest), self.batch_size):
                chunk = rest[i:i+self.batch_size]
                if len(chunk) == self.batch_size or not self.drop_last:
                    yield chunk
            
    def __len__(self) -> int:
        n = sum(len(v) for v in self.group2idx.values())
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size