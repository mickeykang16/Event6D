from easydict import EasyDict as edict
import torch
import random
import cv2
import os
import yaml
import wandb
import pickle
from datetime import datetime
import numpy as np
from collections import OrderedDict, defaultdict

def load_yaml_cfg(path):
    with open(path, "r") as f:
        return edict(yaml.safe_load(f))  # 바로 dot-access 가능
    

def expand_k(x, k: int):
    """
    Expand input by factor k.

    Case 1: list
        각 원소를 k번 반복해서 새 리스트 생성.
        예) [1,2,3], k=3 -> [1,1,1,2,2,2,3,3,3]

    Case 2: torch.Tensor
        맨 앞 차원을 batch라고 생각하고 B -> B*K로 확장.
        예) (B, C, H, W) -> (B*K, C, H, W)

    Args:
        x: list or torch.Tensor
        k: 반복 횟수

    Returns:
        list or torch.Tensor
    """
    if isinstance(x, list):
        return [item for item in x for _ in range(k)]

    elif isinstance(x, torch.Tensor):
        B = x.shape[0]
        # (B, ...) -> (B, K, ...)
        x_exp = x.unsqueeze(1).expand(B, k, *x.shape[1:])
        # -> (B*K, ...)
        x_out = x_exp.reshape(B * k, *x.shape[1:])
        return x_out

    else:
        raise TypeError(f"Unsupported type: {type(x)}")



class pObject(object):
    def __init__(self):
        pass


def load_config_file(name, add_timestamp=True):
    cfg = yaml.safe_load(open(name, 'r'))
    model_cfg_path = cfg['model_cfg']
    assert os.path.exists(model_cfg_path), f"Model config file {model_cfg_path} does not exist."
    cfg['model_cfg'] = yaml.safe_load(open(model_cfg_path, 'r'))
    
    if add_timestamp:
        folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg['output_dir'] = os.path.join(cfg['output_dir'], f'exp_{folder_name}')
        if not os.path.exists(cfg['output_dir']):
            os.makedirs(cfg['output_dir'], exist_ok=False)
    cfg['log_dir'] = os.path.join(cfg['output_dir'], 'logs')
    
    cfg_new = pObject()
    for attr in list(cfg.keys()):
        if attr == 'model_cfg':
            setattr(cfg_new, attr, pObject())
            for sub_attr in list(cfg[attr].keys()):
                setattr(getattr(cfg_new, attr), sub_attr, cfg[attr][sub_attr])
        else:
            setattr(cfg_new, attr, cfg[attr])
    return cfg_new


def to_vars(cfg):
    cfg_ = vars(cfg)
    cfg_new = dict()
    allowed_types = (int, float, str, bool)
    for key in cfg_.keys():
        if cfg_[key] is None:
            continue
        elif isinstance(cfg_[key], allowed_types):
            cfg_new[key] = cfg_[key]
        elif isinstance(cfg_[key], (list, tuple)):
            value_merged = ''
            for value in cfg_[key]:
                value_merged = value_merged.join(f', {str(value)}')
            cfg_new[key] = value_merged
        elif isinstance(cfg_[key], dict):
            for sub_key in cfg_[key].keys():
                cfg_new[f"{key}.{sub_key}"] = cfg_[key][sub_key]
        elif isinstance(cfg_[key], pObject):
            vars_cfg = vars(cfg_[key])
            for sub_key in vars_cfg.keys():
                cfg_new[f"{key}.{sub_key}"] = vars_cfg[sub_key]
        else:
            pass
            
    return cfg_new


def make_yaml_dumpable(D):
    if isinstance(D, np.ndarray):
        return D.tolist()
    for d in D:
        if isinstance(D[d], dict) or isinstance(D[d], OrderedDict) or isinstance(D[d], defaultdict):
            D[d] = dict(D[d])
            D[d] = make_yaml_dumpable(D[d])
            continue
        if isinstance(D[d], np.ndarray):
            D[d] = D[d].tolist()
            continue
        if np.issubdtype(type(D[d]), int):
            D[d] = int(D[d])
            continue
        if np.issubdtype(type(D[d]), float):
            D[d] = float(D[d])
            continue
        if np.issubdtype(type(D[d]), str):
            D[d] = str(D[d])
            continue
        if isinstance(D[d], list):
            for i in range(len(D[d])):
                D[d][i] = make_yaml_dumpable(D[d][i])
            continue
    return dict(D)


def set_seed(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def read_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def setup_wandb(cfg):
    wandb.init(
    project="Track6D",
    name=f"{os.path.basename(cfg.log_dir)}",
    save_code=False,
    config=vars(cfg),
)
    
    
class LRUMeshCache:
    def __init__(self, maxsize=16):
        self.maxsize = maxsize
        self.od = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key, loader, *args, **kwargs):
        """loader: (key, *args, **kwargs) -> value"""
        if key in self.od:
            self.od.move_to_end(key, last=True)
            self.hits += 1
            return self.od[key]
        val = loader(key, *args, **kwargs)  # 인자 확장
        self.od[key] = val
        self.od.move_to_end(key, last=True)
        if len(self.od) > self.maxsize:
            self.od.popitem(last=False)
            self.misses += 1
        return val

    def hit_rate(self):
        t = self.hits + self.misses
        return self.hits / t if t else 0.0
