""" Train for generating LIIF, from image to implicit representation.

    Config:
        train_dataset:
          dataset: $spec; wrapper: $spec; batch_size:
        val_dataset:
          dataset: $spec; wrapper: $spec; batch_size:
        (data_norm):
            inp: {sub: []; div: []}
            gt: {sub: []; div: []}
        (eval_type):
        (eval_bsize):

        model: $spec
        optimizer: $spec
        epoch_max:
        (multi_step_lr):
            milestones: []; gamma: 0.5
        (resume): *.pth

        (epoch_val): ; (epoch_save):
"""

# CUDA_VISIBLE_DEVICES=3 python train.py --config configs/other/train-EvSRNet-timelens.yaml --name EvSRNet_bsergb
# CUDA_VISIBLE_DEVICES=1 python train.py --config configs/other/train-tmnet-super-timelens.yaml --name sup_TMNet_bsergb

import argparse
import os

import yaml
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from src.depth_extrapolation import datasets
from src.depth_extrapolation import models
from src.depth_extrapolation import utils
from src.depth_extrapolation.test_other import eval_psnr
import torch.nn.functional as F
import cv2
from src.e2vid.utils.loading_utils import load_model
from src.e2vid.image_reconstructor import ImageReconstructor

def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    # dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['dataset']).get_train_dataset()
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    dataset[0]

    log('{} dataset: size={}'.format(tag, len(dataset)))

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=(tag == 'train'), num_workers=8, pin_memory=True,
        persistent_workers=True)
    return loader

def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader

def prepare_training():
    if config.get('resume') is not None:
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        for _ in range(epoch_start - 1):
            lr_scheduler.step()
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler

def train(train_loader, model, optimizer, reconstructor, model_name):
    model.train()
    loss_fn = nn.L1Loss()
    loss_ft = nn.KLDivLoss()
    train_loss = utils.Averager()

    data_norm = config['data_norm']
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    # for batch in tqdm(train_loader, leave=False, desc='train'):
    with tqdm(train_loader, desc='train', leave=False) as pbar:
        for batchs in pbar:
            total_loss = 0
            for idx, batch in enumerate(batchs):

                if idx == 0:
                    reconstructor.last_states_for_each_channel = {'grayscale': None}

                for k, v in batch.items():
                    if 'path' in k:
                        continue
                    else:
                        batch[k] = v.cuda()

                inp_start = (batch['start'] - inp_sub) / inp_div
                gt = ((batch['gt'] - gt_sub) / gt_div).cuda()

                event = batch['start_event']
                recon_event = batch['recon_event']

                recon_left_img, states_real_L, latent_real_L = reconstructor.update_reconstruction(recon_event)
                # cv2.imwrite('test.png', 255.0*recon_left_img[0].cpu().numpy().transpose(1,2,0))

                if model_name in ['cbmnet_light_extrapolation', 'cbmnet_light_extrapolation_small', 'cbmnet_light_extrapolation_wo_flow']:
                    pred, g_I0_F_t_0  = model(inp_start.float(), event.float())
                    L1_lossFn = nn.L1Loss()
                    depth_mask = gt > -1
                    recnLoss =  L1_lossFn(pred[depth_mask], gt[depth_mask])
                    loss = recnLoss
                elif model_name in ['cbmnet_light_extrapolation_small_e2vid', 'cbmnet_light_extrapolation_small_e2vid_corr', 'cbmnet_light_extrapolation_small_tma', 'cbmnet_light_extrapolation_small_eemflow']:
                    pred, g_I0_F_t_0  = model(inp_start.float(), event.float(), latent_real_L)
                    L1_lossFn = nn.L1Loss()
                    depth_mask = gt > -1
                    recnLoss =  L1_lossFn(pred[depth_mask], gt[depth_mask])
                    loss = recnLoss

                else:
                    raise Exception("There is no mathced model!")

                # print(batch['gt'].max())
                valid_count = depth_mask.sum()
                if valid_count == 0:
                    continue
                total_loss += loss

            train_loss.add(total_loss.item())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            l = total_loss.item()
            pbar.set_postfix(loss=f"{l:.4f}")  # 진행바 우측에 표시

            pred = None; loss = None; lr_pred = None; total_loss = None

    return train_loss.item()

def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18

    timer = utils.Timer()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path_to_model = 'weights/e2vid.pth.tar'
    front_end_sensor_b, e2vid_decoder = load_model(path_to_model)
    front_end_sensor_b, e2vid_decoder = front_end_sensor_b.to(device), e2vid_decoder.to(device)
    for param in front_end_sensor_b.parameters():
        param.requires_grad = False
    front_end_sensor_b.eval()

    reconstructor = ImageReconstructor(front_end_sensor_b, 480, 640,
                                            5, device,
                                            args)

    for epoch in range(epoch_start, epoch_max + 1):
        model_spec = config['model']
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_loss = train(train_loader, model, optimizer, reconstructor, model_spec['name'])
        if lr_scheduler is not None:
            lr_scheduler.step()

        log_info.append('train: loss={:.4f}'.format(train_loss))
        writer.add_scalars('loss', {'train': train_loss}, epoch)

        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model

        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,
                os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            if n_gpus > 1 and (config.get('eval_bsize') is not None):
                model_ = model.module
            else:
                model_ = model
            val_res, ssim, lr_res, lr_ssim = eval_psnr(val_loader, model_,
                data_norm=config['data_norm'],
                eval_type=config.get('eval_type'),
                eval_bsize=config.get('eval_bsize'),
                model_name = model_spec['name'])

            log_info.append('val: psnr={:.4f}, ssim={:.4f}, lr_val: psnr={:.4f}, ssim={:.4f}'.format(val_res, ssim, lr_res, lr_ssim))
            writer.add_scalars('psnr', {'val': val_res}, epoch)
            if val_res > max_val_v:
                max_val_v = val_res
                torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='3')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path)
