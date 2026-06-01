import argparse
import os
import math
from functools import partial

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from . import datasets
from . import models
from . import utils
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []

        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)

            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred

def batched_predict_event_temp(model, inp, coord, cell, bsize, event, total_event, t):
    with torch.no_grad():
        model.gen_feat(inp, t, event, total_event)
        n = coord.shape[1]
        ql = 0
        preds = []

        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)

            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred

def batched_predict_event(model, inp, coord, cell, bsize, event, t):
    with torch.no_grad():
        model.gen_feat(inp, t, event)
        n = coord.shape[1]
        ql = 0
        preds = []

        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)

            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred

def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False, model_name=None, visualize=None, hr_size=None, sr=None):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    if eval_type is None:
        metric_fn = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        metric_fn = utils.calc_psnr
    # else:
    #     raise NotImplementedError

    val_res = utils.Averager()
    ssim_res = utils.Averager()

    lr_pred = None
    lr_val_res = utils.Averager()
    lr_ssim_res = utils.Averager()
    mse_loss = utils.Averager()
    mse_loss_fn = F.mse_loss
    def RMSELoss(yhat,y):
        return torch.sqrt(torch.mean((yhat-y)**2))

    rmse_loss_fn = RMSELoss

    pbar = tqdm(loader, leave=False, desc='val')
    idx = 0
    for batch in pbar:

        if idx<0:
            idx += 1
            continue
        else:
            for k, v in batch.items():
                # batch[k] = v.cuda()
                if not isinstance(v, list):
                    batch[k] = v.cuda()

            # start = torch.cuda.Event(enable_timing=True)
            # end = torch.cuda.Event(enable_timing=True)
            # start.record()

            inp = (batch['start'] - inp_sub) / inp_div
            if len(batch['start_event']) != 0:
                event = batch['start_event']

            gt = ((batch['gt'] - gt_sub) / gt_div).cuda()
            inp_mask = batch['start_mask']
            inp_rgb = batch['start_rgb']

            if eval_bsize is None:
                with torch.no_grad():
                    if model_name in ['EvSRNet', 'E2SRI', 'DSRNet', 'EGVSR']:
                        # pred = model(lr_inp.float(), batch['time'], event)
                        pred = model(lr_inp.float(), batch['time'], event.float(), lrs[:,1])
                    elif model_name in ['TMNet']:
                        pred = model(inp.float(), batch['time'])
                    elif model_name in ['zooming_slomo', 'RSTT']:
                        pred = model(inp)
                    elif model_name in ['super_slomo', 'qvi']:
                        # lr_gt =  ((batch['lr_gt'] - gt_sub) / gt_div).cuda()
                        pred = model(inp, batch['time'])
                    elif model_name in ['timereplayer', 'timelens']:
                        # lr_gt =  ((batch['lr_gt'] - gt_sub) / gt_div).cuda()
                        # pdb.set_trace()
                        pred = model(lr_inp.float(), event, batch['time'])
                    elif model_name in ['cbmnet', 'cbmnet_light']:
                        pred, _, _ = model(inp.float(), event)
                    elif model_name in ['cbmnet_light_extrapolation']:
                        pred, _ = model(inp.float(), event)
                    elif model_name in ['cbmnet_light_extrapolation_mask']:
                        pred, _ = model(torch.cat([inp.float(), inp_mask.float()], dim=1), event.float(), inp_rgb.float())
                    elif model_name in ['EMA']:
                        pred = model(inp.float(), event)
                    elif model_name in ['e2vid']:
                        # lr_gt =  ((batch['lr_gt'] - gt_sub) / gt_div).cuda()
                        pred = model(inp, event)
                    elif model_name in ['edvr', 'basicvsr', 'basicvsr_plusplus']:
                        inp_target = (batch['lr_gt'] - inp_sub) / inp_div
                        inp = torch.cat((inp_start.unsqueeze(1), inp_target.unsqueeze(1), inp_end.unsqueeze(1)), 1)
                        pred = model(inp)

            # if visualize:
            #     # w, h = hr_size

            if model_name in ['TMNet', 'zooming_slomo', 'RSTT']:
                res = metric_fn(pred[:, 1, :, :, :], batch['gt'][:, 1, :, :, :].to(dtype=torch.float))

            elif model_name in ['super_slomo', 'timereplayer', 'e2vid', 'timelens', 'qvi', 'cbmnet', 'EMA', 'cbmnet_light', 'cbmnet_light_extrapolation']:
                if sr is not None:
                    inp_pred = (pred - inp_sub) / inp_div
                    sr_inp = torch.cat((inp_start.unsqueeze(1), inp_pred.unsqueeze(1), inp_end.unsqueeze(1)), 1)
                    # sr_pred = sr(sr_inp.float())
                    sr_pred = sr(lr_inp.float(), batch['time'], event.float(), inp_pred)
                    sr_pred = sr_pred * gt_div + gt_sub
                    sr_pred.clamp_(0, 1)
                    res = metric_fn(sr_pred, batch['gt'][:, 1, :, :, :].float())
                else:
                    res = metric_fn(pred.float(), batch['gt'].float())
                    # res = metric_fn(m(pred).float(), batch['gt'][:, 1, :, :, :])
            elif model_name in ['cbmnet_light_extrapolation']:
                res = metric_fn(pred.float(), batch['gt'].float())
            elif model_name in ['cbmnet_light_extrapolation_mask']:
                res = metric_fn(pred[:,:1].float(), batch['gt'].float())

            elif model_name in ['edvr', 'basicvsr', 'basicvsr_plusplus', 'EvSRNet', 'E2SRI', 'EGVSR',
                    'DSRNet', 'DSRNet_dcn', 'EvSRNet_DSR_dcn_refine', 'EvSRNet_DSR_dcn_refine_atten_wo_off', 'EvSRNet_DSR_dcn_refine_atten_wo_img', 'EvSRNet_DSR_dcn_refine_edge', 'EvSRNet_DSR_dcn_refine_edge_event']:
                res = metric_fn(pred.float(), batch['gt'][:, 1, :, :, :].float())

            val_res.add(res[0].item(), inp.shape[0])
            ssim_res.add(res[1].item(), inp.shape[0])
            # if model_name in ['TMNet_event_liif_temp_kd']:

            if visualize:
                b,c,h,w = inp.size()

                # Ours
                os.makedirs(args.output, exist_ok=True)

                pred = pred[0, :, :, :] * gt_div + gt_sub
                gt = gt[0, :, :, :] * gt_div + gt_sub

                pred = pred.cpu()
                gt = gt.cpu()

                # sr_pred = sr_pred * gt_div + gt_sub
                # sr_pred.clamp_(0, 1)
                # img = sr_pred[0, :, :, :].cpu()
                name = str(idx) + ".png"
                gt_name = str(idx) + "_gt.png"
                import cv2

                transforms.ToPILImage()(pred).save(os.path.join(args.output, name))
                transforms.ToPILImage()(gt).save(os.path.join(args.output, gt_name))

            idx += 1

        # end.record()
        # torch.cuda.synchronize()
        # print(start.elapsed_time(end))

        if verbose:
            pbar.set_description('val {:.4f}, ssim {:.4f}'.format(val_res.item(), ssim_res.item()))

    print("MSE:", mse_loss.item())
    return val_res.item(), ssim_res.item(), lr_val_res.item(), lr_ssim_res.item()

def feature_visualization(features, idx, save_dir, n=64):
    """
    features:       Features to be visualized
    n:              Maximum number of feature maps to plot
    """

    plt.figure(tight_layout=True)
    blocks = torch.chunk(features, features.shape[1], dim=0)  # block by channel dimension
    n = min(n, len(blocks))

    feature = transforms.ToPILImage()(blocks[46].squeeze())
    plt.imshow(feature)  # cmap='gray'
    plt.axis('off')
    plt.jet()

    # feature = transforms.ToPILImage()(blocks[14].squeeze())
    # plt.imshow(feature)  # cmap='gray'
    # plt.axis('off')

    f = f"layer_{idx}__features.png"
    # print(f'Saving {save_dir / f}...')
    plt.savefig(os.path.join(save_dir, f), dpi=300)
    plt.close()

def flow2rgb(flowmap):
    assert(isinstance(flowmap, torch.Tensor))
    global args

    _, H, W = flowmap.shape
    rgb = torch.ones((3,H,W))
    normalized_flow_map = flowmap[:3] / (flowmap[:3].max())
    rgb[0] += normalized_flow_map[0]
    rgb[1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
    rgb[2] += normalized_flow_map[1]

    # return rgb

    return rgb.clamp(0,1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--gpu', default='2')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--output', default='gopro_tmnet')
    parser.add_argument('--sr', default=None)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset']).get_train_dataset()
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    dataset[0]
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        num_workers=8, pin_memory=True)

    model_spec = torch.load(args.model)['model']

    model = models.make(model_spec, load_sd=True).cuda()
    modelname = model_spec['name']
    sr_model = None
    if args.sr is not None:
        sr_model_spec = torch.load(args.sr)['model']
        sr_model = models.make(sr_model_spec, load_sd=True).cuda()

    res, ssim, _, _ = eval_psnr(loader, model,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        verbose=True,
        model_name=modelname,
        visualize=args.visualize,
        hr_size=config.get('hr_size'),
        sr = sr_model)
    print('result: {:.4f}, ssim: {:.4f}'.format(res, ssim))
