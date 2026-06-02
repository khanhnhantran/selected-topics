"""Inference script for checkpoints produced by train_v2.py.

Differences vs test_v1.py:
- Mirrors the PromptIRModelV2 hparam signature so live-weight loading via
  Lightning's load_from_checkpoint accepts the v2 checkpoint hparams.
- Default checkpoint path points to the v2 artifact.
- --use-ema loads 'ema_state_dict' produced by train_v2's EMACallback.
  Falls back to live weights with a warning if EMA weights are absent.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import lightning.pytorch as pl

from net.model import PromptIR
from utils.dataset_utils import TestSpecificDataset


def parse_gpu_ids(value):
    ids = [int(v) for v in str(value).split(',') if v.strip() != '']
    return [i for i in ids if i >= 0]


def resolve_test_path(path):
    path = os.path.expanduser(path)
    if os.path.isdir(path):
        degraded = os.path.join(path, 'degraded')
        if os.path.isdir(degraded):
            path = degraded
        if not path.endswith('/'):
            path = path + '/'
    return path


def _augment(x, op):
    if op >= 4:
        x = torch.flip(x, dims=[2])
    k = op % 4
    if k:
        x = torch.rot90(x, k, dims=[2, 3])
    return x


def _deaugment(x, op):
    k = op % 4
    if k:
        x = torch.rot90(x, -k, dims=[2, 3])
    if op >= 4:
        x = torch.flip(x, dims=[2])
    return x


def pad_input(input_, img_multiple_of=8):
    height, width = input_.shape[2], input_.shape[3]
    H = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of
    W = ((width + img_multiple_of) // img_multiple_of) * img_multiple_of
    padh = H - height if height % img_multiple_of != 0 else 0
    padw = W - width if width % img_multiple_of != 0 else 0
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
    return input_, height, width


def tile_eval(model, input_, tile=128, tile_overlap=32):
    b, c, h, w = input_.shape
    tile = min(tile, h, w)
    assert tile % 8 == 0, "tile size should be multiple of 8"

    stride = tile - tile_overlap
    h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
    w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
    E = torch.zeros(b, c, h, w).type_as(input_)
    W = torch.zeros_like(E)

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = input_[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
            out_patch = model(in_patch)
            out_patch_mask = torch.ones_like(out_patch)

            E[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch)
            W[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch_mask)
    restored = E.div_(W)
    return torch.clamp(restored, 0, 1)


class PromptIRModelV2(pl.LightningModule):
    """Minimal mirror of train_v2.PromptIRModelV2 for checkpoint loading.

    Only the attributes referenced by state_dict keys ('net.*') and the
    __init__ kwargs saved via save_hyperparameters are needed here. Losses
    and optimizer config are omitted since this module is inference-only.
    """
    def __init__(self,
                 lr=2e-4,
                 weight_decay=1e-4,
                 warmup_epochs=15,
                 max_epochs=150,
                 w_pixel=1.0,
                 w_edge=0.05,
                 w_fft=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.net = PromptIR(decoder=True)

    def forward(self, x):
        return self.net(x)


def load_net(ckpt_path, use_ema, device):
    """Return a PromptIR model with weights loaded according to use_ema.

    - use_ema=False: standard load_from_checkpoint (live weights).
    - use_ema=True : load 'ema_state_dict' into a fresh PromptIR directly.
      If the checkpoint has no EMA weights, fall back to live weights with
      a warning.
    """
    ckpt_path = os.path.expanduser(ckpt_path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if not use_ema:
        print(f"Loading model from {ckpt_path} (live weights)")
        wrapper = PromptIRModelV2.load_from_checkpoint(ckpt_path).to(device)
        wrapper.eval()
        return wrapper

    print(f"Loading model from {ckpt_path} (EMA weights)")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'ema_state_dict' not in ckpt:
        print("  WARNING: 'ema_state_dict' not found in checkpoint, "
              "falling back to live weights.")
        wrapper = PromptIRModelV2.load_from_checkpoint(ckpt_path).to(device)
        wrapper.eval()
        return wrapper

    net = PromptIR(decoder=True)
    missing, unexpected = net.load_state_dict(ckpt['ema_state_dict'], strict=False)
    if missing:
        print(f"  Missing keys when loading EMA: {len(missing)} (showing up to 5) {missing[:5]}")
    if unexpected:
        print(f"  Unexpected keys when loading EMA: {len(unexpected)} (showing up to 5) {unexpected[:5]}")
    net.to(device).eval()
    return net


def predict_once(net, x, opt):
    if opt.tile is False:
        out = net(x)
    else:
        x_pad, h, w = pad_input(x)
        out = tile_eval(net, x_pad, tile=opt.tile_size, tile_overlap=opt.tile_overlap)
        out = out[:, :, :h, :w]
    return out


def predict(net, x, opt):
    if not opt.self_ensemble:
        return predict_once(net, x, opt)
    outs = []
    for op in range(8):
        ya = predict_once(net, _augment(x, op), opt)
        outs.append(_deaugment(ya, op))
    return torch.stack(outs, dim=0).mean(dim=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', type=str,
                        default="train_ckpt/best_rainsnow_edge_v2.ckpt",
                        help='path to the model checkpoint (.ckpt) to load for prediction')
    parser.add_argument('--test-path', type=str,
                        default="data/hw4_realse_dataset/test",
                        help='path to the test data (the test/ folder or test/degraded/ folder)')
    parser.add_argument('--gpu-ids', type=str, default="0",
                        help='GPU id(s) to use, e.g. "0" or "0,1". Use "-1" for CPU')
    parser.add_argument('--output-npz', type=str, default="pred_v2.npz",
                        help='output .npz file path')
    parser.add_argument('--self-ensemble', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='average over 8-way geometric TTA (use --no-self-ensemble to disable)')
    parser.add_argument('--use-ema', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='load EMA weights from checkpoint if available '
                             '(use --no-use-ema for the live weights)')
    parser.add_argument('--tile', type=bool, default=False, help='set to use tiling')
    parser.add_argument('--tile-size', type=int, default=128, help='tile size')
    parser.add_argument('--tile-overlap', type=int, default=32, help='tile overlap')
    opt = parser.parse_args()

    np.random.seed(0)
    torch.manual_seed(0)

    gpu_ids = parse_gpu_ids(opt.gpu_ids)
    use_cuda = torch.cuda.is_available() and len(gpu_ids) > 0
    if use_cuda:
        torch.cuda.set_device(gpu_ids[0])
        device = torch.device(f'cuda:{gpu_ids[0]}')
        if len(gpu_ids) > 1:
            print(f"Note: batch_size=1 inference runs on a single GPU; using cuda:{gpu_ids[0]}")
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    net = load_net(opt.checkpoint_path, opt.use_ema, device)

    opt.test_path = resolve_test_path(opt.test_path)
    print(f"Reading test images from {opt.test_path}")
    test_set = TestSpecificDataset(opt)
    testloader = DataLoader(test_set, batch_size=1, pin_memory=True,
                            shuffle=False, num_workers=0)

    images_dict = {}
    print(f"Start testing... (self_ensemble={opt.self_ensemble}, "
          f"tile={opt.tile}, use_ema={opt.use_ema})")
    with torch.no_grad():
        for ([clean_name], degrad_patch) in tqdm(testloader, colour='green'):
            degrad_patch = degrad_patch.to(device)

            restored = predict(net, degrad_patch, opt)

            restored = torch.clamp(restored, 0, 1)
            img = restored[0].cpu().numpy()
            img = np.round(img * 255.0).astype(np.uint8)

            filename = clean_name[0] + '.png'
            images_dict[filename] = img

    np.savez(opt.output_npz, **images_dict)
    print(f"Saved {len(images_dict)} restored images to {opt.output_npz}")

    sample_key = next(iter(images_dict))
    sample = images_dict[sample_key]
    print(f"Sample: key='{sample_key}', shape={sample.shape}, "
          f"dtype={sample.dtype}, range=[{sample.min()}, {sample.max()}]")
