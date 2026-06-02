import os

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, Callback, LearningRateMonitor

from utils.dataset_utils import PromptTrainDataset
from utils.loss_utils import edge_loss
from utils.schedulers import LinearWarmupCosineAnnealingLR
from utils.val_utils import compute_psnr_ssim
from net.model import PromptIR
from options import options as opt


# --------------------------- Losses ---------------------------

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps2))


def fft_l1_loss(pred, target):
    # rFFT halves the spectrum via conjugate symmetry; matching |F| is enough
    # to enforce per-frequency amplitude reconstruction. Stay in the input
    # dtype (bf16 under bf16-mixed precision) to keep activation memory low.
    pred_f = torch.fft.rfft2(pred, norm='ortho')
    targ_f = torch.fft.rfft2(target, norm='ortho')
    return F.l1_loss(torch.abs(pred_f), torch.abs(targ_f))


# --------------------------- EMA ---------------------------

class ModelEma:
    """EMA shadow weights pinned to CPU to keep GPU memory free.

    update() pays a small H2D->H copy per step but the params (~36M for
    PromptIR) are negligible compared to activations, so this is the right
    trade-off when GPU memory is tight.
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().cpu()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        msd = model.state_dict()
        for k, v in self.shadow.items():
            src = msd[k].detach().to(v.device, non_blocking=True)
            if v.is_floating_point():
                v.mul_(d).add_(src.to(v.dtype), alpha=1.0 - d)
            else:
                v.copy_(src)


class EMACallback(Callback):
    def __init__(self, decay=0.999):
        self.decay = decay
        self.ema = None
        self._pending_state = None

    def on_fit_start(self, trainer, pl_module):
        self.ema = ModelEma(pl_module.net, decay=self.decay)
        if self._pending_state is not None:
            for k, v in self._pending_state.items():
                if k in self.ema.shadow:
                    self.ema.shadow[k].copy_(v.to(self.ema.shadow[k].device))
            self._pending_state = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.ema is not None:
            self.ema.update(pl_module.net)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self.ema is not None:
            checkpoint['ema_state_dict'] = {k: v.detach().cpu()
                                             for k, v in self.ema.shadow.items()}

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if 'ema_state_dict' in checkpoint:
            self._pending_state = checkpoint['ema_state_dict']


# --------------------------- Lightning module ---------------------------

class PromptIRModelV2(pl.LightningModule):
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
        self.pixel_loss = CharbonnierLoss(eps=1e-3)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad, clean) = batch

        restored = self.net(degrad)

        l_pix = self.pixel_loss(restored, clean)
        l_edge = edge_loss(restored, clean)
        l_fft = fft_l1_loss(restored, clean)
        loss = (self.hparams.w_pixel * l_pix
                + self.hparams.w_edge * l_edge
                + self.hparams.w_fft * l_fft)

        bs = degrad.size(0)
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)
        self.log("charbonnier_loss", l_pix, on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)
        self.log("edge_loss", l_edge, on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)
        self.log("fft_loss", l_fft, on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)

        psnr, ssim, _ = compute_psnr_ssim(restored, clean)
        self.log("psnr", psnr, on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)
        self.log("ssim", ssim, on_step=True, on_epoch=True, sync_dist=True, batch_size=bs)

        if self.logger and isinstance(self.logger, WandbLogger) and batch_idx % 6000 == 0:
            grid = torchvision.utils.make_grid(
                torch.cat([degrad[:4], restored[:4], clean[:4]], dim=0),
                nrow=4, normalize=True, scale_each=True)
            self.logger.experiment.log({
                "Sample Input/Output/GT": wandb.Image(grid),
                "global_step": self.global_step,
                "epoch": self.current_epoch,
            })

        return loss

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                betas=(0.9, 0.999),
                                weight_decay=self.hparams.weight_decay)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs,
            eta_min=1e-6,
        )
        return [optimizer], [scheduler]


# --------------------------- Main ---------------------------

def main():
    print("Options")
    print(opt)

    if opt.wblogger is not None:
        run_name = (opt.wandb_name or "v2") + "_v2"
        os.makedirs("wandb_v2", exist_ok=True)
        logger = WandbLogger(project=opt.wblogger, name=run_name, save_dir="wandb_v2")
    else:
        logger = TensorBoardLogger(save_dir="logs/")

    trainset = PromptTrainDataset(opt)
    trainloader = DataLoader(
        trainset,
        batch_size=opt.batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=opt.num_workers,
        persistent_workers=opt.num_workers > 0,
    )

    warmup = min(15, max(1, opt.epochs // 10))
    model = PromptIRModelV2(
        lr=opt.lr,
        weight_decay=1e-4,
        warmup_epochs=warmup,
        max_epochs=opt.epochs,
        w_pixel=1.0,
        w_edge=0.05,
        w_fft=0.1,
    )

    if opt.init_from is not None:
        if opt.resume is not None:
            raise ValueError("--init-from and --resume are mutually exclusive: "
                             "--resume restores full training state, "
                             "--init-from only loads weights for fine-tuning.")
        init_path = os.path.expanduser(opt.init_from)
        if not os.path.isfile(init_path):
            raise FileNotFoundError(f"--init-from checkpoint not found: {init_path}")
        print(f"Initializing weights from {init_path} (no optimizer/epoch restore)")
        ckpt = torch.load(init_path, map_location='cpu')
        state = ckpt.get('state_dict', ckpt)
        net_state = {k[len('net.'):]: v for k, v in state.items() if k.startswith('net.')}
        if not net_state:
            net_state = state
        missing, unexpected = model.net.load_state_dict(net_state, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)} (showing up to 5) {missing[:5]}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)} (showing up to 5) {unexpected[:5]}")

    ckpt_best = ModelCheckpoint(
        monitor="psnr_epoch",
        mode="max",
        save_top_k=1,
        save_last=True,
        dirpath=opt.ckpt_dir,
        filename=opt.ckpt_name + "_v2",
        every_n_epochs=1,
        save_on_train_epoch_end=True,
    )
    ema_cb = EMACallback(decay=0.999)
    lr_mon = LearningRateMonitor(logging_interval='epoch')

    num_devices = len(opt.num_gpus) if isinstance(opt.num_gpus, list) else opt.num_gpus
    strategy = "ddp_find_unused_parameters_true" if num_devices > 1 else "auto"

    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=opt.num_gpus,
        strategy=strategy,
        precision="bf16-mixed",
        gradient_clip_val=0.5,
        logger=logger,
        callbacks=[ckpt_best, ema_cb, lr_mon],
        log_every_n_steps=50,
    )
    trainer.fit(model=model, train_dataloaders=trainloader, ckpt_path=opt.resume)

    if ckpt_best.best_model_score is not None:
        print(f"Best PSNR: {ckpt_best.best_model_score.item():.4f}")
    else:
        print("Best PSNR: N/A (no checkpoint saved — training stopped before first epoch end)")
    print(f"Best ckpt: {ckpt_best.best_model_path or 'N/A'}")
    print(f"Last ckpt: {ckpt_best.last_model_path or 'N/A'}")


if __name__ == '__main__':
    main()


# ---------------------------------------------------------------------------
# Using EMA weights at test time
# ---------------------------------------------------------------------------
# Lightning stores the live (non-EMA) model in 'state_dict'. EMA weights are
# saved under 'ema_state_dict'. To evaluate with EMA, load them manually:
#
#   import torch
#   from net.model import PromptIR
#   ckpt = torch.load("train_ckpt/best_rainsnow_edge_v2.ckpt", map_location="cpu")
#   net = PromptIR(decoder=True)
#   # 'ema_state_dict' keys come from net.state_dict() directly (no 'net.' prefix)
#   net.load_state_dict(ckpt['ema_state_dict'], strict=True)
#   net.eval().cuda()
#
# Then run inference (with self-ensemble TTA already in test.py).
