#!/usr/bin/env python3
"""
Sigma-Logit-UNet++ edge-detection – training script  (v5.6, 2025-08-12)
──────────────────────────────────────────────────────────────────────
Changes vs v5.5:
• Fix black uncertainty visuals by adding adaptive normalization:
  --unc_vis_mode {percentile|minmax|abs} (default: percentile)
  --unc_vis_p (default: 99), --unc_vis_max (default: 1.0 for abs mode)
• Logs std histograms to TensorBoard for quick sanity checks.

v5 lineage summary:
• Distribution knob for Mymodel (--distribution {beta,gs,residual}).
• Top-K-by-monitor checkpointing, EMA, RRC, augment/mix, AP/AUC/mIoU + Precision/Recall/Acc/F1,
  cosine/plateau schedulers, variance-corrected sigmoid (opt), grad clip, YAML config,
  repro.sh, optional Test evaluation. Also logs Prob|Unc panels.
"""

# ───── standard libs ───── #
import argparse, os, sys, time, random, datetime, math, heapq, shlex
from pathlib import Path
from contextlib import nullcontext

# ───── third-party libs ── #
import numpy as np
from PIL import Image, ImageOps
from skimage.morphology import skeletonize

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision.transforms import functional as TF
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode as IM

# Optional YAML (fallback to simple writer if PyYAML not installed)
try:
    import yaml
except Exception:
    yaml = None

# ───── local model import ─ #
from model.sigma_logit_unetpp import Mymodel  # adjust path if necessary


# ═════════ utilities ══════ #

class Logger:
    def __init__(self, fp):
        self.console, self.log = sys.stdout, open(fp, "a", buffering=1)
    def write(self, m): self.console.write(m); self.log.write(m)
    def flush(self):    self.console.flush();  self.log.flush()

class AvgMeter:
    def __init__(self): self.reset()
    def reset(self): self.val=self.avg=self.sum=self.n=0
    def update(self,v,k=1): self.val=v; self.sum+=v*k; self.n+=k; self.avg=self.sum/self.n

def set_seed(seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def safe_yaml_dump(data: dict, fp: Path):
    """Write YAML; if PyYAML unavailable, write a simple YAML-like text."""
    try:
        if yaml is not None:
            with open(fp, "w") as f:
                yaml.safe_dump(data, f, sort_keys=False)
        else:
            with open(fp, "w") as f:
                for k,v in data.items(): f.write(f"{k}: {v}\n")
    except Exception as e:
        print(f"Warning: failed to write YAML config ({e})")

def write_repro_sh(args, run_dir: Path, script_path: str):
    def add_arg(parts, name, value):
        if isinstance(value, bool):
            if value: parts.append(f"--{name}")
        elif isinstance(value, (list, tuple)):
            vals = " ".join(shlex.quote(str(x)) for x in value)
            parts.append(f"--{name} {vals}")
        elif value is not None:
            parts.append(f"--{name} {shlex.quote(str(value))}")
    cmd = [shlex.quote(sys.executable), shlex.quote(script_path)]
    # Required
    add_arg(cmd, "train_image_path", args.train_image_path)
    add_arg(cmd, "train_gt_path",    args.train_gt_path)
    add_arg(cmd, "val_image_path",   args.val_image_path)
    add_arg(cmd, "val_gt_path",      args.val_gt_path)
    # Optional test
    add_arg(cmd, "test_image_path",  args.test_image_path)
    add_arg(cmd, "test_gt_path",     args.test_gt_path)
    # Model / train knobs
    add_arg(cmd, "pretrained_path",  args.pretrained_path)
    if args.skeletonize: cmd.append("--skeletonize")
    add_arg(cmd, "distribution",     args.distribution)
    add_arg(cmd, "amp",              args.amp)
    add_arg(cmd, "batch_size",       args.batch_size)
    add_arg(cmd, "epochs",           args.epochs)
    add_arg(cmd, "lr",               args.lr)
    add_arg(cmd, "weight_decay",     args.weight_decay)
    add_arg(cmd, "itersize",         args.itersize)
    add_arg(cmd, "std_weight",       args.std_weight)
    add_arg(cmd, "std_weight_final", args.std_weight_final)
    add_arg(cmd, "print_freq",       args.print_freq)
    add_arg(cmd, "gpu",              args.gpu)
    # Scheduler
    add_arg(cmd, "scheduler",        args.scheduler)
    add_arg(cmd, "lr_factor",        args.lr_factor)
    add_arg(cmd, "lr_patience",      args.lr_patience)
    add_arg(cmd, "min_lr",           args.min_lr)
    # Aug / mix
    if args.augment: cmd.append("--augment")
    add_arg(cmd, "mix_strategy", args.mix_strategy)
    add_arg(cmd, "mix_alpha",    args.mix_alpha)
    # RRC
    if args.rrc: cmd.append("--rrc")
    add_arg(cmd, "rrc_size",   args.rrc_size)
    add_arg(cmd, "rrc_scale",  args.rrc_scale)
    add_arg(cmd, "rrc_ratio",  args.rrc_ratio)
    # Stabilizers
    if args.ema: cmd.append("--ema")
    add_arg(cmd, "ema_decay", args.ema_decay)
    add_arg(cmd, "grad_clip", args.grad_clip)
    if args.focal: cmd.append("--focal")
    add_arg(cmd, "focal_alpha", args.focal_alpha)
    add_arg(cmd, "focal_gamma", args.focal_gamma)
    if args.use_variance_correction: cmd.append("--use_variance_correction")
    # Monitoring
    add_arg(cmd, "monitor", args.monitor)
    # TopK
    add_arg(cmd, "keep_topk", args.keep_topk)
    # Uncertainty vis
    add_arg(cmd, "unc_vis_mode", args.unc_vis_mode)
    add_arg(cmd, "unc_vis_p",    args.unc_vis_p)
    add_arg(cmd, "unc_vis_max",  args.unc_vis_max)

    sh_path = run_dir / "repro.sh"
    with open(sh_path, "w") as f:
        f.write("#!/usr/bin/env bash\nset -e\n\n")
        f.write(" ".join(cmd) + "\n")
    os.chmod(sh_path, 0o755)


# ═════════ dataset ════════ #

class FolderEdgeDataset(Dataset):
    """Dataset with optional paired augmentations (same transform for img & gt)."""
    def __init__(self, img_dir, gt_dir, skeleton=False, augment=False,
                 rrc=False, rrc_size=0, rrc_scale=(0.6,1.0), rrc_ratio=(0.75, 1.3333)):
        self.imgs = sorted(Path(img_dir).glob("*.*"))
        self.gts  = sorted(Path(gt_dir).glob("*.*"))
        assert len(self.imgs)==len(self.gts), "Image/GT count mismatch"
        self.skel = skeleton
        self.augment = augment
        self.cj = T.ColorJitter(0.2,0.2,0.2,0.1)

        self.use_rrc   = bool(rrc)
        self.rrc_size  = int(rrc_size)  # 0 => keep original size after crop
        self.rrc_scale = rrc_scale
        self.rrc_ratio = rrc_ratio

    def __len__(self): return len(self.imgs)

    def _paired_rrc(self, img, gt):
        i, j, h, w = T.RandomResizedCrop.get_params(img, scale=self.rrc_scale, ratio=self.rrc_ratio)
        if self.rrc_size and self.rrc_size > 0:
            out_h = out_w = self.rrc_size
        else:
            out_w, out_h = img.size
        img = TF.resized_crop(img, i, j, h, w, size=(out_h, out_w), interpolation=IM.BILINEAR)
        gt  = TF.resized_crop(gt,  i, j, h, w, size=(out_h, out_w), interpolation=IM.NEAREST)
        return img, gt

    def _paired_geometric(self, img, gt):
        if random.random() < 0.5:
            img = ImageOps.mirror(img); gt  = ImageOps.mirror(gt)
        if random.random() < 0.5:
            img = ImageOps.flip(img);   gt  = ImageOps.flip(gt)
        if random.random() < 0.5:
            k = random.choice([1,2,3])  # 90/180/270
            img = img.rotate(90*k, expand=False)
            gt  = gt.rotate(90*k,  expand=False)
        return img, gt

    def __getitem__(self, i):
        img = Image.open(self.imgs[i]).convert("RGB")
        gt  = Image.open(self.gts[i]).convert("L")

        if self.augment:
            if self.use_rrc:
                img, gt = self._paired_rrc(img, gt)
            img, gt = self._paired_geometric(img, gt)
            img = self.cj(img)

        img = TF.to_tensor(img)
        gt  = TF.to_tensor(gt)

        if self.skel:
            gt_np = skeletonize(gt.squeeze().numpy() > 0.5)
            gt    = torch.from_numpy(gt_np).float().unsqueeze(0)

        # Per-image GT stats for std_loss target
        gt_mean = gt.mean()
        gt_std  = gt.std()
        return img, gt, gt_mean, gt_std


# ═════════ mixup & cutmix helpers ══════ #

def mixup(img, gt, alpha):
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(img.size(0), device=img.device)
    img2, gt2 = img[perm], gt[perm]
    img = lam * img + (1 - lam) * img2
    gt  = lam * gt  + (1 - lam) * gt2
    return img, gt

def rand_bbox(W, H, lam):
    cut_ratio = math.sqrt(1 - lam)
    cut_w, cut_h = int(W * cut_ratio), int(H * cut_ratio)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = np.clip(cx - cut_w//2, 0, W); y1 = np.clip(cy - cut_h//2, 0, H)
    x2 = np.clip(cx + cut_w//2, 0, W); y2 = np.clip(cy + cut_h//2, 0, H)
    return x1, y1, x2, y2

def cutmix(img, gt, alpha):
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(img.size(0), device=img.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(3), img.size(2), lam)
    img[:,:, bby1:bby2, bbx1:bbx2] = img[perm,:, bby1:bby2, bbx1:bbx2]
    gt [:,:, bby1:bby2, bbx1:bbx2] = gt [perm,:, bby1:bby2, bbx1:bbx2]
    return img, gt


# ═════════ weighting & losses ══════ #

@torch.no_grad()
def compute_weights(tgt, std_map, ada, pos_scale=1.1, cap_exp=2.0, eps=1e-6):
    """Balanced class weights with capped uncertainty, normalized to mean≈1."""
    mask = (tgt > 0.5)
    pos = mask.sum()
    neg = (~mask).sum()

    pos_frac = (pos.float() / (pos + neg + eps)).clamp(1e-4, 0.5)
    pos_w = (1.0 - pos_frac)
    neg_w = pos_frac * pos_scale

    w = torch.where(mask, pos_w, neg_w).float()

    u = torch.exp(torch.clamp(std_map * ada, min=-cap_exp, max=cap_exp)).detach()
    w = w * u

    mean_w = w.mean().clamp(min=eps)
    w = w / mean_w
    return w

def bce_with_logits_weighted(logits, tgt, weight):
    loss_map = F.binary_cross_entropy_with_logits(logits, tgt, reduction='none')
    return (loss_map * weight).sum()

def focal_with_logits_weighted(logits, tgt, weight, alpha=0.75, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, tgt, reduction='none')
    p   = torch.sigmoid(logits).detach()
    mod = (1 - torch.where(tgt>0.5, p, 1-p)).pow(gamma)
    loss_map = alpha * mod * bce
    return (loss_map * weight).sum()

def variance_corrected_sigmoid(logits, std):
    return torch.sigmoid(logits / torch.sqrt(1 + (math.pi*(std**2))/8))


# ═════════ EMA ════════════ #

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad: self.shadow[n] = p.data.clone()

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            self.shadow[n].mul_(d).add_(p.data, alpha=1.0 - d)

    @torch.no_grad()
    def apply_to(self, model):
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            p.data.copy_(self.shadow[n])


# ═════════ AMP compatibility helpers ══════ #

def amp_autocast_ctx(args):
    """Return an autocast context that works across torch versions."""
    enabled = args.amp and getattr(args, "device", torch.device("cpu")).type == "cuda"
    if not enabled:
        return nullcontext()
    # Try the new API first: torch.amp.autocast('cuda', dtype=...)
    try:
        import torch.amp as amp
        return amp.autocast('cuda', dtype=torch.float16)
    except Exception:
        # Fallback to legacy torch.cuda.amp.autocast() (no kwargs)
        return torch.cuda.amp.autocast()

def make_grad_scaler(args):
    """Create a GradScaler that is compatible across torch versions."""
    enabled = args.amp and getattr(args, "device", torch.device("cpu")).type == "cuda"
    try:
        from torch.amp import GradScaler as NewGradScaler
        return NewGradScaler('cuda', enabled=enabled)
    except Exception:
        from torch.cuda.amp import GradScaler as OldGradScaler
        return OldGradScaler(enabled=enabled)


# ═════════ training & eval helpers ══════ #

def forward_model(model, img):
    """Support old/new return styles: (mean,std) or (logits,std_raw)."""
    out = model(img)
    if isinstance(out, (list, tuple)) and len(out) >= 2:
        logits, std_raw = out[0], out[1]
    else:
        logits = out["mean"] if "mean" in out else out["logits"]
        std_raw = out["std"] if "std" in out else out["std_raw"]
    return logits, std_raw

def train_epoch(ld, model, optim, ep, args, writer, ema_obj=None, scaler=None):
    model.train(); t_m,l_m=AvgMeter(),AvgMeter(); tic=time.time()

    # cosine decay of std_weight from initial to final across epochs
    cos_t = ep / (args.epochs - 1) if args.epochs > 1 else 1.0
    std_w = (args.std_weight_final + (args.std_weight - args.std_weight_final) *
             (0.5 * (1 + math.cos(math.pi * cos_t))))

    for i,(img,gt,gt_mean,gt_std) in enumerate(ld):
        img,gt,gt_std = img.to(args.device), gt.to(args.device), gt_std.to(args.device)

        if args.mix_strategy == 'mixup':
            img, gt = mixup(img, gt, args.mix_alpha)
        elif args.mix_strategy == 'cutmix':
            img, gt = cutmix(img, gt, args.mix_alpha)

        with amp_autocast_ctx(args):
            logits, std_raw = forward_model(model, img)

            # Bounded std head
            std = (torch.nn.functional.softplus(std_raw) + 1e-3).clamp(max=7.4)

            # Training uses logits directly (stable)
            ada = (ep + 1) / max(args.epochs, 1)
            w = compute_weights(gt, std, ada, pos_scale=1.1, cap_exp=2.0)

            if args.focal:
                cls_loss = focal_with_logits_weighted(logits, gt, w, alpha=args.focal_alpha, gamma=args.focal_gamma)
            else:
                cls_loss = bce_with_logits_weighted(logits, gt, w)

            # Std loss – per-image scalar alignment, modest weight with cosine decay
            std_img = std.mean(dim=[1,2,3])
            std_loss = ((std_img - gt_std.flatten())**2).mean()

            loss=(cls_loss + std_w * std_loss) / args.itersize

        if args.amp and args.device.type=='cuda':
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i+1)%args.itersize==0:
            if args.grad_clip and args.grad_clip > 0:
                if args.amp and args.device.type=='cuda' and hasattr(scaler, "unscale_"):
                    scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            if args.amp and args.device.type=='cuda':
                scaler.step(optim); scaler.update()
            else:
                optim.step()
            optim.zero_grad()
            if ema_obj is not None:
                ema_obj.update(model)

        l_m.update(loss.item(),img.size(0)); t_m.update(time.time()-tic); tic=time.time()
        if i%args.print_freq==0:
            print(f"Ep[{ep}/{args.epochs}] It[{i}/{len(ld)}] "
                  f"LR {optim.param_groups[0]['lr']:.2e} "
                  f"Loss {l_m.val:.3f} (avg {l_m.avg:.3f}) "
                  f"std_w {std_w:.4f} BT {t_m.val:.2f}s")
    writer.add_scalar("Train/Loss", l_m.avg, ep)
    writer.add_scalar("Train/std_weight", std_w, ep)

def _compute_binary_metrics(gts_bin: np.ndarray, preds_prob: np.ndarray, thr: float = 0.5):
    """Return dict with precision, recall, accuracy, f1, miou given flat arrays."""
    eps = 1e-12
    pred_bin = (preds_prob > thr)
    tp = np.logical_and(pred_bin, gts_bin == 1).sum()
    fp = np.logical_and(pred_bin, gts_bin == 0).sum()
    fn = np.logical_and(~pred_bin, gts_bin == 1).sum()
    tn = np.logical_and(~pred_bin, gts_bin == 0).sum()
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    accuracy  = (tp + tn) / (tp + tn + fp + fn + eps)
    f1        = 2.0 * precision * recall / (precision + recall + eps)

    inter = tp
    union = (tp + fp + fn)
    miou = inter / (union + eps)
    return dict(precision=precision, recall=recall, accuracy=accuracy, f1=f1, miou=miou)

def _make_unc_vis(single_std_map: torch.Tensor, args) -> torch.Tensor:
    """
    single_std_map: (H, W) tensor, nonnegative (already softplus+clamp in caller).
    Returns (H, W) tensor in [0,1] for visualization.
    """
    x = single_std_map.float()
    mode = args.unc_vis_mode
    if mode == "abs":
        scale = max(args.unc_vis_max, 1e-6)
        return torch.clamp(x / scale, 0.0, 1.0)
    elif mode == "minmax":
        mn = float(x.min())
        mx = float(x.max())
        return torch.clamp((x - mn) / (mx - mn + 1e-6), 0.0, 1.0)
    else:  # percentile (default)
        q = min(max(args.unc_vis_p, 1.0), 100.0) / 100.0
        pval = float(torch.quantile(x.flatten(), q))
        scale = max(pval, 1e-6)
        return torch.clamp(x / scale, 0.0, 1.0)

@torch.no_grad()
def _evaluate_split(model, loader, epoch, writer, split_name, args, overlay_n=0, use_ema=False, ema_obj=None):
    """Run eval on a loader; returns dict with loss, ap, auc, miou, precision, recall, accuracy, f1."""
    model.eval()

    # Optionally swap weights to EMA for eval
    backup = {}
    if use_ema and ema_obj is not None:
        for n,p in model.named_parameters():
            if p.requires_grad: backup[n] = p.data.clone()
        ema_obj.apply_to(model)

    gts, preds, loss_m = [], [], AvgMeter()
    for idx, (img, gt, _, gt_std) in enumerate(loader):
        img = img.to(args.device); gt = gt.to(args.device); gt_std = gt_std.to(args.device)

        with amp_autocast_ctx(args):
            logits, std_raw = forward_model(model, img)
            std = (torch.nn.functional.softplus(std_raw) + 1e-3).clamp(max=7.4)

            # In eval, optionally use variance-corrected expectation
            if args.use_variance_correction:
                pred_prob = variance_corrected_sigmoid(logits, std)
            else:
                pred_prob = torch.sigmoid(logits)

            # Comparable loss for logging
            w = compute_weights(gt, std, ada=(epoch+1)/max(args.epochs,1))
            if args.focal:
                cls_loss = focal_with_logits_weighted(logits, gt, w, alpha=args.focal_alpha, gamma=args.focal_gamma)
            else:
                cls_loss = bce_with_logits_weighted(logits, gt, w)
            std_img = std.mean(dim=[1,2,3])
            std_loss = ((std_img - gt_std.flatten())**2).mean()
            loss_val = (cls_loss + args.std_weight * std_loss)

        loss_m.update(loss_val.item(), img.size(0))

        # Flatten for metrics
        p = pred_prob[:, 0].detach().cpu()          # (B,H,W)
        g = (gt[:, 0] > 0.5).float().cpu()          # (B,H,W)
        s = std[:, 0].detach().cpu()                # (B,H,W)

        gts.append(g.reshape(-1)); preds.append(p.reshape(-1))

        # ── TensorBoard visuals: (1) GT|Pred overlay; (2) Prob|Unc panel ── #
        if idx < overlay_n:
            # (1) GT (red) vs Pred (green)
            overlay = torch.stack([g[0], p[0], torch.zeros_like(g[0])], 0)  # CHW
            writer.add_image(f"{split_name}/Overlay_{idx}", overlay, epoch, dataformats="CHW")

            # (2) Probability | Uncertainty side-by-side (grayscale, 3-ch)
            prob_vis = torch.clamp(p[0], 0.0, 1.0)               # (H,W)
            unc_vis  = _make_unc_vis(s[0], args)                 # (H,W)

            # Helpful histogram to spot collapse/saturation
            writer.add_histogram(f"{split_name}/std_hist_{idx}", s[0], epoch)

            panel = torch.cat([prob_vis, unc_vis], dim=1)  # (H, 2W)
            panel = panel.unsqueeze(0).repeat(3, 1, 1)     # (3, H, 2W)
            writer.add_image(f"{split_name}/Prob_Unc_{idx}", panel, epoch, dataformats="CHW")

    # restore original weights if EMA was applied
    if use_ema and ema_obj is not None and backup:
        with torch.no_grad():
            for n,p in model.named_parameters():
                if p.requires_grad and n in backup: p.data.copy_(backup[n])

    gts  = torch.cat(gts).numpy(); preds = torch.cat(preds).numpy()

    # Ranking metrics (threshold-free)
    if gts.max() == gts.min():
        ap = float('nan'); auc = float('nan')
    else:
        ap  = average_precision_score(gts, preds)
        auc = roc_auc_score(gts, preds)

    # Thresholded metrics (thr=0.5)
    m = _compute_binary_metrics(gts.astype(np.uint8), preds.astype(np.float64), thr=0.5)
    miou = m["miou"]

    # Log to TB
    writer.add_scalar(f"{split_name}/Loss", loss_m.avg, epoch)
    writer.add_scalar(f"{split_name}/mIoU", miou, epoch)
    writer.add_scalar(f"{split_name}/AP",   ap,   epoch)
    writer.add_scalar(f"{split_name}/ROC_AUC", auc, epoch)
    writer.add_scalar(f"{split_name}/Precision", m["precision"], epoch)
    writer.add_scalar(f"{split_name}/Recall",    m["recall"],    epoch)
    writer.add_scalar(f"{split_name}/Accuracy",  m["accuracy"],  epoch)
    writer.add_scalar(f"{split_name}/F1",        m["f1"],        epoch)

    print(f"[{split_name}] Epoch {epoch}: "
          f"Loss={loss_m.avg:.4f}  AP={ap:.4f}  AUC={auc:.4f}  "
          f"mIoU={miou:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  "
          f"Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}")

    return {
        "loss": loss_m.avg,
        "ap": ap,
        "auc": auc,
        "miou": miou,
        "precision": m["precision"],
        "recall": m["recall"],
        "accuracy": m["accuracy"],
        "f1": m["f1"],
    }

def validate(model, loader, epoch, writer, args, ema_obj=None):
    return _evaluate_split(model, loader, epoch, writer, "Val", args, overlay_n=5, use_ema=args.ema, ema_obj=ema_obj)

def test_eval(model, loader, epoch, writer, args, ema_obj=None):
    return _evaluate_split(model, loader, epoch, writer, "Test", args, overlay_n=0, use_ema=args.ema, ema_obj=ema_obj)


# ═════════ top-K checkpoint saver (by monitored metric) ════ #

class TopKSaver:
    """Keeps only top-K checkpoints by a chosen metric (e.g., AP or Recall); lower ones are deleted."""
    def __init__(self, k: int, run_dir: Path, metric_name: str):
        self.k = int(k)
        self.run_dir = Path(run_dir)
        self.heap = []  # min-heap of (metric_value, epoch, path)
        self.best = -1.0
        self.metric_name = metric_name  # e.g., "AP" or "Recall"

    def maybe_save(self, metric_value: float, epoch: int, state_dict: dict) -> tuple[bool, bool, Path|None]:
        if metric_value != metric_value:  # NaN
            return False, False, None

        ckpt_path = self.run_dir / f"epoch_{epoch:03d}_{self.metric_name}{metric_value:.4f}.pth"
        torch.save({"epoch": epoch, "state_dict": state_dict, f"val_{self.metric_name.lower()}": metric_value}, ckpt_path)

        heapq.heappush(self.heap, (metric_value, epoch, str(ckpt_path)))
        if len(self.heap) > self.k:
            evicted = heapq.heappop(self.heap)
            try:
                os.remove(evicted[2])
                print(f"  ↳ Removed {Path(evicted[2]).name} (kept top-{self.k})")
            except FileNotFoundError:
                pass

        best_changed = False
        if metric_value > self.best:
            self.best = metric_value
            best_changed = True
            torch.save({"epoch": epoch, "state_dict": state_dict, f"val_{self.metric_name.lower()}": metric_value},
                       self.run_dir / f"best_{self.metric_name.lower()}.pth")
        return True, best_changed, ckpt_path

    def snapshot(self):
        ordered = sorted(self.heap, key=lambda x: -x[0])
        return [(a, e, Path(p).name) for a, e, p in ordered]


# ═════════ CLI ═══════════ #

def get_args():
    p=argparse.ArgumentParser("Sigma-Logit-UNet++ trainer (v5.6)")
    p.add_argument('--train_image_path',required=True)
    p.add_argument('--train_gt_path',required=True)
    p.add_argument('--val_image_path',required=True)
    p.add_argument('--val_gt_path',required=True)
    p.add_argument('--test_image_path', default=None)
    p.add_argument('--test_gt_path',    default=None)

    p.add_argument('--pretrained_path')
    p.add_argument('--skeletonize',action='store_true')

    # Distribution for Mymodel
    p.add_argument('--distribution', default='gs', choices=['beta','gs','residual'],
                   help='kept for compatibility with Mymodel; default "gs"')

    # Half-precision (AMP)
    p.add_argument('--amp', action='store_true', help='Enable automatic mixed precision on CUDA')

    p.add_argument('--batch_size',type=int,default=4)
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--weight_decay',type=float,default=5e-4)
    p.add_argument('--itersize',type=int,default=1)
    p.add_argument('--std_weight',type=float,default=0.1, help="initial std loss weight")
    p.add_argument('--std_weight_final',type=float,default=0.0, help="final std loss weight (cosine decay)")
    p.add_argument('--print_freq',type=int,default=1000)
    p.add_argument('--gpu',default='0')

    # Scheduler
    p.add_argument('--scheduler',choices=['cosine','plateau'],default='cosine')
    p.add_argument('--lr_factor',type=float,default=0.5)
    p.add_argument('--lr_patience',type=int,default=3)
    p.add_argument('--min_lr',type=float,default=1e-8)

    # Augment / mix
    p.add_argument('--augment',action='store_true', help='Enable color/flip/rotation augmentations')
    p.add_argument('--mix_strategy',default='none',choices=['none','mixup','cutmix'])
    p.add_argument('--mix_alpha',type=float,default=0.2)

    # RandomResizedCrop
    p.add_argument('--rrc', action='store_true')
    p.add_argument('--rrc_size', type=int, default=0)
    p.add_argument('--rrc_scale', type=float, nargs=2, default=(0.8, 1.0), help='tighter default, preserves thin edges')
    p.add_argument('--rrc_ratio', type=float, nargs=2, default=(0.9, 1.1111))

    # Stabilizers
    p.add_argument('--ema', action='store_true')
    p.add_argument('--ema_decay', type=float, default=0.999)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--focal', action='store_true')
    p.add_argument('--focal_alpha', type=float, default=0.75)
    p.add_argument('--focal_gamma', type=float, default=2.0)
    p.add_argument('--use_variance_correction', action='store_true')

    # Monitoring / checkpoint retention
    p.add_argument('--monitor', choices=['ap','recall'], default='ap',
                   help='Metric to monitor for scheduler=plateau and Top-K saving.')
    p.add_argument('--keep_topk', type=int, default=5)

    # Uncertainty visualization
    p.add_argument('--unc_vis_mode', choices=['percentile','minmax','abs'], default='percentile',
                   help='How to normalize std for visualization.')
    p.add_argument('--unc_vis_p', type=float, default=99.0,
                   help='Percentile used when unc_vis_mode=percentile (1..100).')
    p.add_argument('--unc_vis_max', type=float, default=1.0,
                   help='Max std mapped to white when unc_vis_mode=abs.')
    return p.parse_args()


# ═════════ main ══════════ #

def main():
    args=get_args(); os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    args.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    run_stamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir=Path("logs")/run_stamp; run_dir.mkdir(parents=True,exist_ok=True)

    # persist config + repro script (serialize device to str)
    cfg = vars(args).copy()
    cfg["script"] = Path(sys.argv[0]).name
    cfg["run_dir"] = str(run_dir)
    if "device" in cfg:
        cfg["device"] = str(cfg["device"])
    safe_yaml_dump(cfg, run_dir/"config.yaml")
    write_repro_sh(args, run_dir, script_path=sys.argv[0])

    # data
    train_ds=FolderEdgeDataset(args.train_image_path,args.train_gt_path,
                               skeleton=args.skeletonize, augment=args.augment,
                               rrc=args.rrc, rrc_size=args.rrc_size,
                               rrc_scale=tuple(args.rrc_scale),
                               rrc_ratio=tuple(args.rrc_ratio))
    val_ds  =FolderEdgeDataset(args.val_image_path,args.val_gt_path,
                               skeleton=args.skeletonize, augment=False)
    train_ld=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,
                        drop_last=True,num_workers=8,pin_memory=True)
    val_ld  =DataLoader(val_ds,batch_size=1,shuffle=False,num_workers=8,pin_memory=True)

    # optional test loader
    test_ld = None
    if args.test_image_path and args.test_gt_path:
        test_ds = FolderEdgeDataset(args.test_image_path, args.test_gt_path,
                                    skeleton=args.skeletonize, augment=False)
        test_ld = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    # MixUp/CutMix guard for small batch
    if args.mix_strategy in ("mixup","cutmix") and args.batch_size < 2:
        print(f"⚠️  batch_size={args.batch_size}: disabling {args.mix_strategy} (requires batch ≥ 2)")
        args.mix_strategy = "none"

    # model
    model=Mymodel(args).to(args.device)
    if args.pretrained_path and Path(args.pretrained_path).is_file():
        ckpt=torch.load(args.pretrained_path,map_location=args.device)
        model.load_state_dict(ckpt.get("state_dict",ckpt),strict=False)

    # optimizer & scheduler
    optim=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == 'cosine':
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=args.min_lr)
    else:
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode='max', factor=args.lr_factor, patience=args.lr_patience,
            threshold=1e-4, min_lr=args.min_lr, verbose=True)

    # AMP GradScaler (cross-version)
    scaler = make_grad_scaler(args)

    sys.stdout=Logger(run_dir/"train.log")
    writer=SummaryWriter(run_dir/"tb")

    # pick monitored metric name (pretty)
    monitor_name = "AP" if args.monitor == "ap" else "Recall"
    topk = TopKSaver(args.keep_topk, run_dir, metric_name=monitor_name)
    ema_obj = EMA(model, decay=args.ema_decay) if args.ema else None

    for ep in range(args.epochs):
        train_epoch(train_ld,model,optim,ep,args,writer,ema_obj,scaler)

        val_metrics = validate(model,val_ld,ep,writer,args,ema_obj)

        # choose value for scheduler/TopK based on monitor flag
        if args.monitor == "ap":
            monitored_value = float(val_metrics["ap"])
        else:  # recall
            monitored_value = float(val_metrics["recall"])

        if args.scheduler == 'cosine':
            scheduler.step()
        else:
            # plateau steps on the monitored metric
            scheduler.step(0.0 if (monitored_value != monitored_value) else monitored_value)

        saved, best_changed, _ = topk.maybe_save(
            metric_value=monitored_value, epoch=ep, state_dict=model.state_dict()
        )
        if saved:
            if best_changed:
                writer.add_scalar(f"Val/Best_{monitor_name}", topk.best, ep)
                print(f"  ↳ New BEST {monitor_name} {topk.best:.4f} → saved as best_{monitor_name.lower()}.pth")
            snap = ", ".join([f"e{e}:{monitor_name}{a:.4f}" for a,e,_ in topk.snapshot()])
            print(f"  ↳ Top-{args.keep_topk}: [{snap}]")
        else:
            print("  ↳ Not in top-K; no checkpoint saved this epoch.")

    # ----- Final Test Eval (using best checkpoint on monitored metric) -----
    best_path = run_dir / (f"best_{monitor_name.lower()}.pth")
    if test_ld is not None and best_path.is_file():
        print(f"\nEvaluating on TEST split using {best_path.name} …")
        best = torch.load(best_path, map_location=args.device)
        model.load_state_dict(best["state_dict"], strict=False)

        test_metrics = test_eval(model, test_ld, args.epochs, writer, args, ema_obj)

        with open(run_dir/"test_metrics.txt", "w") as f:
            f.write(f"Test Loss:     {test_metrics['loss']:.6f}\n")
            f.write(f"Test AP:       {test_metrics['ap']:.6f}\n")
            f.write(f"Test AUC:      {test_metrics['auc']:.6f}\n")
            f.write(f"Test mIoU:     {test_metrics['miou']:.6f}\n")
            f.write(f"Test Precision:{test_metrics['precision']:.6f}\n")
            f.write(f"Test Recall:   {test_metrics['recall']:.6f}\n")
            f.write(f"Test Accuracy: {test_metrics['accuracy']:.6f}\n")
            f.write(f"Test F1:       {test_metrics['f1']:.6f}\n")

        print(f"\n[TEST] AP={test_metrics['ap']:.4f}  AUC={test_metrics['auc']:.4f}  "
              f"mIoU={test_metrics['miou']:.4f}  P={test_metrics['precision']:.4f}  "
              f"R={test_metrics['recall']:.4f}  Acc={test_metrics['accuracy']:.4f}  "
              f"F1={test_metrics['f1']:.4f}")
    else:
        if test_ld is None:
            print("\nNo TEST split provided; skipping test evaluation.")
        else:
            print(f"\n{best_path.name} not found; skipping test evaluation.")

    writer.close()
    print("Training complete.")

if __name__=="__main__":
    torch.backends.cudnn.benchmark=True
    main()

