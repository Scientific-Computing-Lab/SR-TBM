#!/usr/bin/env python3
"""
Sigma-Logit-UNet++ — evaluation script (eval v3.9.0, 2025-08-20)
────────────────────────────────────────────────────────────
What's new vs v3.8.1:
• NEW: --prob_dir to evaluate from precomputed probability maps (no model).
• --ckpt is now optional when --prob_dir is used.
• Prob-loading supports .png/.jpg/.jpeg/.tif/.tiff (uint8→[0,1]) and .npy (float).
• In prob_dir mode: uncertainty/std are not produced; report columns left blank.

Existing features (unchanged):
• Per-image metrics in report.csv; threshold by --thresh_mode (recall / max_f1).
• Dataset-level PR & ROC curves (CSV + PNG).
• Probability maps, binary masks, and overlays (uncertainty when using model).
• Pooled Heyn for PRED and GT in summary.
• Optional random Heyn line placement with per-image deterministic seeds.
"""

# ───── standard libs ───── #
import argparse, os, sys, math, csv, datetime, random, hashlib, shutil
from pathlib import Path
from typing import Optional, Tuple

# ───── third-party libs ── #
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    roc_curve, precision_recall_curve
)

# Optional YAML
try:
    import yaml
except Exception:
    yaml = None

# ───── local model import (only if needed) ─ #
def _lazy_import_model():
    from model.sigma_logit_unetpp import Mymodel  # adjust if needed
    return Mymodel


# ═════════ helpers matching training ════════ #

def set_seed(seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def amp_autocast_ctx(args):
    if not (args.amp and args.device.type == "cuda"):
        from contextlib import nullcontext
        return nullcontext()
    try:
        import torch.amp as amp
        return amp.autocast('cuda', dtype=torch.float16)
    except Exception:
        return torch.cuda.amp.autocast()

def _softplus_std(std_raw: torch.Tensor) -> torch.Tensor:
    return (F.softplus(std_raw) + 1e-3).clamp(max=7.4)

def variance_corrected_sigmoid(logits: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits / torch.sqrt(1 + (math.pi * (std**2)) / 8))

def _make_unc_vis(single_std_map: torch.Tensor, mode="percentile", p=99.0, abs_max=1.0) -> torch.Tensor:
    x = single_std_map.float()
    if mode == "abs":
        scale = max(abs_max, 1e-6); return torch.clamp(x / scale, 0.0, 1.0)
    elif mode == "minmax":
        mn = float(x.min()); mx = float(x.max())
        return torch.clamp((x - mn) / (mx - mn + 1e-6), 0.0, 1.0)
    else:
        q = min(max(p, 1.0), 100.0) / 100.0
        pval = float(torch.quantile(x.flatten(), q))
        scale = max(pval, 1e-6); return torch.clamp(x / scale, 0.0, 1.0)

def forward_model(model, img):
    out = model(img)
    if isinstance(out, (list, tuple)) and len(out) >= 2:
        logits, std_raw = out[0], out[1]
    else:
        logits = out["mean"] if "mean" in out else out["logits"]
        std_raw = out["std"] if "std" in out else out["std_raw"]
    return logits, std_raw

def _compute_binary_metrics(gts_bin: np.ndarray, preds_prob: np.ndarray, thr: float):
    eps = 1e-12
    pred_bin = (preds_prob > thr)
    tp = np.logical_and(pred_bin, gts_bin == 1).sum()
    fp = np.logical_and(pred_bin, gts_bin == 0).sum()
    fn = np.logical_and(~pred_bin, gts_bin == 1).sum()
    tn = np.logical_and(~pred_bin, gts_bin == 0).sum()
    precision = tp / (tp + fp + eps); recall = tp / (tp + fn + eps)
    accuracy  = (tp + tn) / (tp + tn + fp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    inter = tp; union = (tp + fp + fn)
    miou = inter / (union + eps)
    return dict(precision=precision, recall=recall, accuracy=accuracy, f1=f1, miou=miou)

def _downsample_indices(n, max_points=2000):
    if n <= 0: return np.array([], dtype=int)
    if n <= max_points: return np.arange(n, dtype=int)
    return np.unique(np.linspace(0, n - 1, max_points).round().astype(int))

def _threshold_at_recall_target(y_true: np.ndarray, y_prob: np.ndarray, target: float):
    if y_true.max() == y_true.min():
        return np.nan, np.nan, np.nan
    precision, recall, thr = precision_recall_curve(y_true, y_prob)
    if thr.size == 0:
        return np.nan, np.nan, np.nan
    p1, r1, t = precision[1:], recall[1:], thr
    idx = np.where(r1 >= target)[0]
    if idx.size > 0:
        # among candidates: maximize precision; tie-breaker: highest recall
        best = idx[np.lexsort((-r1[idx], p1[idx]))][-1]
    else:
        best = int(np.argmax(r1))
    return float(t[best]), float(p1[best]), float(r1[best])

def _threshold_at_max_f1(y_true: np.ndarray, y_prob: np.ndarray):
    if y_true.max() == y_true.min():
        return np.nan, np.nan
    precision, recall, thr = precision_recall_curve(y_true, y_prob)
    if thr.size == 0:
        return np.nan, np.nan
    p1, r1, t = precision[1:], recall[1:], thr
    f1 = (2 * p1 * r1) / np.clip(p1 + r1, 1e-12, None)
    i = int(np.nanargmax(f1))
    return float(t[i]), float(f1[i])

def _save_u8_img(arr01: np.ndarray, out_path: Path):
    arr01 = np.clip(arr01, 0.0, 1.0)
    Image.fromarray((arr01 * 255.0 + 0.5).astype(np.uint8)).save(out_path)

def save_prob_map(prob: torch.Tensor, out_path: Path): _save_u8_img(prob.cpu().numpy(), out_path)
def save_unc_map(unc_vis: torch.Tensor, out_path: Path): _save_u8_img(unc_vis.cpu().numpy(), out_path)
def save_bin_mask(bin01: np.ndarray, out_path: Path): _save_u8_img(bin01.astype(np.float32), out_path)

def save_overlay(gt_bin_t: torch.Tensor, pred_bin_np: np.ndarray, out_path: Path):
    g = gt_bin_t.cpu().numpy().astype(np.uint8)
    p = pred_bin_np.astype(np.uint8)
    inter = (g & p).astype(np.uint8)
    gt_only = (g & (1 - p)).astype(np.uint8)
    pred_only = ((1 - g) & p).astype(np.uint8)
    H, W = g.shape
    overlay = np.zeros((H, W, 3), dtype=np.uint8)
    overlay[..., 0] = np.where(gt_only == 1, 255, overlay[..., 0])     # RED
    overlay[..., 1] = np.where(pred_only == 1, 255, overlay[..., 1])   # GREEN
    overlay[..., 0] = np.where(inter == 1, 255, overlay[..., 0])       # YELLOW
    overlay[..., 1] = np.where(inter == 1, 255, overlay[..., 1])
    Image.fromarray(overlay).save(out_path)


# ═════════ Heyn helpers ════════ #

def _count_one_runs(line01: np.ndarray) -> int:
    line = (line01.astype(np.uint8) > 0).astype(np.uint8)
    p = np.pad(line, (1, 1), mode="constant", constant_values=0)
    starts = (p[1:-1] == 1) & (p[:-2] == 0)
    return int(starts.sum())

def _stable_int_from_str(s: str) -> int:
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="little", signed=False)

def _pick_indices_random(lo: int, hi: int, n: int, rng: np.random.Generator):
    if lo > hi:
        lo, hi = 0, max(lo, hi)
    candidates = np.arange(lo, hi + 1, dtype=int)
    if candidates.size == 0:
        return np.array([], dtype=int)
    replace = candidates.size < n
    return rng.choice(candidates, size=n, replace=replace)

def heyn_lineal_intercept(edge01: np.ndarray,
                          num_lines: int = 20,
                          orientation: str = "both",
                          margin: float = 0.05,
                          skeleton: bool = False,
                          ys: Optional[np.ndarray] = None,
                          xs: Optional[np.ndarray] = None):
    e = edge01.astype(bool)
    if skeleton:
        e = skeletonize(e)
    H, W = e.shape
    lines_used = 0
    total_L, total_M = 0, 0

    def place_indices_fixed(n, lo, hi):
        if n <= 0: return np.array([], dtype=int)
        if lo > hi:
            lo, hi = 0, max(lo, hi)
        return np.linspace(lo, hi, n).round().astype(int)

    lo_y, hi_y = int(margin*(H-1)), int((1-margin)*(H-1))
    lo_x, hi_x = int(margin*(W-1)), int((1-margin)*(W-1))

    do_h = orientation in ("horizontal", "both")
    do_v = orientation in ("vertical", "both")

    if do_h:
        y_idx = ys if ys is not None else place_indices_fixed(num_lines, lo_y, hi_y)
        y_idx = np.clip(y_idx, 0, max(0, H-1))
        for y in y_idx:
            total_M += _count_one_runs(e[int(y), :])
            total_L += W
            lines_used += 1

    if do_v:
        x_idx = xs if xs is not None else place_indices_fixed(num_lines, lo_x, hi_x)
        x_idx = np.clip(x_idx, 0, max(0, W-1))
        for x in x_idx:
            total_M += _count_one_runs(e[:, int(x)])
            total_L += H
            lines_used += 1

    if total_M == 0:
        return float("nan"), 0, float(total_L), lines_used
    return float(total_L / total_M), int(total_M), float(total_L), lines_used


# ═════════ dataset ════════ #

class FolderEdgeDataset(Dataset):
    def __init__(self, img_dir, gt_dir, skeleton=False):
        self.imgs = sorted([p for p in Path(img_dir).glob("*") if p.is_file()])
        self.gts  = sorted([p for p in Path(gt_dir).glob("*")  if p.is_file()])
        assert len(self.imgs) == len(self.gts), f"Image/GT count mismatch: {len(self.imgs)} vs {len(self.gts)}"
        self.skeleton = skeleton

    def __len__(self): return len(self.imgs)

    def __getitem__(self, i):
        img_path = self.imgs[i]; gt_path = self.gts[i]
        img = Image.open(img_path).convert("RGB")
        gt  = Image.open(gt_path).convert("L")
        img_t = torch.from_numpy(np.array(img).transpose(2,0,1)).float() / 255.0   # (3,H,W)
        gt_np = (np.array(gt) > 127).astype(np.float32)
        if self.skeleton:
            gt_np = skeletonize(gt_np > 0.5).astype(np.float32)
        gt_t = torch.from_numpy(gt_np).unsqueeze(0)  # (1,H,W)
        return img_t, gt_t, str(img_path), str(gt_path)


# ═════════ Random-lines visualization (GT only; display) ════════ #

def _clip_line_to_image(cx, cy, theta, W, H):
    c, s = math.cos(theta), math.sin(theta)
    eps = 1e-9
    t_vals = []
    if abs(c) > eps:
        t = (0.0 - cx) / c
        y = cy + t * s
        if 0.0 <= y < H: t_vals.append(t)
        t = ((W - 1.0) - cx) / c
        y = cy + t * s
        if 0.0 <= y < H: t_vals.append(t)
    if abs(s) > eps:
        t = (0.0 - cy) / s
        x = cx + t * c
        if 0.0 <= x < W: t_vals.append(t)
        t = ((H - 1.0) - cy) / s
        x = cx + t * c
        if 0.0 <= x < W: t_vals.append(t)
    if len(t_vals) < 2:
        return None
    t_min, t_max = min(t_vals), max(t_vals)
    x0, y0 = cx + t_min * c, cy + t_min * s
    x1, y1 = cx + t_max * c, cy + t_max * s
    return (x0, y0, x1, y1)

def _sample_random_lines(H, W, n_lines, margin_frac, min_len_frac_diag, rng):
    lines = []
    diag = math.hypot(W, H)
    min_len = min_len_frac_diag * diag
    xmin, xmax = margin_frac * (W - 1), (1 - margin_frac) * (W - 1)
    ymin, ymax = margin_frac * (H - 1), (1 - margin_frac) * (H - 1)
    attempts_limit = 50 * n_lines
    attempts = 0
    while len(lines) < n_lines and attempts < attempts_limit:
        attempts += 1
        cx = rng.uniform(xmin, xmax) if xmax > xmin else (W - 1) / 2.0
        cy = rng.uniform(ymin, ymax) if ymax > ymin else (H - 1) / 2.0
        theta = rng.uniform(0.0, math.pi)
        seg = _clip_line_to_image(cx, cy, theta, W, H)
        if seg is None:
            continue
        x0, y0, x1, y1 = seg
        if math.hypot(x1 - x0, y1 - y0) >= min_len:
            lines.append((x0, y0, x1, y1))
    return lines

def _plot_random_lines_over_gt(gt_bin: np.ndarray,
                               out_path: Path,
                               n_lines: int,
                               margin_frac: float,
                               min_len_frac_diag: float,
                               seed: int):
    H, W = gt_bin.shape
    rng = np.random.default_rng(seed)
    lines = _sample_random_lines(H, W, n_lines, margin_frac, min_len_frac_diag, rng)

    fig = plt.figure(figsize=(max(4, W/256), max(4, H/256)))
    plt.imshow(gt_bin, cmap="gray", vmin=0, vmax=1)
    for (x0, y0, x1, y1) in lines:
        plt.plot([x0, x1], [y0, y1], linewidth=0.8)
    plt.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ═════════ probability-map loading (new) ════════ #

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
_NPY_EXTS = {".npy"}

def _load_prob_from_file(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext in _IMG_EXTS:
        arr = np.array(Image.open(path).convert("L"), dtype=np.uint8)  # 0..255
        return (arr.astype(np.float32) / 255.0)
    elif ext in _NPY_EXTS:
        arr = np.load(path)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        arr = arr.astype(np.float32)
        # robustly scale if values are not within [0,1]
        if np.nanmax(arr) > 1.0 or np.nanmin(arr) < 0.0:
            mn, mx = float(np.nanmin(arr)), float(np.nanmax(arr))
            if mx > mn:
                arr = (arr - mn) / (mx - mn)
            else:
                arr = np.clip(arr, 0.0, 1.0)
        return np.clip(arr, 0.0, 1.0)
    else:
        raise ValueError(f"Unsupported probability map extension: {path.suffix}")

def _find_prob_path(prob_dir: Path, base_stem: str) -> Optional[Path]:
    # Try common extensions, prefer PNG then NPY
    for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy"]:
        p = prob_dir / f"{base_stem}{ext}"
        if p.exists():
            return p
    # Fallback: any file whose stem matches
    for p in prob_dir.iterdir():
        if p.is_file() and p.stem == base_stem:
            return p
    return None


# ═════════ main eval ════════ #

def parse_args():
    p = argparse.ArgumentParser("Sigma-Logit-UNet++ — evaluator")
    p.add_argument("--image_dir", required=True, type=str)
    p.add_argument("--gt_dir",    required=True, type=str)

    # Model path is optional when using prob_dir
    p.add_argument("--ckpt",      default="", type=str,
                   help="Checkpoint to load. Optional if --prob_dir is provided.")
    p.add_argument("--prob_dir",  default="", type=str,
                   help="Directory with precomputed probability maps. "
                        "If set, model inference is skipped.")

    p.add_argument("--out_dir",   required=True, type=str)

    p.add_argument("--gpu",       default="0")
    p.add_argument("--amp",       action="store_true")
    p.add_argument("--skeletonize", action="store_true")
    p.add_argument("--use_variance_correction", action="store_true")

    # Thresholding mode
    p.add_argument("--thresh_mode", choices=["recall","max_f1"], default="recall",
                   help="Choose per-image thresholding strategy.")
    p.add_argument("--recall_target", type=float, default=0.90,
                   help="Target recall when --thresh_mode=recall.")

    # Std→unc visualization
    p.add_argument("--unc_vis_mode", choices=["percentile","minmax","abs"], default="percentile")
    p.add_argument("--unc_vis_p",    type=float, default=99.0)
    p.add_argument("--unc_vis_max",  type=float, default=1.0)
    p.add_argument("--distribution", default="gs", choices=["beta","gs","residual"])

    # Optional raw std saving
    p.add_argument("--save_std_raw_npy", action="store_true")

    # Curve CSV downsampling (dataset-level only)
    p.add_argument("--rocpr_points", type=int, default=2000, help="Max points written to curve CSVs.")

    # Regular Heyn parameters
    p.add_argument("--heyn_lines", type=int, default=20, help="Number of lines per orientation.")
    p.add_argument("--heyn_orientation", choices=["horizontal","vertical","both"], default="both")
    p.add_argument("--heyn_margin", type=float, default=0.05, help="Fraction to skip near borders.")
    p.add_argument("--heyn_skeleton", action="store_true", help="Skeletonize before Heyn (off by default).")

    # === NEW: Random placement for EVAL ===
    p.add_argument("--heyn_eval_random", action="store_true",
                   help="If set, place Heyn evaluation lines at random positions (within margins).")
    p.add_argument("--heyn_eval_seed", type=int, default=2025,
                   help="Base seed for random Heyn evaluation line placement (deterministic per image).")

    # === Random-lines visualization (GT only) ===
    p.add_argument("--heyn_random_vis", action="store_true",
                   help="If set, save a visualization of random straight lines over the GT mask.")
    p.add_argument("--heyn_random_count", type=int, default=60,
                   help="Number of random lines to draw on GT visualization.")
    p.add_argument("--heyn_random_seed", type=int, default=1337,
                   help="Seed for random-line visualization.")
    p.add_argument("--heyn_random_minlen", type=float, default=0.40,
                   help="Minimum line length as a fraction of image diagonal for random lines.")

    # Batch size
    p.add_argument("--batch_size", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    use_probdir = bool(args.prob_dir)
    prob_dir = Path(args.prob_dir) if use_probdir else None
    if use_probdir and not prob_dir.exists():
        raise FileNotFoundError(f"--prob_dir not found: {prob_dir}")

    if not use_probdir and not args.ckpt:
        raise ValueError("Either provide --ckpt for model inference or --prob_dir for external probabilities.")

    # Folder naming based on thresh mode
    mask_tag = f"recall{int(round(args.recall_target * 100))}" if args.thresh_mode == "recall" else "f1opt"

    out_dir = Path(args.out_dir)
    probs_dir        = out_dir / "probs";               probs_dir.mkdir(parents=True, exist_ok=True)
    unc_dir          = out_dir / "unc";                 unc_dir.mkdir(parents=True, exist_ok=True)
    std_raw_dir      = out_dir / "std_raw";             std_raw_dir.mkdir(parents=True, exist_ok=True)
    bin_dir          = out_dir / f"bin_{mask_tag}";     bin_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir      = out_dir / f"overlay_{mask_tag}"; overlay_dir.mkdir(parents=True, exist_ok=True)
    heyn_vis_gt_dir  = out_dir / "heyn_random_vis_gt"
    if args.heyn_random_vis:
        heyn_vis_gt_dir.mkdir(parents=True, exist_ok=True)

    # === Equivalence-ready CSVs ===
    heyn_pred_csv  = out_dir / "heyn_pred.csv"
    heyn_gt_csv    = out_dir / "heyn_gt.csv"
    heyn_pairs_csv = out_dir / "heyn_pairs.csv"
    hpred_f = open(heyn_pred_csv,  "w", newline=""); hpred_w  = csv.writer(hpred_f)
    hgt_f   = open(heyn_gt_csv,    "w", newline=""); hgt_w    = csv.writer(hgt_f)
    hpairs_f= open(heyn_pairs_csv, "w", newline=""); hpairs_w = csv.writer(hpairs_f)

    hpred_w.writerow([
        "image_id","image_path","gt_path","checkpoint_path",
        "thresh_mode","thr",
        "lbar","M","L_px","lines_used","orientation",
        "eval_random","eval_seed_img"
    ])
    hgt_w.writerow([
        "image_id","image_path","gt_path","checkpoint_path",
        "thresh_mode","thr",
        "lbar","M","L_px","lines_used","orientation",
        "eval_random","eval_seed_img"
    ])
    hpairs_w.writerow([
        "image_id",
        "pred_lbar","pred_M","pred_L_px",
        "gt_lbar","gt_M","gt_L_px",
        "thresh_mode","thr","orientation","lines_used",
        "eval_random","eval_seed_img"
    ])

    # Snapshot config
    cfg = vars(args).copy()
    cfg["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cfg["device"] = str(args.device)
    cfg["mask_tag"] = mask_tag
    if yaml is not None:
        with open(out_dir / "config.yaml", "w") as f: yaml.safe_dump(cfg, f, sort_keys=False)
    else:
        with open(out_dir / "config.yaml", "w") as f:
            for k,v in cfg.items(): f.write(f"{k}: {v}\n")

    # Data
    ds = FolderEdgeDataset(args.image_dir, args.gt_dir, skeleton=args.skeletonize)
    ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # Model (only if not using prob_dir)
    model = None
    if not use_probdir:
        Mymodel = _lazy_import_model()
        model = Mymodel(args).to(args.device)
        ckpt = torch.load(args.ckpt, map_location=args.device)
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        model.eval()

    # Accumulators (dataset-level)
    all_gts, all_probs = [], []

    # Pooled Heyn accumulators (pred & GT)
    heyn_pred_L_total = 0.0
    heyn_pred_M_total = 0
    heyn_pred_lines_total = 0
    heyn_pred_imgs_skipped = 0

    heyn_gt_L_total = 0.0
    heyn_gt_M_total = 0
    heyn_gt_lines_total = 0
    heyn_gt_imgs_zero_intercepts = 0

    # CSV paths
    report_csv = out_dir / "report.csv"
    roc_csv    = out_dir / "roc_curve.csv"
    pr_csv     = out_dir / "pr_curve.csv"

    with open(report_csv, "w", newline="") as frep:
        report_writer = csv.writer(frep)
        report_writer.writerow([
            "index", "image_path", "gt_path", "checkpoint_path",
            "prob_map_path", "unc_map_path",
            "thresh_mode", "thresh_param", "thr",
            "precision@thr", "recall@thr", "accuracy@thr", "f1@thr", "miou@thr",
            "average_precision", "roc_auc",
            "bin_mask_path", "overlay_path",
            # Heyn (PRED)
            "heyn_pred_lbar_px", "heyn_pred_intercepts", "heyn_pred_total_length_px",
            # Heyn (GT)
            "heyn_gt_lbar_px", "heyn_gt_intercepts", "heyn_gt_total_length_px",
            # Shared provenance
            "heyn_lines_used", "heyn_orientation",
            # Abs diff
            "heyn_abs_diff_px"
        ])

        idx_base = 0
        with torch.no_grad():
            for _, (img, gt, img_paths, gt_paths) in enumerate(ld):
                img = img.to(args.device)
                gt  = gt.to(args.device)

                # Inference or load probs
                if not use_probdir:
                    with amp_autocast_ctx(args):
                        logits, std_raw = forward_model(model, img)
                        std = _softplus_std(std_raw)
                        prob = (variance_corrected_sigmoid(logits, std)
                                if args.use_variance_correction else torch.sigmoid(logits))
                else:
                    # Build a batch tensor of probability maps aligned with 'img'
                    B, _, H, W = img.shape
                    prob_list = []
                    for b in range(B):
                        base = Path(img_paths[b]).stem
                        ppath = _find_prob_path(prob_dir, base)
                        if ppath is None:
                            raise FileNotFoundError(f"No prob map found in {prob_dir} for base name '{base}'")
                        p_np = _load_prob_from_file(ppath)
                        if p_np.shape != (H, W):
                            # resize to match image size
                            p_img = Image.fromarray((np.clip(p_np,0,1)*255.0+0.5).astype(np.uint8))
                            p_img = p_img.resize((W, H), resample=Image.BILINEAR)
                            p_np = np.array(p_img, dtype=np.uint8).astype(np.float32) / 255.0
                        prob_list.append(torch.from_numpy(p_np).unsqueeze(0))  # (1,H,W)
                    prob = torch.stack(prob_list, dim=0).to(args.device)  # (B,1,H,W)
                    std = None  # no uncertainty in prob_dir mode
                    std_raw = None

                B = img.size(0)
                for b in range(B):
                    i_global = idx_base + b
                    gtb   = (gt[b,0] > 0.5).float()
                    probb = prob[b,0].clamp(0,1).float()

                    base = Path(img_paths[b]).stem
                    prob_path = probs_dir / f"{base}.png"; save_prob_map(probb, prob_path)

                    # Uncertainty (only when model is used)
                    if (std is not None) and (std_raw is not None):
                        stdb  = std[b,0].float()
                        unc_vis  = _make_unc_vis(stdb, mode=args.unc_vis_mode, p=args.unc_vis_p, abs_max=args.unc_vis_max)
                        unc_path = unc_dir / f"{base}.png";    save_unc_map(unc_vis,  unc_path)
                        if args.save_std_raw_npy:
                            np.save(std_raw_dir / f"{base}.npy", std[b,0].detach().cpu().numpy())
                    else:
                        unc_path = ""  # leave empty in report

                    # Flatten for metrics
                    g_np = gtb.cpu().numpy().astype(np.uint8).reshape(-1)
                    p_np = probb.cpu().numpy().astype(np.float64).reshape(-1)
                    g_img = gtb.cpu().numpy().astype(np.uint8)  # (H,W)

                    # Optional GT random-lines visualization (display-only)
                    if args.heyn_random_vis:
                        vis_out = heyn_vis_gt_dir / f"{base}.png"
                        _plot_random_lines_over_gt(
                            g_img, vis_out,
                            n_lines=args.heyn_random_count,
                            margin_frac=args.heyn_margin,
                            min_len_frac_diag=args.heyn_random_minlen,
                            seed=args.heyn_random_seed
                        )

                    if g_np.max() == g_np.min():
                        ap_img = float('nan'); auc_img = float('nan')
                        thr = float('nan'); thr_param = float('nan')
                        m_at_thr = dict(precision=np.nan, recall=np.nan, accuracy=np.nan, f1=np.nan, miou=np.nan)
                        bin_path = ""; overlay_path = ""

                        # Decide eval line positions (deterministic per image if random mode)
                        eval_seed_img = None; ys_eval = xs_eval = None
                        if args.heyn_eval_random:
                            eval_seed_img = args.heyn_eval_seed ^ _stable_int_from_str(base)
                            rng = np.random.default_rng(eval_seed_img)
                            H, W = g_img.shape
                            lo_y, hi_y = int(args.heyn_margin*(H-1)), int((1-args.heyn_margin)*(H-1))
                            lo_x, hi_x = int(args.heyn_margin*(W-1)), int((1-args.heyn_margin)*(W-1))
                            if args.heyn_orientation in ("horizontal","both"):
                                ys_eval = _pick_indices_random(lo_y, hi_y, args.heyn_lines, rng)
                            if args.heyn_orientation in ("vertical","both"):
                                xs_eval = _pick_indices_random(lo_x, hi_x, args.heyn_lines, rng)

                        # Heyn GT
                        heyn_gt_lbar, heyn_gt_M, heyn_gt_L, heyn_lines = heyn_lineal_intercept(
                            g_img,
                            num_lines=args.heyn_lines,
                            orientation=args.heyn_orientation,
                            margin=args.heyn_margin,
                            skeleton=args.heyn_skeleton,
                            ys=ys_eval, xs=xs_eval
                        )
                        heyn_pred_lbar = float("nan"); heyn_pred_M = 0; heyn_pred_L = 0.0
                        heyn_abs_diff = float("nan")
                        heyn_pred_imgs_skipped += 1
                    else:
                        ap_img  = average_precision_score(g_np, p_np)
                        try:
                            auc_img = roc_auc_score(g_np, p_np)
                        except ValueError:
                            auc_img = float('nan')

                        if args.thresh_mode == "recall":
                            thr, _, _ = _threshold_at_recall_target(g_np, p_np, args.recall_target)
                            thr_param = args.recall_target
                        else:
                            thr, _ = _threshold_at_max_f1(g_np, p_np)
                            thr_param = float('nan')

                        # Decide eval line positions (deterministic per image if random mode)
                        eval_seed_img = None; ys_eval = xs_eval = None
                        if args.heyn_eval_random:
                            eval_seed_img = args.heyn_eval_seed ^ _stable_int_from_str(base)
                            rng = np.random.default_rng(eval_seed_img)
                            H, W = g_img.shape
                            lo_y, hi_y = int(args.heyn_margin*(H-1)), int((1-args.heyn_margin)*(H-1))
                            lo_x, hi_x = int(args.heyn_margin*(W-1)), int((1-args.heyn_margin)*(W-1))
                            if args.heyn_orientation in ("horizontal","both"):
                                ys_eval = _pick_indices_random(lo_y, hi_y, args.heyn_lines, rng)
                            if args.heyn_orientation in ("vertical","both"):
                                xs_eval = _pick_indices_random(lo_x, hi_x, args.heyn_lines, rng)

                        # Heyn GT
                        heyn_gt_lbar, heyn_gt_M, heyn_gt_L, heyn_lines = heyn_lineal_intercept(
                            g_img,
                            num_lines=args.heyn_lines,
                            orientation=args.heyn_orientation,
                            margin=args.heyn_margin,
                            skeleton=args.heyn_skeleton,
                            ys=ys_eval, xs=xs_eval
                        )

                        # Pred-dependent pieces
                        if not np.isnan(thr):
                            pred_bin_img = (p_np.reshape(gtb.shape) > thr).astype(np.uint8)  # (H,W)
                            m_at_thr = _compute_binary_metrics(g_np, p_np, thr=thr)
                            bin_path = (bin_dir / f"{base}.png");  save_bin_mask(pred_bin_img, bin_path)
                            overlay_path = (overlay_dir / f"{base}.png"); save_overlay(gtb, pred_bin_img, overlay_path)

                            heyn_pred_lbar, heyn_pred_M, heyn_pred_L, _ = heyn_lineal_intercept(
                                pred_bin_img,
                                num_lines=args.heyn_lines,
                                orientation=args.heyn_orientation,
                                margin=args.heyn_margin,
                                skeleton=args.heyn_skeleton,
                                ys=ys_eval, xs=xs_eval
                            )
                            heyn_pred_L_total += float(heyn_pred_L)
                            heyn_pred_M_total += int(heyn_pred_M)
                            heyn_pred_lines_total += int(heyn_lines)
                            heyn_abs_diff = (abs(heyn_pred_lbar - heyn_gt_lbar)
                                             if (not np.isnan(heyn_pred_lbar) and not np.isnan(heyn_gt_lbar))
                                             else float('nan'))
                        else:
                            m_at_thr = dict(precision=np.nan, recall=np.nan, accuracy=np.nan, f1=np.nan, miou=np.nan)
                            bin_path = ""; overlay_path = ""
                            heyn_pred_lbar = float("nan"); heyn_pred_M = 0; heyn_pred_L = 0.0
                            heyn_abs_diff = float("nan")
                            heyn_pred_imgs_skipped += 1

                    # Accumulate pooled Heyn (GT) always
                    heyn_gt_L_total += float(heyn_gt_L)
                    heyn_gt_M_total += int(heyn_gt_M)
                    heyn_gt_lines_total += int(heyn_lines)
                    if heyn_gt_M == 0:
                        heyn_gt_imgs_zero_intercepts += 1

                    # record provenance: if prob_dir mode, store that instead of ckpt
                    ckpt_or_probsrc = args.ckpt if not use_probdir else f"PROB_DIR={prob_dir}"

                    # report.csv row
                    report_writer.writerow([
                        i_global, img_paths[b], gt_paths[b], ckpt_or_probsrc,
                        str(prob_path), str(unc_path) if unc_path else "",
                        args.thresh_mode, f"{thr_param:.6f}" if not np.isnan(thr_param) else "nan",
                        f"{thr:.6f}" if not np.isnan(thr) else "nan",
                        f"{m_at_thr['precision']:.6f}" if not np.isnan(m_at_thr['precision']) else "nan",
                        f"{m_at_thr['recall']:.6f}"    if not np.isnan(m_at_thr['recall'])    else "nan",
                        f"{m_at_thr['accuracy']:.6f}"  if not np.isnan(m_at_thr['accuracy'])  else "nan",
                        f"{m_at_thr['f1']:.6f}"        if not np.isnan(m_at_thr['f1'])        else "nan",
                        f"{m_at_thr['miou']:.6f}"      if not np.isnan(m_at_thr['miou'])      else "nan",
                        f"{ap_img:.6f}" if not np.isnan(ap_img) else "nan",
                        f"{auc_img:.6f}" if not np.isnan(auc_img) else "nan",
                        str(bin_path) if bin_path else "",
                        str(overlay_path) if overlay_path else "",
                        # Heyn PRED
                        f"{heyn_pred_lbar:.6f}" if not np.isnan(heyn_pred_lbar) else "nan",
                        heyn_pred_M, f"{heyn_pred_L:.0f}",
                        # Heyn GT
                        f"{heyn_gt_lbar:.6f}" if not np.isnan(heyn_gt_lbar) else "nan",
                        heyn_gt_M, f"{heyn_gt_L:.0f}",
                        # Provenance
                        heyn_lines, args.heyn_orientation,
                        # Abs diff
                        f"{heyn_abs_diff:.6f}" if not np.isnan(heyn_abs_diff) else "nan",
                    ])

                    # === Equivalence CSV rows ===
                    thr_str = f"{thr:.6f}" if not np.isnan(thr) else "nan"
                    eval_seed_col = eval_seed_img if eval_seed_img is not None else ""
                    eval_rand_col = 1 if args.heyn_eval_random else 0

                    # PRED
                    hpred_w.writerow([
                        base, img_paths[b], gt_paths[b], ckpt_or_probsrc,
                        args.thresh_mode, thr_str,
                        f"{heyn_pred_lbar:.6f}" if not np.isnan(heyn_pred_lbar) else "nan",
                        heyn_pred_M, f"{heyn_pred_L:.0f}",
                        heyn_lines, args.heyn_orientation,
                        eval_rand_col, eval_seed_col
                    ])
                    # GT
                    hgt_w.writerow([
                        base, img_paths[b], gt_paths[b], ckpt_or_probsrc,
                        args.thresh_mode, thr_str,
                        f"{heyn_gt_lbar:.6f}" if not np.isnan(heyn_gt_lbar) else "nan",
                        heyn_gt_M, f"{heyn_gt_L:.0f}",
                        heyn_lines, args.heyn_orientation,
                        eval_rand_col, eval_seed_col
                    ])
                    # PAIRS
                    hpairs_w.writerow([
                        base,
                        f"{heyn_pred_lbar:.6f}" if not np.isnan(heyn_pred_lbar) else "nan",
                        heyn_pred_M, f"{heyn_pred_L:.0f}",
                        f"{heyn_gt_lbar:.6f}" if not np.isnan(heyn_gt_lbar) else "nan",
                        heyn_gt_M, f"{heyn_gt_L:.0f}",
                        args.thresh_mode, thr_str, args.heyn_orientation, heyn_lines,
                        eval_rand_col, eval_seed_col
                    ])

                    # Accumulate for dataset-level curves/threshold
                    all_gts.append(g_np)
                    all_probs.append(p_np)

                idx_base += B

    # Close files
    hpred_f.close(); hgt_f.close(); hpairs_f.close()

    # Dataset-level metrics & curves (micro)
    all_gts  = np.concatenate(all_gts, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    if all_gts.max() == all_gts.min():
        ap_ds = float('nan'); auc_ds = float('nan')
        roc_fpr_ds = np.array([]); roc_tpr_ds = np.array([]); roc_thr_ds = np.array([])
        pr_prec_ds = np.array([]); pr_rec_ds = np.array([]); pr_thr_ds = np.array([])
        thr_ds = float('nan'); thr_param_ds = float('nan')
        m_ds = dict(precision=np.nan, recall=np.nan, accuracy=np.nan, f1=np.nan, miou=np.nan)
    else:
        ap_ds  = average_precision_score(all_gts, all_probs)
        try:
            auc_ds = roc_auc_score(all_gts, all_probs)
        except ValueError:
            auc_ds = float('nan')
        roc_fpr_ds, roc_tpr_ds, roc_thr_ds = roc_curve(all_gts, all_probs)
        pr_prec_ds, pr_rec_ds, pr_thr_ds   = precision_recall_curve(all_gts, all_probs)

        if args.thresh_mode == "recall":
            thr_ds, _, _ = _threshold_at_recall_target(all_gts, all_probs, args.recall_target)
            thr_param_ds = args.recall_target
        else:
            thr_ds, _    = _threshold_at_max_f1(all_gts, all_probs)
            thr_param_ds = float('nan')

        if not np.isnan(thr_ds):
            m_ds = _compute_binary_metrics(all_gts.astype(np.uint8), all_probs.astype(np.float64), thr=thr_ds)
        else:
            m_ds = dict(precision=np.nan, recall=np.nan, accuracy=np.nan, f1=np.nan, miou=np.nan)

        # Save dataset-level curves to CSV (downsampled)
        if roc_fpr_ds.size > 0:
            idx_ds = _downsample_indices(roc_fpr_ds.size, args.rocpr_points)
            with open(roc_csv, "w", newline="") as froc:
                roc_writer = csv.writer(froc)
                roc_writer.writerow(["point_idx", "fpr", "tpr", "threshold"])
                thr_pad = np.concatenate([roc_thr_ds, [np.nan]])  # align sizes
                for j, k in enumerate(idx_ds):
                    roc_writer.writerow([j, float(roc_fpr_ds[k]), float(roc_tpr_ds[k]),
                                         "" if np.isnan(thr_pad[k]) else float(thr_pad[k])])
        else:
            with open(roc_csv, "w", newline="") as froc:
                csv.writer(froc).writerow(["point_idx", "fpr", "tpr", "threshold"])

        if pr_thr_ds.size > 0:
            idx_ds = _downsample_indices(pr_thr_ds.size, args.rocpr_points)
            with open(pr_csv, "w", newline="") as fprc:
                pr_writer = csv.writer(fprc)
                pr_writer.writerow(["point_idx", "precision", "recall", "threshold"])
                p1, r1, t1 = pr_prec_ds[1:], pr_rec_ds[1:], pr_thr_ds
                for j, k in enumerate(idx_ds):
                    pr_writer.writerow([j, float(p1[k]), float(r1[k]), float(t1[k])])
        else:
            with open(pr_csv, "w", newline="") as fprc:
                csv.writer(fprc).writerow(["point_idx", "precision", "recall", "threshold"])

        # Plot & save PR curve
        if pr_prec_ds.size > 0:
            try:
                fig = plt.figure(figsize=(6, 5))
                plt.plot(pr_rec_ds, pr_prec_ds, lw=1)
                plt.xlabel("Recall"); plt.ylabel("Precision")
                plt.title(f"Dataset Precision-Recall (AP={ap_ds:.3f})")
                plt.xlim(0, 1); plt.ylim(0, 1)
                plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
                plt.tight_layout()
                plt.savefig(out_dir / "pr_curve.png", dpi=180)
                plt.close(fig)
            except Exception as e:
                print(f"[warn] Failed to save PR plot: {e}")

        # Plot & save ROC curve
        if roc_fpr_ds.size > 0:
            try:
                fig = plt.figure(figsize=(6, 5))
                plt.plot(roc_fpr_ds, roc_tpr_ds, lw=1, label=f"AUC={auc_ds:.3f}" if not np.isnan(auc_ds) else "AUC=nan")
                plt.plot([0, 1], [0, 1], linestyle="--", lw=1)
                plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
                plt.title("Dataset ROC Curve"); plt.legend()
                plt.xlim(0, 1); plt.ylim(0, 1)
                plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
                plt.tight_layout()
                plt.savefig(out_dir / "roc_curve.png", dpi=180)
                plt.close(fig)
            except Exception as e:
                print(f"[warn] Failed to save ROC plot: {e}")

    # Pooled Heyn (pred & GT)
    heyn_pred_lbar_pooled = (heyn_pred_L_total / heyn_pred_M_total) if heyn_pred_M_total > 0 else float('nan')
    heyn_gt_lbar_pooled   = (heyn_gt_L_total   / heyn_gt_M_total)   if heyn_gt_M_total > 0   else float('nan')
    heyn_pooled_abs_diff  = (abs(heyn_pred_lbar_pooled - heyn_gt_lbar_pooled)
                             if (not np.isnan(heyn_pred_lbar_pooled) and not np.isnan(heyn_gt_lbar_pooled))
                             else float('nan'))

    # Append dataset-level summary row to report.csv
    with open(report_csv, "a", newline="") as frep:
        report_writer = csv.writer(frep)
        report_writer.writerow([])
        report_writer.writerow([
            "SUMMARY",
            f"image_dir={args.image_dir}",
            f"gt_dir={args.gt_dir}",
            f"ckpt={args.ckpt if args.ckpt else '—'}",
            "-", "-",
            args.thresh_mode,
            f"{thr_param_ds:.6f}" if not np.isnan(thr_param_ds) else "nan",
            f"{thr_ds:.6f}" if not np.isnan(thr_ds) else "nan",
            f"{m_ds['precision']:.6f}" if not np.isnan(m_ds['precision']) else "nan",
            f"{m_ds['recall']:.6f}"    if not np.isnan(m_ds['recall'])    else "nan",
            f"{m_ds['accuracy']:.6f}"  if not np.isnan(m_ds['accuracy'])  else "nan",
            f"{m_ds['f1']:.6f}"        if not np.isnan(m_ds['f1'])        else "nan",
            f"{m_ds['miou']:.6f}"      if not np.isnan(m_ds['miou'])      else "nan",
            f"{ap_ds:.6f}" if not np.isnan(ap_ds) else "nan",
            f"{auc_ds:.6f}" if not np.isnan(auc_ds) else "nan",
            "-", "-",
            f"{heyn_pred_lbar_pooled:.6f}" if not np.isnan(heyn_pred_lbar_pooled) else "nan",
            heyn_pred_M_total, f"{heyn_pred_L_total:.0f}",
            f"{heyn_gt_lbar_pooled:.6f}" if not np.isnan(heyn_gt_lbar_pooled) else "nan",
            heyn_gt_M_total, f"{heyn_gt_L_total:.0f}",
            heyn_pred_lines_total + heyn_gt_lines_total,
            args.heyn_orientation,
            f"{heyn_pooled_abs_diff:.6f}" if not np.isnan(heyn_pooled_abs_diff) else "nan",
        ])

    # Console
    print("\n=== DATASET (micro; threshold by", args.thresh_mode, ") ===")
    print(f"AP            : {ap_ds:.6f}" if not np.isnan(ap_ds) else "AP            : nan")
    print(f"ROC-AUC       : {auc_ds:.6f}" if not np.isnan(auc_ds) else "ROC-AUC       : nan")
    print(f"thr           : {thr_ds:.6f}" if not np.isnan(thr_ds) else "thr           : nan")
    print(f"Precision@thr : {m_ds['precision']:.6f}" if not np.isnan(m_ds['precision']) else "Precision@thr : nan")
    print(f"Recall@thr    : {m_ds['recall']:.6f}"    if not np.isnan(m_ds['recall'])    else "Recall@thr    : nan")
    print(f"Accuracy@thr  : {m_ds['accuracy']:.6f}"  if not np.isnan(m_ds['accuracy'])  else "Accuracy@thr  : nan")
    print(f"F1@thr        : {m_ds['f1']:.6f}"        if not np.isnan(m_ds['f1'])        else "F1@thr        : nan")
    print(f"mIoU@thr      : {m_ds['miou']:.6f}"      if not np.isnan(m_ds['miou'])      else "mIoU@thr      : nan")
    if use_probdir:
        print("\n[mode] Using precomputed probabilities from:", prob_dir)

    print("\nHeyn pooled (PRED):",
          f"lbar={heyn_pred_lbar_pooled:.6f} px (M={heyn_pred_M_total}, L={heyn_pred_L_total:.0f} px)"
          if heyn_pred_M_total > 0 else "nan (no pred masks / no intercepts)")
    print("Heyn pooled (GT)  :",
          f"lbar={heyn_gt_lbar_pooled:.6f} px (M={heyn_gt_M_total}, L={heyn_gt_L_total:.0f} px)"
          if heyn_gt_M_total > 0 else "nan (no GT intercepts)")
    if not np.isnan(heyn_pooled_abs_diff):
        print(f"Heyn pooled Δ|pred-gt| : {heyn_pooled_abs_diff:.6f} px")

    print(f"\nSaved to: {out_dir}")
    print(f"  • Probability maps : {probs_dir}")
    if not use_probdir:
        print(f"  • Uncertainty maps : {unc_dir}")
        if args.save_std_raw_npy:
            print(f"  • Raw std (npy)    : {std_raw_dir}")
    print(f"  • Bin masks ({mask_tag}) : {bin_dir}")
    print(f"  • Overlays  ({mask_tag}) : {overlay_dir}")
    if args.heyn_random_vis:
        print(f"  • Random-lines GT  : {heyn_vis_gt_dir}")
    print(f"  • Report CSV       : {report_csv}")
    print(f"  • PR curve CSV/PNG : {pr_csv}, {out_dir/'pr_curve.png'}")
    print(f"  • ROC curve CSV/PNG: {roc_csv}, {out_dir/'roc_curve.png'}")
    print(f"  • Heyn PRED CSV    : {heyn_pred_csv}")
    print(f"  • Heyn GT   CSV    : {heyn_gt_csv}")
    print(f"  • Heyn PAIRS CSV   : {heyn_pairs_csv}")
    if args.heyn_eval_random:
        print(f"  • Heyn EVAL LINES  : random (seed base={args.heyn_eval_seed}, per-image deterministic)")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()

