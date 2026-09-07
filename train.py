#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from scipy.ndimage import distance_transform_edt, binary_erosion
from skimage.morphology import skeletonize

from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import cv2
from PIL import Image
import matplotlib.pyplot as plt


# =========================
# CFG：只改这里
# =========================
@dataclass
class CFG:
    # ---------- 设备/基础 ----------
    SEED: int = 42
    IMG_SIZE: Tuple[int, int] = (512, 512)
    THRESHOLD: float = 0.5

    # ---------- 预处理：原始数据 -> data/xxx ----------
    CLEAN_EXISTING: bool = True
    ORIGINAL_DATA_PATHS = [
        # stage1: 粗标注
        ('original_data/stage1/images_8000', 'original_data/stage1/masks_8000',
         'data/stage1/images', 'data/stage1/masks_coarse'),
        # stage2: 精标注
        ('original_data/stage2/images', 'original_data/stage2/masks_fine',
         'data/stage2/images', 'data/stage2/masks_fine'),
        # test: 精标注
        ('original_data/test/images', 'original_data/test/masks_fine',
         'data/test/images', 'data/test/masks_fine')
    ]

    # ---------- 数据路径（训练时用 data/xxx） ----------
    STAGE1_IMAGES: str = 'data/stage1/images'
    STAGE1_MASKS: str = 'data/stage1/masks_coarse'
    STAGE2_IMAGES: str = 'Dataset/LES/train/images'
    STAGE2_MASKS: str = 'Dataset/LES/train/gt'
    TEST_IMAGES: str = 'Dataset/LES/test/images'
    TEST_MASKS: str = 'Dataset/LES/test/gt'

    # ---------- DataLoader ----------
    BATCH_SIZE: int = 4
    NUM_WORKERS: int = 0  # Windows 建议 0

    # ---------- Stage1：可选加载已训练好的 ----------
    USE_PRETRAINED_STAGE1: bool = True
    PRETRAINED_STAGE1_PATH: str = 'models_clean/best_stage1_clean.pth'
    STAGE1_BASE_CH: int = 64  # 必须和你保存的 stage1 权重结构一致

    # ---------- Stage2：通道可配 ----------
    STAGE2_BASE_CH: int = 32  # 64/32/16...

    # ---------- 是否冻结 Stage1 在 Stage2 ----------
    FREEZE_STAGE1_IN_STAGE2: bool = False  # False=微调 Stage1（推荐）

    # ---------- 学习率 ----------
    LR_STAGE1: float = 1e-4
    LR_STAGE2_STAGE1_FINETUNE: float = 1e-5
    LR_STAGE2: float = 1e-4

    # ---------- epoch/早停 ----------
    EPOCHS_STAGE1: int = 100
    EPOCHS_STAGE2: int = 200
    PATIENCE_STAGE1: int = 10
    PATIENCE_STAGE2: int = 10
    MIN_DELTA_DICE: float = 1e-4  # dice 提升阈值（用于早停）

    # ---------- Loss 权重 ----------
    LOSS_BCE_W: float = 0.5
    LOSS_DICE_W: float = 0.5  # 建议 0.5/0.5

    # ---------- 输出 ----------
    MODELS_DIR: str = 'models'
    RESULTS_DIR: str = 'results_perimage'
    BEST_STAGE1_PATH: str = 'models/best_stage1_model.pth'
    BEST_STAGE2_PATH: str = 'models/best_stage2_model.pth'


# =========================
# Utils
# =========================
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def preprocess_and_match_data(original_images_path, original_masks_path, output_images_path, output_masks_path,
                              clean_existing=True):
    """把匹配到的 image/mask 复制到 data/xxx 下"""
    if clean_existing:
        if os.path.exists(output_images_path):
            shutil.rmtree(output_images_path)
        if os.path.exists(output_masks_path):
            shutil.rmtree(output_masks_path)

    os.makedirs(output_images_path, exist_ok=True)
    os.makedirs(output_masks_path, exist_ok=True)

    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    image_files = [f for f in os.listdir(original_images_path) if f.lower().endswith(exts)]
    mask_files = [f for f in os.listdir(original_masks_path) if f.lower().endswith(exts)]

    used_masks = set()
    pairs = []

    for imgf in image_files:
        stem = os.path.splitext(imgf)[0]
        img_ext = os.path.splitext(imgf)[1]
        cand1 = stem + "_mask" + img_ext
        cand2 = stem + img_ext

        found = None
        for mf in mask_files:
            if mf in used_masks:
                continue
            mstem = os.path.splitext(mf)[0]
            if mf == cand1 or mf == cand2 or mstem == stem or mstem == stem + "_mask":
                found = mf
                break

        if found is not None:
            pairs.append((imgf, found))
            used_masks.add(found)

    if len(pairs) == 0:
        raise ValueError(f"没有找到匹配对: {original_images_path}")

    for imgf, mf in pairs:
        shutil.copy2(os.path.join(original_images_path, imgf), os.path.join(output_images_path, imgf))
        shutil.copy2(os.path.join(original_masks_path, mf), os.path.join(output_masks_path, mf))

    return len(pairs)


class FeatureAlign(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.align = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.align(x)


# =========================
# Model: UNet(base_ch可配)
# =========================
class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, base_ch=64):
        super().__init__()

        c1 = base_ch
        c2 = base_ch * 2
        c3 = base_ch * 4
        c4 = base_ch * 8
        c5 = base_ch * 16

        self.enc1 = self.block(in_channels, c1)
        self.enc2 = self.block(c1, c2)
        self.enc3 = self.block(c2, c3)
        self.enc4 = self.block(c3, c4)

        self.middle = self.block(c4, c5)

        self.dec4 = self.block(c5 + c4, c4)
        self.dec3 = self.block(c4 + c3, c3)
        self.dec2 = self.block(c3 + c2, c2)
        self.dec1 = self.block(c2 + c1, c1)

        self.final = nn.Conv2d(c1, out_channels, 1)

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, return_feature=False, return_logit=False):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        m = self.middle(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up(m), e4], 1))
        d3 = self.dec3(torch.cat([self.up(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], 1))

        logit = self.final(d1)
        prob = torch.sigmoid(logit)

        if return_feature and return_logit:
            return prob, d1, logit

        if return_feature:
            return prob, d1

        return prob


class TwoStageUNet(nn.Module):
    def __init__(self, stage1_model, stage1_ch=64, stage2_base_ch=32, freeze_stage1=False):
        super().__init__()
        self.stage1 = stage1_model
        self.freeze_stage1 = freeze_stage1
        self.align = FeatureAlign(stage1_ch, stage1_ch)

        in_ch = 1 + 1 + stage1_ch
        self.stage2 = UNet(
            in_channels=in_ch,
            out_channels=1,
            base_ch=stage2_base_ch
        )

        for p in self.stage1.parameters():
            p.requires_grad = not freeze_stage1

    def forward(self, x):
        if self.freeze_stage1:
            with torch.no_grad():
                p1, f1, logit1 = self.stage1(x, return_feature=True, return_logit=True)
        else:
            p1, f1, logit1 = self.stage1(x, return_feature=True, return_logit=True)

        f1 = self.align(f1)
        x2 = torch.cat([x, p1, f1], dim=1)
        residual = self.stage2(x2)
        p2 = torch.sigmoid(logit1 + residual)
        return p1, p2


# =========================
# Dataset (使用 PIL 安全读取，修复 Transform 参数报错)
# =========================
class VesselDataset(Dataset):
    def __init__(self, images_path, masks_path, transform=None, stage=1):
        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.stage = stage

        if not os.path.exists(images_path):
            raise ValueError(f"图像路径不存在: {images_path}")
        if not os.path.exists(masks_path):
            raise ValueError(f"掩码路径不存在: {masks_path}")

        self.images = sorted(
            [f for f in os.listdir(images_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])
        self.masks = sorted(
            [f for f in os.listdir(masks_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])

        if len(self.images) == 0:
            raise ValueError(f"在路径中没有找到图像文件: {images_path}")
        if len(self.masks) == 0:
            raise ValueError(f"在路径中没有找到掩码文件: {masks_path}")

        print(f"找到图像文件: {len(self.images)} 个")
        print(f"找到掩码文件: {len(self.masks)} 个")

        # ---------------------
        # 文件安全匹配逻辑
        # ---------------------
        mask_dict = {os.path.splitext(f)[0]: f for f in self.masks}
        self.matched_pairs = []

        for img_file in self.images:
            img_name = os.path.splitext(img_file)[0]
            found_mask = mask_dict.get(img_name)

            if not found_mask:
                for s in ["_mask", "_manual", "_label", "_gt", "_seg", "_1stHO"]:
                    if img_name + s in mask_dict:
                        found_mask = mask_dict[img_name + s]
                        break

            if not found_mask:
                for m_name, m_file in mask_dict.items():
                    if m_name.startswith(img_name + "_") or m_name.startswith(img_name + " "):
                        found_mask = m_file
                        break

            if found_mask:
                self.matched_pairs.append((img_file, found_mask))

        if len(self.matched_pairs) == 0:
            raise ValueError("无法匹配图像和掩码文件")

        print(f"第{stage}阶段数据集加载成功: {len(self.matched_pairs)} 对图像-掩码")

    def __len__(self):
        return len(self.matched_pairs)

    def __getitem__(self, idx):
        image_name, mask_name = self.matched_pairs[idx]
        image_path = os.path.join(self.images_path, image_name)
        mask_path = os.path.join(self.masks_path, mask_name)

        try:
            # 1. 弃用 cv2，改用 PIL 强行读取并转换为灰度模式 ("L")
            image_pil = Image.open(image_path).convert("L")

            # 2. 读取 Mask，保持原格式避免破坏彩色数据
            mask_pil = Image.open(mask_path)
            mask_np = np.array(mask_pil)

            # 3. 彩色动静脉标签保护 (跨通道融合)
            if len(mask_np.shape) >= 3:
                mask_np = np.sum(mask_np, axis=2)

            # 4. 强制二值化为 0 和 255
            mask_np = (mask_np > 0).astype(np.uint8) * 255
            mask_pil = Image.fromarray(mask_np)

            # 5. 执行数据增强和 Tensor 转换
            if self.transform:
                image_tensor = self.transform(image_pil)
                mask_tensor = self.transform(mask_pil)
            else:
                image_tensor = T.ToTensor()(image_pil)
                mask_tensor = T.ToTensor()(mask_pil)

            return image_tensor, mask_tensor

        except Exception as e:
            print(f"🚨 PIL处理图像 {image_name} 失败: {e}")
            return torch.zeros(1, 512, 512), torch.zeros(1, 512, 512)


# =========================
# Loss: BCE + SoftDiceLoss (per-image)
# =========================
class SoftDiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, prob: torch.Tensor, target: torch.Tensor):
        b = prob.shape[0]
        p = prob.view(b, -1)
        t = target.view(b, -1)

        inter = (p * t).sum(dim=1)
        den = p.sum(dim=1) + t.sum(dim=1)
        dice = (2 * inter + self.eps) / (den + self.eps)
        loss = 1.0 - dice
        return loss.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_w=0.5, dice_w=0.5):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = SoftDiceLoss()
        self.bce_w = bce_w
        self.dice_w = dice_w

    def forward(self, prob, target):
        return self.bce_w * self.bce(prob, target) + self.dice_w * self.dice(prob, target)


# =========================
# per-image macro metrics（逐张算再平均）
# =========================
@torch.no_grad()
def cldice(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    skel_pred = skeletonize(pred)
    skel_gt = skeletonize(gt)
    tprec = (skel_pred & gt).sum() / (skel_pred.sum() + 1e-7)
    tsens = (skel_gt & pred).sum() / (skel_gt.sum() + 1e-7)
    cl = 2 * tprec * tsens / (tprec + tsens + 1e-7)
    return cl


def get_surface(binary_mask):
    """提取边缘的辅助函数，用于修正 HD95 的计算"""
    return binary_mask ^ binary_erosion(binary_mask)


def hd95(pred, gt, default_value=np.nan):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    # 双方都全黑，完美匹配，距离为0
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    # 一方有，一方没有，完全预测失败，返回 np.nan 避免拉低平均分
    if pred.sum() == 0 or gt.sum() == 0:
        return default_value

    # 提取单像素边缘
    surf_pred = get_surface(pred)
    surf_gt = get_surface(gt)

    # 生成基于边缘的距离场
    dt_pred = distance_transform_edt(~surf_pred)
    dt_gt = distance_transform_edt(~surf_gt)

    # 严格计算边缘点到对方边缘的距离
    sds1 = dt_gt[surf_pred]
    sds2 = dt_pred[surf_gt]

    all_sds = np.concatenate([sds1, sds2])

    return np.percentile(all_sds, 95)


def metrics_per_image_macro(prob: torch.Tensor, target: torch.Tensor, threshold=0.5, compute_cldice=False,
                            compute_hd95=False) -> Dict[str, float]:
    pred = (prob > threshold).float()
    gt = (target > 0.5).float()

    b = pred.shape[0]
    p = pred.view(b, -1)
    t = gt.view(b, -1)

    tp = (p * t).sum(dim=1)
    fp = (p * (1 - t)).sum(dim=1)
    fn = ((1 - p) * t).sum(dim=1)
    tn = ((1 - p) * (1 - t)).sum(dim=1)

    eps = 1e-7
    acc = (tp + tn) / (tp + fp + fn + tn + eps)
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    cldice_val = np.nan
    hd95_val = np.nan

    if compute_cldice or compute_hd95:
        cl_list = []
        hd_list = []

        for i in range(b):
            pred_np = pred[i, 0].cpu().numpy().astype(bool)
            gt_np = gt[i, 0].cpu().numpy().astype(bool)
            if compute_cldice:
                cl_list.append(cldice(pred_np, gt_np))
            if compute_hd95:
                hd_list.append(hd95(pred_np, gt_np))

        if compute_cldice:
            cldice_val = np.nanmean(cl_list)
        if compute_hd95:
            hd95_val = np.nanmean(hd_list)

    return {
        "acc": acc.mean().item(),
        "precision": prec.mean().item(),
        "recall": rec.mean().item(),
        "f1": f1.mean().item(),
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "cldice": float(cldice_val),
        "hd95": float(hd95_val)
    }


# =========================
# train/val/test
# =========================
def run_one_epoch(model, loader, optimizer, device, loss_fn, threshold, train: bool, desc: str):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    sum_m = {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "dice": 0.0, "iou": 0.0, "cldice": 0.0,
             "hd95": 0.0}
    n_batches = 0

    for imgs, msks in tqdm(loader, desc=desc):
        imgs = imgs.to(device, non_blocking=True)
        msks = msks.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad()
            out = model(imgs)
            if isinstance(out, torch.Tensor):
                prob = out
                loss = loss_fn(prob, msks)
            else:
                p1, p2 = out
                loss1 = loss_fn(p1, msks)
                loss2 = loss_fn(p2, msks)
                loss = 0.3 * loss1 + 0.7 * loss2
                prob = p2
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        else:
            with torch.no_grad():
                out = model(imgs)
                if isinstance(out, tuple):
                    prob = out[1]
                else:
                    prob = out

                # ---------------------
                # 仅修复 Loss 打印为 0 的问题
                # ---------------------
                loss = loss_fn(prob, msks)
                total_loss += loss.item()

        if train:
            m = metrics_per_image_macro(prob, msks, threshold=threshold, compute_cldice=False, compute_hd95=False)
        else:
            m = metrics_per_image_macro(prob, msks, threshold=threshold, compute_cldice=True, compute_hd95=True)

        for k in sum_m:
            sum_m[k] += m[k]
        n_batches += 1

    avg_loss = total_loss / max(1, n_batches)
    avg_m = {k: sum_m[k] / max(1, n_batches) for k in sum_m}
    return avg_loss, avg_m


def split_dataset(dataset, train_ratio=0.8, seed=42):
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = n - n_train
    torch.manual_seed(seed)
    return random_split(dataset, [n_train, n_val])


def save_json(obj: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def plot_curves(history: Dict[str, Any], out_png: str):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure(figsize=(14, 8))

    # loss
    plt.subplot(2, 3, 1)
    plt.plot(history["stage1"]["train_loss"], label="S1 train")
    plt.plot(history["stage1"]["val_loss"], label="S1 val")
    plt.plot(range(len(history["stage1"]["train_loss"]),
                   len(history["stage1"]["train_loss"]) + len(history["stage2"]["train_loss"])),
             history["stage2"]["train_loss"], label="S2 train")
    plt.plot(range(len(history["stage1"]["val_loss"]),
                   len(history["stage1"]["val_loss"]) + len(history["stage2"]["val_loss"])),
             history["stage2"]["val_loss"], label="S2 val")
    plt.title("Loss")
    plt.legend()
    plt.grid(True)

    # dice
    plt.subplot(2, 3, 2)
    s1_tr = [m["dice"] for m in history["stage1"]["train_metrics"]]
    s1_va = [m["dice"] for m in history["stage1"]["val_metrics"]]
    s2_tr = [m["dice"] for m in history["stage2"]["train_metrics"]]
    s2_va = [m["dice"] for m in history["stage2"]["val_metrics"]]
    plt.plot(s1_tr, label="S1 train")
    plt.plot(s1_va, label="S1 val")
    plt.plot(range(len(s1_tr), len(s1_tr) + len(s2_tr)), s2_tr, label="S2 train")
    plt.plot(range(len(s1_va), len(s1_va) + len(s2_va)), s2_va, label="S2 val")
    plt.title("Dice (per-image macro)")
    plt.legend()
    plt.grid(True)

    # iou
    plt.subplot(2, 3, 3)
    s1_tr = [m["iou"] for m in history["stage1"]["train_metrics"]]
    s1_va = [m["iou"] for m in history["stage1"]["val_metrics"]]
    s2_tr = [m["iou"] for m in history["stage2"]["train_metrics"]]
    s2_va = [m["iou"] for m in history["stage2"]["val_metrics"]]
    plt.plot(s1_tr, label="S1 train")
    plt.plot(s1_va, label="S1 val")
    plt.plot(range(len(s1_tr), len(s1_tr) + len(s2_tr)), s2_tr, label="S2 train")
    plt.plot(range(len(s1_va), len(s1_va) + len(s2_va)), s2_va, label="S2 val")
    plt.title("IoU (per-image macro)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


# =========================
# Main
# =========================
if __name__ == "__main__":
    cfg = CFG()
    set_seed(cfg.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    for orig_img, orig_msk, out_img, out_msk in cfg.ORIGINAL_DATA_PATHS:
        if os.path.exists(orig_img) and os.path.exists(orig_msk):
            print(f"[PRE] {orig_img} -> {out_img}")
            n = preprocess_and_match_data(orig_img, orig_msk, out_img, out_msk, clean_existing=cfg.CLEAN_EXISTING)
            print(f"[PRE] matched={n}")
        else:
            print(f"[PRE] skip (missing): {orig_img} or {orig_msk}")

    # 保持你最原始的共享 Transform 逻辑
    transform = T.Compose([T.Resize(cfg.IMG_SIZE), T.ToTensor()])

    loss_fn = BCEDiceLoss(bce_w=cfg.LOSS_BCE_W, dice_w=cfg.LOSS_DICE_W)

    history = {
        "stage1": {"train_loss": [], "val_loss": [], "train_metrics": [], "val_metrics": []},
        "stage2": {"train_loss": [], "val_loss": [], "train_metrics": [], "val_metrics": []},
    }

    # =========================
    # Stage 1
    # =========================
    stage1 = UNet(in_channels=1, out_channels=1, base_ch=cfg.STAGE1_BASE_CH).to(device)

    if cfg.USE_PRETRAINED_STAGE1 and os.path.exists(cfg.PRETRAINED_STAGE1_PATH):
        stage1.load_state_dict(torch.load(cfg.PRETRAINED_STAGE1_PATH, map_location=device))
        print(f"[Stage1] loaded: {cfg.PRETRAINED_STAGE1_PATH} | base_ch={cfg.STAGE1_BASE_CH}")
    else:
        print("[Stage1] train from scratch (per-image macro metrics + BCE+DiceLoss)")

        ds1 = VesselDataset(cfg.STAGE1_IMAGES, cfg.STAGE1_MASKS, transform=transform, stage=1)
        tr1, va1 = split_dataset(ds1, train_ratio=0.8, seed=cfg.SEED)
        dl_tr1 = DataLoader(tr1, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS,
                            pin_memory=True if device.type == 'cuda' else False)
        dl_va1 = DataLoader(va1, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS,
                            pin_memory=True if device.type == 'cuda' else False)

        opt1 = optim.Adam(stage1.parameters(), lr=cfg.LR_STAGE1)

        best_dice = -1.0
        bad = 0

        for epoch in range(cfg.EPOCHS_STAGE1):
            tr_loss, tr_m = run_one_epoch(stage1, dl_tr1, opt1, device, loss_fn, cfg.THRESHOLD,
                                          train=True, desc=f"Stage1 Train E{epoch + 1}")
            va_loss, va_m = run_one_epoch(stage1, dl_va1, opt1, device, loss_fn, cfg.THRESHOLD,
                                          train=False, desc=f"Stage1 Val   E{epoch + 1}")

            history["stage1"]["train_loss"].append(tr_loss)
            history["stage1"]["val_loss"].append(va_loss)
            history["stage1"]["train_metrics"].append(tr_m)
            history["stage1"]["val_metrics"].append(va_m)

            print(f"[Stage1] E{epoch + 1} | loss {tr_loss:.4f}/{va_loss:.4f} "
                  f"| dice {tr_m['dice']:.4f}/{va_m['dice']:.4f} (per-image macro)")

            if va_m["dice"] > best_dice + cfg.MIN_DELTA_DICE:
                best_dice = va_m["dice"]
                torch.save(stage1.state_dict(), cfg.BEST_STAGE1_PATH)
                print(f"[Stage1] ✅ save best: dice={best_dice:.4f} -> {cfg.BEST_STAGE1_PATH}")
                bad = 0
            else:
                bad += 1
                if bad >= cfg.PATIENCE_STAGE1:
                    print(f"[Stage1] early stop at epoch={epoch + 1} | best_dice={best_dice:.4f}")
                    break

        stage1.load_state_dict(torch.load(cfg.BEST_STAGE1_PATH, map_location=device))
        print(f"[Stage1] loaded best: {cfg.BEST_STAGE1_PATH}")

    # =========================
    # Stage 2
    # =========================
    print(
        f"[Stage2] train (per-image macro) | stage2_base_ch={cfg.STAGE2_BASE_CH} | freeze_stage1={cfg.FREEZE_STAGE1_IN_STAGE2}")
    ds2 = VesselDataset(cfg.STAGE2_IMAGES, cfg.STAGE2_MASKS, transform=transform, stage=2)
    tr2, va2 = split_dataset(ds2, train_ratio=0.8, seed=cfg.SEED)

    dl_tr2 = DataLoader(tr2, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS,
                        pin_memory=True if device.type == 'cuda' else False)
    dl_va2 = DataLoader(va2, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS,
                        pin_memory=True if device.type == 'cuda' else False)

    two_stage = TwoStageUNet(stage1_model=stage1, stage2_base_ch=cfg.STAGE2_BASE_CH,
                             freeze_stage1=cfg.FREEZE_STAGE1_IN_STAGE2, stage1_ch=cfg.STAGE1_BASE_CH).to(device)

    param_groups = []

    if not cfg.FREEZE_STAGE1_IN_STAGE2:
        param_groups.append({
            "params": two_stage.stage1.parameters(),
            "lr": cfg.LR_STAGE2_STAGE1_FINETUNE
        })

    param_groups.append({
        "params": two_stage.stage2.parameters(),
        "lr": cfg.LR_STAGE2
    })
    opt2 = optim.Adam(param_groups, weight_decay=1e-5)

    best_dice2 = -1.0
    bad2 = 0

    for epoch in range(cfg.EPOCHS_STAGE2):
        tr_loss, tr_m = run_one_epoch(two_stage, dl_tr2, opt2, device, loss_fn, cfg.THRESHOLD,
                                      train=True, desc=f"Stage2 Train E{epoch + 1}")
        va_loss, va_m = run_one_epoch(two_stage, dl_va2, opt2, device, loss_fn, cfg.THRESHOLD,
                                      train=False, desc=f"Stage2 Val   E{epoch + 1}")

        history["stage2"]["train_loss"].append(tr_loss)
        history["stage2"]["val_loss"].append(va_loss)
        history["stage2"]["train_metrics"].append(tr_m)
        history["stage2"]["val_metrics"].append(va_m)

        print(f"[Stage2] E{epoch + 1} | loss {tr_loss:.4f}/{va_loss:.4f} "
              f"| dice {tr_m['dice']:.4f}/{va_m['dice']:.4f} (per-image macro)")

        if va_m["dice"] > best_dice2 + cfg.MIN_DELTA_DICE:
            best_dice2 = va_m["dice"]
            torch.save(two_stage.state_dict(), cfg.BEST_STAGE2_PATH)
            print(f"[Stage2] ✅ save best: dice={best_dice2:.4f} -> {cfg.BEST_STAGE2_PATH}")
            bad2 = 0
        else:
            bad2 += 1
            if bad2 >= cfg.PATIENCE_STAGE2:
                print(f"[Stage2] early stop at epoch={epoch + 1} | best_dice={best_dice2:.4f}")
                break

    two_stage.load_state_dict(torch.load(cfg.BEST_STAGE2_PATH, map_location=device))
    print(f"[Stage2] loaded best: {cfg.BEST_STAGE2_PATH} | best_dice={best_dice2:.4f}")

    # =========================
    # Test（per-image macro）
    # =========================
    dsT = VesselDataset(cfg.TEST_IMAGES, cfg.TEST_MASKS, transform=transform, stage=2)
    dlT = DataLoader(dsT, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS,
                     pin_memory=True if device.type == 'cuda' else False)

    test_loss, test_m = run_one_epoch(two_stage, dlT, opt2, device, loss_fn, cfg.THRESHOLD,
                                      train=False, desc="TEST")

    print("\n========== TEST RESULTS (per-image macro) ==========")
    print(f"Samples   : {len(dsT)}")
    print(f"Loss      : {test_loss:.4f}")
    print(f"Dice      : {test_m['dice']:.4f}")
    print(f"IoU       : {test_m['iou']:.4f}")
    print(f"Acc       : {test_m['acc']:.4f}")
    print(f"Precision : {test_m['precision']:.4f}")
    print(f"Recall    : {test_m['recall']:.4f}")
    print(f"F1        : {test_m['f1']:.4f}")
    print(f"clDice    : {test_m['cldice']:.4f}")
    print(f"HD95      : {test_m['hd95']:.4f}")
    print("===============================================\n")

    # 保存结果
    out = {
        "timestamp": datetime.now().isoformat(),
        "cfg": {
            "stage1_loaded": bool(cfg.USE_PRETRAINED_STAGE1 and os.path.exists(cfg.PRETRAINED_STAGE1_PATH)),
            "stage1_ckpt": cfg.PRETRAINED_STAGE1_PATH,
            "stage1_base_ch": cfg.STAGE1_BASE_CH,
            "stage2_base_ch": cfg.STAGE2_BASE_CH,
            "freeze_stage1_in_stage2": cfg.FREEZE_STAGE1_IN_STAGE2,
            "threshold": cfg.THRESHOLD,
            "img_size": cfg.IMG_SIZE,
            "batch_size": cfg.BATCH_SIZE,
            "loss": {"bce_w": cfg.LOSS_BCE_W, "dice_w": cfg.LOSS_DICE_W},
            "lrs": {"stage1": cfg.LR_STAGE1, "stage2_stage1_finetune": cfg.LR_STAGE2_STAGE1_FINETUNE,
                    "stage2": cfg.LR_STAGE2}
        },
        "best_val_dice_stage2": best_dice2,
        "test": {"loss": test_loss, "metrics_macro": test_m},
        "history": history
    }

    save_json(out, os.path.join(cfg.RESULTS_DIR, "train_test_results.json"))
    print(f"[SAVE] {os.path.join(cfg.RESULTS_DIR, 'train_test_results.json')}")

    # 画曲线
    plot_curves(history, os.path.join(cfg.RESULTS_DIR, "curves.png"))
    print(f"[SAVE] {os.path.join(cfg.RESULTS_DIR, 'curves.png')}")
    print("全部完成 ✅")
