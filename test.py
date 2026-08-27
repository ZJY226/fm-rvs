#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from train import (
    UNet,
    TwoStageUNet,
    BCEDiceLoss,
    metrics_per_image_macro
)

# -------------------------
# CONFIG
# -------------------------

TEST_IMAGES = "data/test/images"
TEST_MASKS = "data/test/masks_fine"

# TEST_IMAGES = "data/test/images"
# TEST_MASKS  = "data/test/masks_fine"

# TEST_IMAGES = "Dataset/CHASEDB1/images"
# TEST_MASKS  = "Dataset/CHASEDB1/labels2st"

# TEST_IMAGES = "Dataset/HRF/images"
# TEST_MASKS  = "Dataset/HRF/manual1"

# TEST_IMAGES = "Dataset/LES/images"
# TEST_MASKS  = "Dataset/LES/arteries-and-veins"

# TEST_IMAGES = "Dataset/STARE/images"
# TEST_MASKS  = "Dataset/STARE/shows"

MODEL_PATH = "models/best_stage2_model.pth"

BATCH_SIZE = 1
NUM_WORKERS = 0
THRESHOLD = 0.5

SAVE_DIR = "test_predict_vis"
SAVE_BINARY = True
SAVE_OVERLAY = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor()
])

loss_fn = BCEDiceLoss()


# -------------------------
# DATASET
# -------------------------
class VesselDataset(Dataset):
    def __init__(self, images_path, masks_path, transform=None, stage=2, mask_type='binary'):
        """
        mask_type:
            'binary' -> 0/1 mask
            'artery_vein' -> 0/1/2 mask
        """
        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.stage = stage
        self.mask_type = mask_type

        if not os.path.exists(images_path):
            raise ValueError(f"图像路径不存在: {images_path}")
        if not os.path.exists(masks_path):
            raise ValueError(f"掩码路径不存在: {masks_path}")

        self.images = sorted([
            f for f in os.listdir(images_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ])
        self.masks = sorted([
            f for f in os.listdir(masks_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ])

        if len(self.images) == 0 or len(self.masks) == 0:
            raise ValueError("图像或掩码文件夹为空")

        self.matched_pairs = []
        used_masks = set()

        for img_file in self.images:
            img_name = os.path.splitext(img_file)[0]
            found_mask = None

            # 1. 优先同名匹配
            for mask_file in self.masks:
                if mask_file in used_masks:
                    continue
                mask_name = os.path.splitext(mask_file)[0]
                if mask_name == img_name:
                    found_mask = mask_file
                    break

            # 2. 同名找不到，尝试包含匹配
            if found_mask is None:
                for mask_file in self.masks:
                    if mask_file in used_masks:
                        continue
                    mask_name = os.path.splitext(mask_file)[0]
                    if img_name in mask_name:
                        found_mask = mask_file
                        break

            if found_mask:
                self.matched_pairs.append((img_file, found_mask))
                used_masks.add(found_mask)

        if len(self.matched_pairs) == 0:
            raise ValueError("无法匹配图像和掩码文件")

        if len(self.matched_pairs) != len(self.images):
            print(f"⚠️ 警告: 只匹配了 {len(self.matched_pairs)}/{len(self.images)} 个图像文件")

        print(f"Stage {stage} 数据集加载成功: {len(self.matched_pairs)} 对图像-掩码")

    def __len__(self):
        return len(self.matched_pairs)

    def __getitem__(self, idx):
        img_file, mask_file = self.matched_pairs[idx]

        try:
            img = cv2.imread(os.path.join(self.images_path, img_file), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(os.path.join(self.masks_path, mask_file), cv2.IMREAD_GRAYSCALE)

            if img is None:
                raise ValueError(f"无法读取图像: {img_file}")
            if mask is None:
                raise ValueError(f"无法读取掩码: {mask_file}")

            if self.mask_type == 'binary':
                mask = (mask > 0).astype(np.uint8) * 255
            elif self.mask_type == 'artery_vein':
                mask = mask.astype(np.uint8)
            else:
                raise ValueError(f"未知 mask_type={self.mask_type}")

            img_pil = Image.fromarray(img)
            mask_pil = Image.fromarray(mask)

            if self.transform:
                img_tensor = self.transform(img_pil)
                mask_tensor = self.transform(mask_pil)
            else:
                img_tensor = torch.from_numpy(img).unsqueeze(0).float() / 255.0
                mask_tensor = torch.from_numpy(mask).unsqueeze(0).float() / 255.0

            return img_tensor, mask_tensor, img_file

        except Exception as e:
            print(f"处理图像 {img_file} 时出错: {e}")
            return torch.zeros(1, 512, 512), torch.zeros(1, 512, 512), "error.png"


# -------------------------
# UTILS
# -------------------------
def tensor_to_uint8_gray(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255).astype(np.uint8)
    return x


def make_red_overlay(gray_img_u8, pred_bin_u8, alpha=0.45):
    """
    gray_img_u8: uint8, [H, W]
    pred_bin_u8: uint8, [H, W], 前景255
    """
    base = cv2.cvtColor(gray_img_u8, cv2.COLOR_GRAY2BGR)
    overlay = base.copy()

    red_mask = pred_bin_u8 > 0
    overlay[red_mask] = (0, 0, 255)

    fused = cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0)
    return fused


def save_prediction_results(imgs, msks, preds, names, save_dir, threshold=0.5,
                            save_binary=True, save_overlay=True):
    os.makedirs(save_dir, exist_ok=True)

    preds_bin = (preds > threshold).float()

    for i in range(imgs.size(0)):
        name = names[i]
        stem = os.path.splitext(os.path.basename(name))[0]

        img_u8 = tensor_to_uint8_gray(imgs[i, 0])
        gt_u8 = tensor_to_uint8_gray(msks[i, 0])
        pred_prob_u8 = tensor_to_uint8_gray(preds[i, 0])
        pred_bin_u8 = tensor_to_uint8_gray(preds_bin[i, 0])

        # 保存概率图
        cv2.imwrite(os.path.join(save_dir, f"{stem}_pred_prob.png"), pred_prob_u8)

        # 保存二值图
        if save_binary:
            cv2.imwrite(os.path.join(save_dir, f"{stem}_pred_bin.png"), pred_bin_u8)

        # 保存GT
        cv2.imwrite(os.path.join(save_dir, f"{stem}_gt.png"), gt_u8)

        # 保存红色叠加图
        overlay = make_red_overlay(img_u8, pred_bin_u8, alpha=0.45)
        if save_overlay:
            cv2.imwrite(os.path.join(save_dir, f"{stem}_overlay.png"), overlay)

        # 保存四联图：原图 | GT | 概率图 | 叠加图
        img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
        gt_bgr = cv2.cvtColor(gt_u8, cv2.COLOR_GRAY2BGR)
        pred_prob_bgr = cv2.cvtColor(pred_prob_u8, cv2.COLOR_GRAY2BGR)

        vis = np.concatenate([img_bgr, gt_bgr, pred_prob_bgr, overlay], axis=1)
        cv2.imwrite(os.path.join(save_dir, f"{stem}_vis.png"), vis)


# -------------------------
# MODEL
# -------------------------
stage1 = UNet(in_channels=1, out_channels=1, base_ch=64)

model = TwoStageUNet(
    stage1_model=stage1,
    stage1_ch=64,
    stage2_base_ch=32,
    freeze_stage1=False
)

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt)

model = model.to(DEVICE)
model.eval()


# -------------------------
# DATA
# -------------------------
dsT = VesselDataset(TEST_IMAGES, TEST_MASKS, transform=transform)

dlT = DataLoader(
    dsT,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda")
)


# -------------------------
# TEST + SAVE PREDICTIONS
# -------------------------
model.eval()

sum_m = {
    "acc": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "f1": 0.0,
    "dice": 0.0,
    "iou": 0.0,
    "cldice": 0.0,
    "hd95": 0.0
}
test_loss = 0.0
n_batches = 0

os.makedirs(SAVE_DIR, exist_ok=True)

with torch.no_grad():
    for imgs, msks, names in tqdm(dlT, desc="TEST"):
        imgs = imgs.to(DEVICE, non_blocking=True)
        msks = msks.to(DEVICE, non_blocking=True)

        out = model(imgs)

        if isinstance(out, tuple):
            p1, p2 = out
        else:
            p2 = out

        loss = loss_fn(p2, msks)
        test_loss += loss.item()

        m = metrics_per_image_macro(
            p2,
            msks,
            threshold=THRESHOLD,
            compute_cldice=True,
            compute_hd95=True
        )

        for k in sum_m:
            sum_m[k] += m[k]

        save_prediction_results(
            imgs=imgs,
            msks=msks,
            preds=p2,
            names=names,
            save_dir=SAVE_DIR,
            threshold=THRESHOLD,
            save_binary=SAVE_BINARY,
            save_overlay=SAVE_OVERLAY
        )

        n_batches += 1

test_loss /= max(1, n_batches)
test_m = {k: sum_m[k] / max(1, n_batches) for k in sum_m}

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
print("===============================================")
print(f"[SAVE] 预测结果已保存到: {SAVE_DIR}")