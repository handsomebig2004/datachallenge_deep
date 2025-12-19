# -*- coding: utf-8 -*-
"""
ResNet34-UNet（简化版，可直接跑）
- 数据都在同一个目录：Desktop/deep_datachallenge/images
- 训练标签：Y_train_T9NrBYo.csv（flatten + -1 padding）
- 训练井：Well 1–6
- 测试井：Well 7–11（同目录里筛选）
- 输出：submission.csv（每行一个 patch，flatten，pad 到 160*272 用 -1）

注意你只需要改：
1) DATA_ROOT 路径
2) EPOCHS/BATCH_SIZE 等超参数按你显存调整
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from torchvision.models import resnet34, ResNet34_Weights


# =========================
# 0. 超参数与路径
# =========================
DATA_ROOT = Path(r"C:\Users\lenovo\Desktop\deep_datachallenge") 
IMAGES_DIR = DATA_ROOT / "images"
Y_TRAIN_CSV = DATA_ROOT / "Y_train_T9NrBYo.csv"

TARGET_H = 160
TARGET_W = 272

NUM_CLASSES = 3          # 你确认 CSV 里只有 0/1/2
IGNORE_INDEX = -1        # CSV padding

BATCH_SIZE = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# 1. 工具函数
# =========================
def parse_well_id(name: str) -> int:
    """从 well_1_section_0_patch_0 提取 well id=1"""
    m = re.search(r"well_(\d+)_", name)
    return int(m.group(1)) if m else -1


def minmax_normalize(x: np.ndarray) -> np.ndarray:
    """min-max 归一化；NaN/inf 置 0"""
    x = x.astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x_min = float(x.min())
    x_max = float(x.max())
    if x_max - x_min < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min)


def pad_to_160x272(img: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """把 (160,160) 或 (160,272) pad 到 (160,272)"""
    h, w = img.shape
    assert h == TARGET_H, f"期望高度 {TARGET_H}，但拿到 {h}"
    if w == TARGET_W:
        return img
    if w < TARGET_W:
        out = np.full((TARGET_H, TARGET_W), fill_value, dtype=img.dtype)
        out[:, :w] = img
        return out
    # 若更宽，简单裁剪（一般不会发生）
    return img[:, :TARGET_W]


def decode_mask_from_csv_row(row_values: np.ndarray) -> np.ndarray:
    """
    从 CSV 一行恢复 mask：
    - row_values: flatten + -1 padding
    - 去掉 -1 后 reshape 成 (160, w)
    """
    valid = row_values[row_values != IGNORE_INDEX]
    assert len(valid) % TARGET_H == 0, f"mask 有效长度 {len(valid)} 不能被 160 整除"
    w = len(valid) // TARGET_H
    return valid.reshape(TARGET_H, w).astype(np.int64)


def pad_mask_to_160x272(mask: np.ndarray) -> np.ndarray:
    """把 (160,w) pad 到 (160,272)，pad 用 -1（ignore）"""
    h, w = mask.shape
    assert h == TARGET_H
    if w == TARGET_W:
        return mask
    out = np.full((TARGET_H, TARGET_W), IGNORE_INDEX, dtype=np.int64)
    out[:, :w] = mask
    return out


# =========================
# 2. Dataset（支持按 wells 过滤）
# =========================
class WellSegDataset(Dataset):
    def __init__(self, images_dir: Path, y_csv_path: Path = None, wells=None):
        """
        wells: 例如 {1,2,3,4,5,6} 或 {7,8,9,10,11}
        y_csv_path=None 表示无标签（测试）
        """
        self.images_dir = images_dir
        self.has_label = y_csv_path is not None

        all_paths = sorted(images_dir.glob("*.npy"))
        all_names = [p.stem for p in all_paths]

        if wells is not None:
            keep = []
            for p, n in zip(all_paths, all_names):
                w = parse_well_id(n)
                if w in wells:
                    keep.append((p, n))
            self.image_paths = [x[0] for x in keep]
            self.names = [x[1] for x in keep]
        else:
            self.image_paths = all_paths
            self.names = all_names

        if self.has_label:
            # CSV index 通常就是 patch 名（不含 .npy）
            self.y_df = pd.read_csv(y_csv_path, index_col=0)
        else:
            self.y_df = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        img_path = self.image_paths[idx]

        # 读图
        img = np.load(img_path)              # (160,160) or (160,272)
        raw_w = img.shape[1]                 # 记录原始宽度（推理时要裁回去）
        img = minmax_normalize(img)
        img = pad_to_160x272(img, fill_value=0.0)
        img_t = torch.from_numpy(img).unsqueeze(0).float()  # (1,160,272)

        if not self.has_label:
            return {"name": name, "image": img_t, "raw_w": raw_w}

        # 读 mask
        row = self.y_df.loc[name].values.astype(np.int64)
        mask = decode_mask_from_csv_row(row)     # (160,w)
        mask = pad_mask_to_160x272(mask)         # (160,272)
        mask_t = torch.from_numpy(mask).long()

        return {"name": name, "image": img_t, "mask": mask_t, "raw_w": raw_w}


# =========================
# 3. ResNet34-UNet（简化实现）
# =========================
class ConvRelu(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = ConvRelu(in_ch + skip_ch, out_ch)
        self.conv2 = ConvRelu(out_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResNet34UNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

        # 输入单通道：把第一层卷积改成 1 通道
        old_conv1 = backbone.conv1
        new_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv1.weight[:] = old_conv1.weight.mean(dim=1, keepdim=True)
        backbone.conv1 = new_conv1

        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # /2
        self.pool0 = backbone.maxpool                                           # /4
        self.enc1 = backbone.layer1                                             # /4
        self.enc2 = backbone.layer2                                             # /8
        self.enc3 = backbone.layer3                                             # /16
        self.enc4 = backbone.layer4                                             # /32

        self.center = nn.Sequential(ConvRelu(512, 512), ConvRelu(512, 512))
        self.up4 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 64, 64)

        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        e0 = self.enc0(x)               # 64, H/2, W/2
        e1 = self.enc1(self.pool0(e0))  # 64, H/4, W/4
        e2 = self.enc2(e1)              # 128, H/8, W/8
        e3 = self.enc3(e2)              # 256, H/16, W/16
        e4 = self.enc4(e3)              # 512, H/32, W/32

        c = self.center(e4)
        d4 = self.up4(c, e3)
        d3 = self.up3(d4, e2)
        d2 = self.up2(d3, e1)
        d1 = self.up1(d2, e0)

        out = self.head(d1)
        out = F.interpolate(out, size=(TARGET_H, TARGET_W), mode="bilinear", align_corners=False)
        return out


# =========================
# 4. 训练与验证（最简单）
# =========================
def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0

    for batch in loader:
        x = batch["image"].to(DEVICE)    # (B,1,160,272)
        y = batch["mask"].to(DEVICE)     # (B,160,272) 包含 -1

        logits = model(x)                # (B,C,160,272)
        loss = F.cross_entropy(logits, y, ignore_index=IGNORE_INDEX)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def valid_one_epoch(model, loader):
    model.eval()
    total_loss = 0.0

    for batch in loader:
        x = batch["image"].to(DEVICE)
        y = batch["mask"].to(DEVICE)

        logits = model(x)
        loss = F.cross_entropy(logits, y, ignore_index=IGNORE_INDEX)
        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


# =========================
# 5. 推理并生成提交 CSV
# =========================
@torch.no_grad()
def predict_and_make_submission(model, images_dir: Path, out_csv_path: Path, test_wells: set):
    """
    从 images_dir 中筛选 test_wells 预测并生成提交 CSV
    - 每行：一个 patch
    - 长度：160*272
    - 如果原始宽度 < 272，剩余用 -1 padding
    """
    model.eval()

    test_ds = WellSegDataset(images_dir, y_csv_path=None, wells=test_wells)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    preds_dict = {}

    for batch in test_loader:
        name = batch["name"][0]
        raw_w = int(batch["raw_w"][0])  # 原始宽度 160 或 272
        x = batch["image"].to(DEVICE)   # (1,1,160,272)

        logits = model(x)               # (1,C,160,272)
        pred_full = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int64)  # (160,272)

        # 裁回原始宽度
        pred = pred_full[:, :raw_w]     # (160,raw_w)

        # flatten + pad 到 160*272
        if raw_w < TARGET_W:
            padded = np.full((TARGET_H * TARGET_W,), IGNORE_INDEX, dtype=np.int64)
            padded[: TARGET_H * raw_w] = pred.flatten()
            preds_dict[name] = padded
        else:
            preds_dict[name] = pred.flatten()

    sub = pd.DataFrame(preds_dict, dtype="int64").T
    sub.to_csv(out_csv_path)
    print(f"[OK] submission 已保存: {out_csv_path}")


# =========================
# 6. 主函数：严格按井划分训练/验证，测试井预测提交
# =========================
def main():
    # ====== (A) 训练井与测试井定义 ======
    TRAIN_WELLS = {1, 2, 3, 4, 5, 6}
    TEST_WELLS = {7, 8, 9, 10, 11}

    # 验证集：从训练井里“按井留出”避免泄漏（例：留 well6）
    VAL_WELLS = {6}

    # ====== (B) 构建训练集（只读 well1-6） ======
    train_ds_all = WellSegDataset(IMAGES_DIR, Y_TRAIN_CSV, wells=TRAIN_WELLS)

    # 按井划分 train/val
    train_indices, val_indices = [], []
    for i, name in enumerate(train_ds_all.names):
        w = parse_well_id(name)
        if w in VAL_WELLS:
            val_indices.append(i)
        else:
            train_indices.append(i)

    train_ds = Subset(train_ds_all, train_indices)
    val_ds = Subset(train_ds_all, val_indices)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"训练样本数: {len(train_ds)} | 验证样本数: {len(val_ds)} | val_wells={VAL_WELLS}")

    # ====== (C) 模型与优化器 ======
    model = ResNet34UNet(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # ====== (D) 训练 ======
    best_val = 1e9
    best_path = DATA_ROOT / "best_resnet34_unet.pth"

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer)
        va_loss = valid_one_epoch(model, val_loader)

        print(f"Epoch {epoch:02d}/{EPOCHS} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), best_path)
            print(f"  -> 保存最优模型: {best_path}")

    # ====== (E) 生成提交（从同一个 images/ 里筛选 well7-11） ======
    out_csv = DATA_ROOT / "submission.csv"
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    predict_and_make_submission(model, IMAGES_DIR, out_csv, test_wells=TEST_WELLS)


if __name__ == "__main__":
    main()
