# -*- coding: utf-8 -*-
"""
Mask2Former + Semi-Supervised Pseudo Label (Full runnable)

Data:
- Labeled train images:   X_train_uDRk9z9/images (well1-6)
- Labeled train labels:   Y_train_T9NrBYo.csv (flatten + -1 padding)
- Unlabeled images:       X_unlabeled_mtkxUlo/images (well12-14)
- Test images:            X_test_xNbnvIa/images (well7-11)

Split:
- Train labeled: well1-5
- Val labeled:   well6
- Unlabeled:     well12-14 (no labels)

Output:
- submission.csv, each row = one patch
- flattened mask, padded to 160*272 with -1

Install:
    pip install transformers accelerate
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from transformers import (
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation,
)

# =========================
# 0) Paths & Hyperparameters
# =========================
DATA_ROOT = Path(r"C:\Users\lenovo\Desktop\deep_datachallenge")

TRAIN_IMAGES_DIR = DATA_ROOT / "X_train_uDRk9z9" / "images"
TEST_IMAGES_DIR  = DATA_ROOT / "X_test_xNbnvIa" / "images"
UNLABELED_DIR     = DATA_ROOT / "X_unlabeled_mtkxUlo" / "images"
Y_TRAIN_CSV       = DATA_ROOT / "Y_train_T9NrBYo.csv"

# submission size
TARGET_H = 160
TARGET_W = 272

# model size
MODEL_H = 224
MODEL_W = 224

NUM_CLASSES = 3
IGNORE_INDEX = -1

BATCH_SIZE_L = 2         # labeled batch
BATCH_SIZE_U = 2         # unlabeled batch
LR = 5e-5
WEIGHT_DECAY = 1e-4
EPOCHS = 10

# semi-supervised hyperparams
PSEUDO_TH = 0.85         # 伪标签置信度阈值(越高越保守)
LAMBDA_U = 0.5           # 无标签loss权重(0.2~1.0可调)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PRETRAINED = "facebook/mask2former-swin-tiny-ade-semantic"


# =========================
# 1) Utils
# =========================
def parse_well_id(name: str) -> int:
    """well_12_section_0_patch_0 -> 12"""
    m = re.search(r"well_(\d+)_", name)
    return int(m.group(1)) if m else -1


def minmax_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def pad_to_160x272(img: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    h, w = img.shape
    assert h == TARGET_H, f"Expected height {TARGET_H}, got {h}"
    if w == TARGET_W:
        return img
    if w < TARGET_W:
        out = np.full((TARGET_H, TARGET_W), fill_value, dtype=img.dtype)
        out[:, :w] = img
        return out
    return img[:, :TARGET_W]


def decode_mask_from_csv_row(row_values: np.ndarray) -> np.ndarray:
    valid = row_values[row_values != IGNORE_INDEX]
    assert len(valid) % TARGET_H == 0, f"Valid mask length {len(valid)} not divisible by 160"
    w = len(valid) // TARGET_H
    return valid.reshape(TARGET_H, w).astype(np.int64)


def pad_mask_to_160x272(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    assert h == TARGET_H
    if w == TARGET_W:
        return mask
    out = np.full((TARGET_H, TARGET_W), IGNORE_INDEX, dtype=np.int64)
    out[:, :w] = mask
    return out


def resize_image_torch(img_1hw: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """(1,H,W)->(1,h,w) bilinear"""
    x = img_1hw.unsqueeze(0)  # (1,1,H,W)
    x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    return x.squeeze(0)


def resize_mask_torch(mask_hw: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """(H,W)->(h,w) nearest"""
    y = mask_hw.unsqueeze(0).unsqueeze(0).float()
    y = F.interpolate(y, size=(h, w), mode="nearest")
    return y.squeeze(0).squeeze(0).long()


def semantic_to_mask2former_targets(
    semantic_mask: torch.Tensor,
    num_classes: int,
    ignore_index: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    semantic_mask: (H,W) with ignore_index
    return:
      class_labels: (N,)
      mask_labels:  (N,H,W) float(0/1)
    """
    valid = semantic_mask != ignore_index
    if valid.sum() == 0:
        class_labels = torch.tensor([0], dtype=torch.long)
        mask_labels = torch.zeros((1, semantic_mask.shape[0], semantic_mask.shape[1]), dtype=torch.float32)
        return class_labels, mask_labels

    present = torch.unique(semantic_mask[valid]).tolist()
    present = [int(c) for c in present if 0 <= int(c) < num_classes]
    if len(present) == 0:
        class_labels = torch.tensor([0], dtype=torch.long)
        mask_labels = torch.zeros((1, semantic_mask.shape[0], semantic_mask.shape[1]), dtype=torch.float32)
        return class_labels, mask_labels

    masks, classes = [], []
    for c in present:
        m = (semantic_mask == c) & valid
        if m.sum() == 0:
            continue
        masks.append(m.float())
        classes.append(c)

    if len(classes) == 0:
        class_labels = torch.tensor([0], dtype=torch.long)
        mask_labels = torch.zeros((1, semantic_mask.shape[0], semantic_mask.shape[1]), dtype=torch.float32)
        return class_labels, mask_labels

    class_labels = torch.tensor(classes, dtype=torch.long)
    mask_labels = torch.stack(masks, dim=0).float()
    return class_labels, mask_labels


# =========================
# 2) 简单增强（无标签用）
# =========================
def aug_weak(x: torch.Tensor) -> torch.Tensor:
    """弱增强：随机左右翻转 + 轻噪声"""
    # x: (1,224,224)
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[2])
    noise = 0.02 * torch.randn_like(x)
    return torch.clamp(x + noise, 0.0, 1.0)


def aug_strong(x: torch.Tensor) -> torch.Tensor:
    """强增强：随机翻转 + 更强噪声 + 亮度对比度扰动"""
    if torch.rand(1).item() < 0.5:
        x = torch.flip(x, dims=[2])
    # brightness/contrast
    contrast = 0.8 + 0.4 * torch.rand(1).item()   # [0.8,1.2]
    brightness = -0.1 + 0.2 * torch.rand(1).item() # [-0.1,0.1]
    x = x * contrast + brightness
    # noise
    noise = 0.05 * torch.randn_like(x)
    x = x + noise
    return torch.clamp(x, 0.0, 1.0)


# =========================
# 3) Dataset
# =========================
class LabeledDataset(Dataset):
    def __init__(self, images_dir: Path, y_csv_path: Path):
        self.image_paths = sorted(images_dir.glob("*.npy"))
        self.names = [p.stem for p in self.image_paths]
        self.y_df = pd.read_csv(y_csv_path, index_col=0)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        img = np.load(self.image_paths[idx])
        raw_w = int(img.shape[1])

        img = minmax_normalize(img)
        img = pad_to_160x272(img, fill_value=0.0)
        img_t = torch.from_numpy(img).unsqueeze(0).float()         # (1,160,272)
        img_t = resize_image_torch(img_t, MODEL_H, MODEL_W)        # (1,224,224)

        row = self.y_df.loc[name].values.astype(np.int64)
        mask = decode_mask_from_csv_row(row)                       # (160,w)
        mask = pad_mask_to_160x272(mask)                           # (160,272)
        mask_t = torch.from_numpy(mask).long()                     # (160,272)
        mask_t = resize_mask_torch(mask_t, MODEL_H, MODEL_W)        # (224,224)

        return {"name": name, "image": img_t, "mask": mask_t, "raw_w": raw_w}


class UnlabeledDataset(Dataset):
    def __init__(self, images_dir: Path):
        self.image_paths = sorted(images_dir.glob("*.npy"))
        self.names = [p.stem for p in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        img = np.load(self.image_paths[idx])
        raw_w = int(img.shape[1])

        img = minmax_normalize(img)
        img = pad_to_160x272(img, fill_value=0.0)
        img_t = torch.from_numpy(img).unsqueeze(0).float()         # (1,160,272)
        img_t = resize_image_torch(img_t, MODEL_H, MODEL_W)         # (1,224,224)

        # 返回 base 图像（增强在 collate 做）
        return {"name": name, "image": img_t, "raw_w": raw_w}


# =========================
# 4) Collate
# =========================
def collate_labeled(batch: List[Dict]) -> Dict:
    names = [b["name"] for b in batch]
    raw_ws = torch.tensor([b["raw_w"] for b in batch], dtype=torch.long)

    imgs_1 = torch.stack([b["image"] for b in batch], dim=0)       # (B,1,224,224)
    pixel_values = imgs_1.repeat(1, 3, 1, 1)                       # (B,3,224,224)
    pixel_mask = torch.ones((pixel_values.shape[0], MODEL_H, MODEL_W), dtype=torch.long)

    class_labels_list, mask_labels_list = [], []
    for b in batch:
        y = b["mask"]  # (224,224)
        cls, msk = semantic_to_mask2former_targets(y, NUM_CLASSES, IGNORE_INDEX)
        class_labels_list.append(cls)
        mask_labels_list.append(msk)

    return {
        "names": names,
        "raw_ws": raw_ws,
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "class_labels": class_labels_list,
        "mask_labels": mask_labels_list,
    }


def collate_unlabeled(batch: List[Dict]) -> Dict:
    names = [b["name"] for b in batch]

    imgs = [b["image"] for b in batch]  # list of (1,224,224)

    # weak / strong augmentation
    imgs_w = torch.stack([aug_weak(x.clone()) for x in imgs], dim=0)    # (B,1,224,224)
    imgs_s = torch.stack([aug_strong(x.clone()) for x in imgs], dim=0)  # (B,1,224,224)

    pixel_values_w = imgs_w.repeat(1, 3, 1, 1)  # (B,3,224,224)
    pixel_values_s = imgs_s.repeat(1, 3, 1, 1)

    pixel_mask = torch.ones((pixel_values_w.shape[0], MODEL_H, MODEL_W), dtype=torch.long)

    return {
        "names": names,
        "pixel_values_w": pixel_values_w,
        "pixel_values_s": pixel_values_s,
        "pixel_mask": pixel_mask,
    }


# =========================
# 5) Model builder
# =========================
def build_model(num_classes: int):
    id2label = {0: "class0", 1: "class1", 2: "class2"}
    label2id = {v: k for k, v in id2label.items()}

    processor = AutoImageProcessor.from_pretrained(PRETRAINED)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        PRETRAINED,
        ignore_mismatched_sizes=True,
        id2label=id2label,
        label2id=label2id,
        num_labels=num_classes,
    )
    return model, processor


# =========================
# 6) Pseudo label from Mask2Former outputs
# =========================
@torch.no_grad()
def pseudo_from_outputs(outputs, num_classes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从 Mask2Former 输出构造像素级类别概率：
    class_probs (softmax) * mask_probs (sigmoid) -> per-pixel scores

    outputs.class_queries_logits: (B, Q, C+1)  (最后一个通常是 no-object)
    outputs.masks_queries_logits: (B, Q, H, W)

    return:
      pseudo: (B,H,W) long  (0..C-1)
      conf:   (B,H,W) float (max score)
    """
    class_logits = outputs.class_queries_logits  # (B,Q,C+1)
    mask_logits = outputs.masks_queries_logits   # (B,Q,H,W)

    class_prob = class_logits.softmax(dim=-1)[..., :num_classes]   # (B,Q,C)
    mask_prob = mask_logits.sigmoid()                              # (B,Q,H,W)

    # (B,C,H,W)  einsum: sum_q class_prob[b,q,c] * mask_prob[b,q,h,w]
    score = torch.einsum("bqc,bqhw->bchw", class_prob, mask_prob)
    conf, pseudo = torch.max(score, dim=1)  # (B,H,W)
    return pseudo.long(), conf.float()


# =========================
# 7) Train / Validate (Semi-Supervised)
# =========================
def train_one_epoch_semi(model, labeled_loader, unlabeled_loader, optimizer):
    model.train()

    total_l, total_u = 0.0, 0.0
    n_l, n_u = 0, 0

    unlabeled_iter = iter(unlabeled_loader)

    for batch_l in labeled_loader:
        # ---- labeled step ----
        pixel_values = batch_l["pixel_values"].to(DEVICE)
        pixel_mask = batch_l["pixel_mask"].to(DEVICE)
        class_labels = [x.to(DEVICE) for x in batch_l["class_labels"]]
        mask_labels = [x.to(DEVICE) for x in batch_l["mask_labels"]]

        out_l = model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            class_labels=class_labels,
            mask_labels=mask_labels,
        )
        loss_l = out_l.loss

        # ---- unlabeled step (pseudo-label) ----
        try:
            batch_u = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            batch_u = next(unlabeled_iter)

        pv_w = batch_u["pixel_values_w"].to(DEVICE)  # weak
        pv_s = batch_u["pixel_values_s"].to(DEVICE)  # strong
        pm_u = batch_u["pixel_mask"].to(DEVICE)

        # teacher prediction on weak
        model.eval()
        out_u_teacher = model(pixel_values=pv_w, pixel_mask=pm_u)
        pseudo, conf = pseudo_from_outputs(out_u_teacher, NUM_CLASSES)  # (B,224,224)

        # 置信度过滤：低于阈值的像素设为 IGNORE
        pseudo = pseudo.clone()
        pseudo[conf < PSEUDO_TH] = IGNORE_INDEX

        # 将 pseudo semantic mask -> mask2former targets(list)
        class_labels_u, mask_labels_u = [], []
        for i in range(pseudo.shape[0]):
            cls_i, msk_i = semantic_to_mask2former_targets(pseudo[i], NUM_CLASSES, IGNORE_INDEX)
            class_labels_u.append(cls_i.to(DEVICE))
            mask_labels_u.append(msk_i.to(DEVICE))

        model.train()
        out_u_student = model(
            pixel_values=pv_s,
            pixel_mask=pm_u,
            class_labels=class_labels_u,
            mask_labels=mask_labels_u,
        )
        loss_u = out_u_student.loss

        # ---- total loss ----
        loss = loss_l + LAMBDA_U * loss_u

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_l += float(loss_l.item()) * pixel_values.size(0)
        total_u += float(loss_u.item()) * pv_s.size(0)
        n_l += pixel_values.size(0)
        n_u += pv_s.size(0)

    return total_l / max(1, n_l), total_u / max(1, n_u)


@torch.no_grad()
def valid_one_epoch(model, loader):
    model.eval()
    total = 0.0
    n = 0

    for batch in loader:
        pixel_values = batch["pixel_values"].to(DEVICE)
        pixel_mask = batch["pixel_mask"].to(DEVICE)
        class_labels = [x.to(DEVICE) for x in batch["class_labels"]]
        mask_labels = [x.to(DEVICE) for x in batch["mask_labels"]]

        out = model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            class_labels=class_labels,
            mask_labels=mask_labels,
        )
        total += float(out.loss.item()) * pixel_values.size(0)
        n += pixel_values.size(0)

    return total / max(1, n)


# =========================
# 8) Inference & submission
# =========================
@torch.no_grad()
def predict_and_make_submission(model, processor, test_images_dir: Path, out_csv_path: Path):
    model.eval()

    # 这里复用 UnlabeledDataset 结构（只有图像，无标签）
    test_ds = UnlabeledDataset(test_images_dir)

    def collate_test(batch: List[Dict]) -> Dict:
        names = [b["name"] for b in batch]
        raw_ws = torch.tensor([b["raw_w"] for b in batch], dtype=torch.long)
        imgs_1 = torch.stack([b["image"] for b in batch], dim=0)    # (B,1,224,224)
        pixel_values = imgs_1.repeat(1, 3, 1, 1)
        pixel_mask = torch.ones((pixel_values.shape[0], MODEL_H, MODEL_W), dtype=torch.long)
        return {"names": names, "raw_ws": raw_ws, "pixel_values": pixel_values, "pixel_mask": pixel_mask}

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_test)

    preds_dict = {}

    for batch in test_loader:
        name = batch["names"][0]
        raw_w = int(batch["raw_ws"][0].item())

        pixel_values = batch["pixel_values"].to(DEVICE)
        pixel_mask = batch["pixel_mask"].to(DEVICE)

        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        # 使用 processor 的语义后处理（稳定）
        seg_list = processor.post_process_semantic_segmentation(outputs, target_sizes=[(MODEL_H, MODEL_W)])
        seg_224 = seg_list[0].to(torch.int64)  # (224,224)

        seg_224 = seg_224.unsqueeze(0).unsqueeze(0).float()
        seg_160_272 = F.interpolate(seg_224, size=(TARGET_H, TARGET_W), mode="nearest").squeeze(0).squeeze(0)
        seg_160_272 = seg_160_272.cpu().numpy().astype(np.int64)

        pred = seg_160_272[:, :raw_w]

        padded = np.full((TARGET_H * TARGET_W,), IGNORE_INDEX, dtype=np.int64)
        padded[: TARGET_H * raw_w] = pred.flatten()
        preds_dict[name] = padded

    sub = pd.DataFrame(preds_dict, dtype="int64").T
    sub.to_csv(out_csv_path)
    print(f"[OK] submission saved to: {out_csv_path}")


# =========================
# 9) Main
# =========================
def main():
    print(f"DEVICE: {DEVICE}")
    print(f"Labeled train dir: {TRAIN_IMAGES_DIR}")
    print(f"Unlabeled dir:     {UNLABELED_DIR}")
    print(f"Test dir:          {TEST_IMAGES_DIR}")
    print(f"Pretrained:        {PRETRAINED}")
    print(f"Pseudo TH={PSEUDO_TH}, lambda_u={LAMBDA_U}")

    # ---- labeled dataset (well1-6) ----
    ds_all = LabeledDataset(TRAIN_IMAGES_DIR, Y_TRAIN_CSV)

    # split by well (val=6)
    train_idx, val_idx = [], []
    for i, name in enumerate(ds_all.names):
        w = parse_well_id(name)
        if w == 6:
            val_idx.append(i)
        else:
            train_idx.append(i)

    train_ds = Subset(ds_all, train_idx)  # well1-5
    val_ds   = Subset(ds_all, val_idx)    # well6

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE_L,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_labeled,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE_L,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_labeled,
    )

    # ---- unlabeled dataset (well12-14) ----
    unlab_ds = UnlabeledDataset(UNLABELED_DIR)
    unlab_loader = DataLoader(
        unlab_ds,
        batch_size=BATCH_SIZE_U,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_unlabeled,
    )

    print(f"Labeled train: {len(train_ds)} | Val: {len(val_ds)} | Unlabeled: {len(unlab_ds)}")

    model, processor = build_model(NUM_CLASSES)
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = 1e9
    best_path = DATA_ROOT / "best_mask2former_semi.pth"

    for epoch in range(1, EPOCHS + 1):
        tr_l, tr_u = train_one_epoch_semi(model, train_loader, unlab_loader, optimizer)
        va = valid_one_epoch(model, val_loader)

        print(f"Epoch {epoch:02d}/{EPOCHS} | train_l={tr_l:.4f} | train_u={tr_u:.4f} | val={va:.4f}")

        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), best_path)
            print(f"  -> Best saved: {best_path}")

    # ---- inference ----
    out_csv = DATA_ROOT / "submission.csv"
    model.load_state_dict(torch.load(best_path, map_location=DEVICE, weights_only=True))
    predict_and_make_submission(model, processor, TEST_IMAGES_DIR, out_csv)


if __name__ == "__main__":
    main()
