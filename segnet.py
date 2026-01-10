import re
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from functools import partial

# -----------------------
# split utils (by well)
# -----------------------
_WELL_RE = re.compile(r"^well_(\d+)_section_(\d+)_patch_(\d+)$")

def parse_patch_id(patch_id: str) -> tuple[int, int, int]:
    m = _WELL_RE.match(patch_id)
    if not m:
        raise ValueError(f"Bad patch id format: {patch_id}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))

def split_ids_by_well(all_ids, val_wells: list[int]):
    val_wells = set(val_wells)
    tr_ids, va_ids = [], []
    for pid in all_ids:
        w, _, _ = parse_patch_id(str(pid))
        (va_ids if w in val_wells else tr_ids).append(str(pid))
    return tr_ids, va_ids


# -----------------------
# small utils
# -----------------------
def patch_minmax(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32)
    x = np.nan_to_num(x, nan=0.0)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)

def infer_num_classes(y_df: pd.DataFrame) -> int:
    v = y_df.values.reshape(-1)
    v = v[v != -1]
    v = v[~pd.isna(v)]
    return int(v.max()) + 1

def resize_img(x: torch.Tensor, size_hw):
    return F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)

def resize_mask(y: torch.Tensor, size_hw):
    # y: [B,H,W] int64
    y = F.interpolate(y.unsqueeze(1).float(), size=size_hw, mode="nearest")
    return y.squeeze(1).long()


# -----------------------
# dataset
# -----------------------
class WellDataset(Dataset):
    def __init__(self, x_dir: Path, y_df: pd.DataFrame | None = None, include_ids=None):
        self.x_dir = Path(x_dir)
        self.y_df = y_df
        self.is_train = y_df is not None

        files = sorted(self.x_dir.rglob("*.npy"))

        if self.is_train:
            idx = set(self.y_df.index.astype(str))
            files = [p for p in files if p.stem in idx]

        if include_ids is not None:
            keep = set(map(str, include_ids))
            files = [p for p in files if p.stem in keep]

        self.files = files

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        p = self.files[i]
        stem = p.stem
        img = np.load(p)
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[..., 0]
        img = patch_minmax(img)                     # NaN->0 + minmax
        H, W = img.shape
        item = {
            "id": stem,
            "orig_hw": (H, W),
            "img": torch.from_numpy(img).unsqueeze(0),  # [1,H,W]
        }
        if self.is_train:
            row = self.y_df.loc[stem].to_numpy()
            row = row[row != -1]                   # remove padding -1 :contentReference[oaicite:1]{index=1}
            mask = row.reshape(160, -1).astype(np.int64)
            item["mask"] = torch.from_numpy(mask)  # [160,W]
        return item

def collate(batch, size_hw=(160, 272), is_train=True):
    Ht, Wt = size_hw  # (160,272)

    # 把所有图像pad到272
    imgs = []
    for b in batch:
        x = b["img"]
        H, W = x.shape[-2], x.shape[-1]
        # 高度理论上都是160；这里做个保险
        if H < Ht:
            x = F.pad(x, (0, 0, 0, Ht - H), value=0.0)
        elif H > Ht:
            x = x[..., :Ht, :]

        if W < Wt:
            x = F.pad(x, (0, Wt - W, 0, 0), value=0.0)
        elif W > Wt:
            x = x[..., :, :Wt]

        imgs.append(x)
    imgs = torch.stack(imgs, 0)          

    out = {"id":[b["id"] for b in batch], "orig_hw":[b["orig_hw"] for b in batch], "img":imgs}

    if is_train:
        masks = []
        for b in batch:
            y = b["mask"]  # [160,W]
            H, W = y.shape[-2], y.shape[-1]

            if H < Ht:
                y = F.pad(y, (0, 0, 0, Ht - H), value=-1)
            elif H > Ht:
                y = y[:Ht, :]
            if W < Wt:
                y = F.pad(y, (0, Wt - W, 0, 0), value=-1) # pad -1 for ignore_index
            elif W > Wt:
                y = y[:, :Wt]
            masks.append(y)
        masks = torch.stack(masks, 0)         # [B,160,Wmax]
        out["mask"] = masks

    return out

# -----------------------
# SegNet (VGG-like)
# -----------------------
def vgg_block(in_ch, out_ch, n_conv):
    layers = []
    ch = in_ch
    for _ in range(n_conv):
        layers += [nn.Conv2d(ch, out_ch, 3, padding=1),
                   nn.BatchNorm2d(out_ch),
                   nn.ReLU(inplace=True)]
        ch = out_ch
    return nn.Sequential(*layers)

class SegNetVGG(nn.Module):
    def __init__(self, in_channels=1, num_classes=3):
        super().__init__()
        self.enc1 = vgg_block(in_channels, 64, 2)
        self.enc2 = vgg_block(64, 128, 2)
        self.enc3 = vgg_block(128, 256, 3)
        self.enc4 = vgg_block(256, 512, 3)
        self.enc5 = vgg_block(512, 512, 3)
        self.pool = nn.MaxPool2d(2,2, return_indices=True)

        self.unpool = nn.MaxUnpool2d(2,2)
        self.dec5 = vgg_block(512, 512, 3)
        self.dec4 = vgg_block(512, 256, 3)
        self.dec3 = vgg_block(256, 128, 3)
        self.dec2 = vgg_block(128, 64, 2)
        self.dec1 = vgg_block(64, 64, 2)
        self.head = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        sizes, idxs = [], []

        x = self.enc1(x); sizes.append(x.size()); x, i = self.pool(x); idxs.append(i)
        x = self.enc2(x); sizes.append(x.size()); x, i = self.pool(x); idxs.append(i)
        x = self.enc3(x); sizes.append(x.size()); x, i = self.pool(x); idxs.append(i)
        x = self.enc4(x); sizes.append(x.size()); x, i = self.pool(x); idxs.append(i)
        x = self.enc5(x); sizes.append(x.size()); x, i = self.pool(x); idxs.append(i)

        x = self.unpool(x, idxs.pop(), output_size=sizes.pop()); x = self.dec5(x)
        x = self.unpool(x, idxs.pop(), output_size=sizes.pop()); x = self.dec4(x)
        x = self.unpool(x, idxs.pop(), output_size=sizes.pop()); x = self.dec3(x)
        x = self.unpool(x, idxs.pop(), output_size=sizes.pop()); x = self.dec2(x)
        x = self.unpool(x, idxs.pop(), output_size=sizes.pop()); x = self.dec1(x)

        return self.head(x)

# iou  验证
@torch.no_grad()
def mean_iou_batch(logits, target, num_classes: int, ignore_index: int = -1):
    # logits: [B,C,H,W], target: [B,H,W]
    pred = torch.argmax(logits, dim=1)

    ious = []
    for b in range(pred.shape[0]):
        p = pred[b].reshape(-1)
        t = target[b].reshape(-1)
        valid = (t != ignore_index)
        p = p[valid]
        t = t[valid]

        per_class = []
        for c in range(1, num_classes):  # 跳过背景0（按需改）
            inter = torch.sum((p == c) & (t == c)).item()
            union = torch.sum((p == c) | (t == c)).item()
            if union > 0:
                per_class.append(inter / union)

        # 如果这一张图目标类都不存在，按官方通常视为 IoU=1（X,Y 都空时得分为1）:contentReference[oaicite:9]{index=9}
        ious.append(1.0 if len(per_class) == 0 else sum(per_class) / len(per_class))

    return float(sum(ious) / max(1, len(ious)))

# -----------------------
# train + predict + submit
# -----------------------
@torch.no_grad()
def make_submission(model, loader, device, out_csv: Path, size_labels=272):
    if len(loader.dataset) == 0:
        raise RuntimeError("Test dataset is empty. Check X_TEST_DIR path or use rglob for nested files.")
    
    model.eval()
    rows = {}

    for batch in loader:
        x = batch["img"].to(device, non_blocking=True)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).cpu()   # [B,160,160]

        for k, pid in enumerate(batch["id"]):
            H0, W0 = batch["orig_hw"][k]

            # resize back to original (160x160 or 160x272)
            mk = resize_mask(pred[k].unsqueeze(0), (H0, W0)).squeeze(0).numpy()

            # csv must be 160*272 with -1 padding for shorter widths :contentReference[oaicite:3]{index=3}
            if W0 != size_labels:
                aux = (-1 + np.zeros(160 * size_labels, dtype=np.int64))
                aux[: 160 * W0] = mk.reshape(-1)
                rows[pid] = aux
            else:
                rows[pid] = mk.reshape(-1).astype(np.int64)

    pd.DataFrame(rows, dtype="int").T.to_csv(out_csv)
    print("saved:", out_csv)

def main():
    # ---- EDIT PATHS HERE ----
    ROOT = Path("data")
    X_TRAIN_DIR = ROOT / "X_train_uDRk9z9"
    X_TEST_DIR  = ROOT / "X_test_xNbnvIa"
    Y_TRAIN_CSV = ROOT / "Y_train_T9NrBYo.csv"

    train_size = (160,272)
    batch_size = 16
    epochs = 20
    lr = 1e-4
    collate_train = partial(collate, size_hw=train_size, is_train=True)
    collate_test  = partial(collate, size_hw=train_size, is_train=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # speed up on fixed input size

    y_df = pd.read_csv(Y_TRAIN_CSV, index_col=0)
    num_classes = infer_num_classes(y_df)

    # ---- split by well (choose your val wells) ----
    val_wells = [3, 5]  # 你也可以试 [4] 或者 leave-one-well-out
    tr_ids, va_ids = split_ids_by_well(y_df.index.astype(str), val_wells)

    ds_tr = WellDataset(X_TRAIN_DIR, y_df=y_df, include_ids=tr_ids)
    ds_va = WellDataset(X_TRAIN_DIR, y_df=y_df, include_ids=va_ids)
    ds_te = WellDataset(X_TEST_DIR, y_df=None)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                    num_workers=2, pin_memory=torch.cuda.is_available(),
                    collate_fn=collate_train)

    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False,
                    num_workers=2, pin_memory=torch.cuda.is_available(),
                    collate_fn=collate_train)  # 验证也用同样collate
    
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False,
                    num_workers=2, pin_memory=torch.cuda.is_available(),
                    collate_fn=collate_test)

    # print("len(ds_te) =", len(ds_te))
    # print("len(dl_te) =", len(dl_te))

    model = SegNetVGG(in_channels=1, num_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss(ignore_index=-1)

    best = -1.0
    for ep in range(1, epochs + 1):
        model.train()
        running, n = 0.0, 0
        for batch in dl_tr:
            x = batch["img"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()

            running += float(loss.item())
            n += 1

        # ---- val ----
        model.eval()
        miou_sum, m = 0.0, 0
        for batch in dl_va:
            x = batch["img"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True)
            logits = model(x)
            miou_sum += mean_iou_batch(logits, y, num_classes=num_classes, ignore_index=-1)
            m += 1
        val_miou = miou_sum / max(1, m)

        print(f"epoch {ep:03d}/{epochs} loss={running/max(1,n):.4f}  val_mIoU={val_miou:.4f}")

        if val_miou > best:
            best = val_miou
            torch.save({"model": model.state_dict(), "num_classes": num_classes}, "model/segnet_vgg_best.pth")
            print("  saved best:", best)
            
    # 1) 用最后一轮（可选）
    make_submission(model, dl_te, device, Path("model/submission_last.csv"), size_labels=272)

    # 2) 用 best（推荐提交用这个）
    ckpt = torch.load("model/segnet_vgg_best.pth", map_location=device)
    model.load_state_dict(ckpt["model"])
    make_submission(model, dl_te, device, Path("model/submission_best.csv"), size_labels=272)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()