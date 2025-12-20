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
    def __init__(self, x_dir: Path, y_df: pd.DataFrame | None = None):
        self.x_dir = Path(x_dir)
        self.y_df = y_df
        self.is_train = y_df is not None
        files = sorted(self.x_dir.rglob("*.npy"))
        if self.is_train:
            idx = set(self.y_df.index.astype(str))
            files = [p for p in files if p.stem in idx]
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

def collate(batch, train_size=(160,160), is_train=True):
    # pad to max H/W in batch then resize to train_size (baseline-style) :contentReference[oaicite:2]{index=2}
    Hmax = max(b["img"].shape[-2] for b in batch)
    Wmax = max(b["img"].shape[-1] for b in batch)

    imgs = []
    for b in batch:
        x = b["img"]
        x = F.pad(x, (0, Wmax - x.shape[-1], 0, Hmax - x.shape[-2]), value=0.0)
        imgs.append(x)
    imgs = torch.stack(imgs, 0)          # [B,1,Hmax,Wmax]
    imgs = resize_img(imgs, train_size)  # [B,1,160,160]

    out = {"id":[b["id"] for b in batch], "orig_hw":[b["orig_hw"] for b in batch], "img":imgs}

    if is_train:
        masks = []
        for b in batch:
            y = b["mask"]  # [160,W]
            y = F.pad(y, (0, Wmax - y.shape[-1], 0, 0), value=0)
            masks.append(y)
        masks = torch.stack(masks, 0)         # [B,160,Wmax]
        masks = resize_mask(masks, train_size) # [B,160,160]
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

    train_size = (160,160)  # baseline-style resize :contentReference[oaicite:4]{index=4}
    batch_size = 16
    epochs = 20
    lr = 1e-4
    collate_train = partial(collate, train_size=train_size, is_train=True)
    collate_test  = partial(collate, train_size=train_size, is_train=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # speed up on fixed input size

    y_df = pd.read_csv(Y_TRAIN_CSV, index_col=0)
    num_classes = infer_num_classes(y_df)
    print("device:", device, "num_classes:", num_classes, "train_samples:", len(y_df))

    ds_tr = WellDataset(X_TRAIN_DIR, y_df=y_df)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                   num_workers=2, pin_memory=torch.cuda.is_available(),
                   collate_fn=collate_train)

    ds_te = WellDataset(X_TEST_DIR, y_df=None)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False,
                   num_workers=2, pin_memory=torch.cuda.is_available(),
                   collate_fn=collate_test)

    # print("len(ds_te) =", len(ds_te))
    # print("len(dl_te) =", len(dl_te))

    model = SegNetVGG(in_channels=1, num_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    model.train()
    for ep in range(1, epochs + 1):
        running = 0.0
        n = 0
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
        print(f"epoch {ep:03d}/{epochs}  loss={running/max(1,n):.4f}")

    torch.save({"model": model.state_dict(), "num_classes": num_classes}, "model/segnet_vgg.pth")
    make_submission(model, dl_te, device, Path("model/submission_segnet_vgg.csv"), size_labels=272)

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()