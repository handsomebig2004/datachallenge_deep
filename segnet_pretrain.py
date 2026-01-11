from pathlib import Path
from functools import partial
import argparse
import math
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np

# 复用你 segnet.py 里的函数/模型
from segnet import SegNetVGG, patch_minmax, collate


class UnlabeledDataset(Dataset):
    """Loads *.npy from unlabeled_data (or any folder) and returns dict compatible with collate()."""
    def __init__(self, x_dir: Path):
        self.x_dir = Path(x_dir)
        self.files = sorted(self.x_dir.rglob("*.npy"))
        if len(self.files) == 0:
            raise RuntimeError(f"No .npy found under: {self.x_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        p = self.files[i]
        stem = p.stem
        img = np.load(p)

        # 兼容 HxW 或 HxWx1
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img[..., 0]

        img = patch_minmax(img)  # NaN->0 + minmax to [0,1]
        H, W = img.shape
        return {
            "id": stem,
            "orig_hw": (H, W),
            "img": torch.from_numpy(img).unsqueeze(0),  # [1,H,W]
        }


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def random_mask_like(x: torch.Tensor, mask_ratio: float, block: int = 8):
    """
    x: [B,1,H,W] in [0,1]
    Return x_masked, mask (1 means masked).
    Block masking: create mask on a coarse grid then upsample.
    """
    B, C, H, W = x.shape
    assert C == 1

    gh = math.ceil(H / block)
    gw = math.ceil(W / block)

    # coarse mask
    m = (torch.rand(B, 1, gh, gw, device=x.device) < mask_ratio).float()
    # upsample to full res
    m = torch.nn.functional.interpolate(m, size=(H, W), mode="nearest")
    x_masked = x * (1.0 - m)  # masked region -> 0
    return x_masked, m


@torch.no_grad()
def load_encoder_weights_from_ssl(seg_model: nn.Module, ssl_ckpt_path: Path, device):
    """Load only enc1~enc5 from ssl checkpoint into segmentation model."""
    ckpt = torch.load(ssl_ckpt_path, map_location=device)
    sd = ckpt["model"]

    own = seg_model.state_dict()
    for k in list(sd.keys()):
        if k.startswith("enc"):
            if k in own and own[k].shape == sd[k].shape:
                own[k].copy_(sd[k])
    seg_model.load_state_dict(own, strict=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x_unlabeled_dir", type=str, default="data/X_unlabeled_mtkxUlo")
    ap.add_argument("--out_ckpt", type=str, default="model/segnet_ssl_dae.pth")
    ap.add_argument("--size_h", type=int, default=160)
    ap.add_argument("--size_w", type=int, default=272)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mask_ratio", type=float, default=0.4)
    ap.add_argument("--mask_block", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = UnlabeledDataset(Path(args.x_unlabeled_dir))
    collate_u = partial(collate, size_hw=(args.size_h, args.size_w), is_train=False)

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_u,
        drop_last=True,
    )

    # Autoencoder: reuse SegNetVGG, but output 1 channel for reconstruction
    model = SegNetVGG(in_channels=1, num_classes=1).to(device)  # <-- 1 channel output
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    best = float("inf")
    out_path = Path(args.out_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        model.train()
        run, n = 0.0, 0

        for batch in dl:
            x = batch["img"].to(device, non_blocking=True)  # [B,1,160,272] in [0,1]
            x_masked, m = random_mask_like(x, args.mask_ratio, args.mask_block)

            opt.zero_grad(set_to_none=True)
            pred = model(x_masked)           # [B,1,H,W]
            pred = torch.sigmoid(pred)       # keep in [0,1]
            loss = mse(pred, x)              # reconstruct full image
            loss.backward()
            opt.step()

            run += float(loss.item())
            n += 1

        avg = run / max(1, n)
        print(f"[SSL] epoch {ep:03d}/{args.epochs} mse={avg:.6f}")

        if avg < best:
            best = avg
            torch.save(
                {
                    "model": model.state_dict(),
                    "ssl_type": "dae",
                    "size_hw": (args.size_h, args.size_w),
                    "mask_ratio": args.mask_ratio,
                    "mask_block": args.mask_block,
                },
                out_path,
            )
            print("  saved best ssl:", best, "->", out_path)

    print("done, best ssl mse:", best)


if __name__ == "__main__":
    main()
