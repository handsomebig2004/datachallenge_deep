# make_submission_best.py
from pathlib import Path
from functools import partial
import argparse

import torch
from torch.utils.data import DataLoader

# 直接复用 segnet.py 里的实现
from segnet import SegNetVGG, WellDataset, collate, make_submission


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="model/segnet_vgg_best.pth", help="path to best checkpoint")
    ap.add_argument("--x_test_dir", type=str, default="data/X_test_QB9kE8D", help="test npy directory")
    ap.add_argument("--out_csv", type=str, default="model/submission_best.csv", help="output submission csv")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--size_h", type=int, default=160, help="collate target height")
    ap.add_argument("--size_w", type=int, default=272, help="collate target width")
    ap.add_argument("--size_labels", type=int, default=272, help="csv label width (usually 272)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if "num_classes" not in ckpt:
        raise KeyError("Checkpoint missing key 'num_classes'.")
    if "model" not in ckpt:
        raise KeyError("Checkpoint missing key 'model' (state_dict).")

    num_classes = int(ckpt["num_classes"])
    model = SegNetVGG(in_channels=1, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    x_test_dir = Path(args.x_test_dir)
    if not x_test_dir.exists():
        raise FileNotFoundError(f"Test dir not found: {x_test_dir}")

    ds_te = WellDataset(x_test_dir, y_df=None)

    collate_test = partial(collate, size_hw=(args.size_h, args.size_w), is_train=False)
    dl_te = DataLoader(
        ds_te,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_test,
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # 调用 segnet.py 里的提交生成函数
    make_submission(model, dl_te, device, out_csv, size_labels=args.size_labels)


if __name__ == "__main__":
    main()
