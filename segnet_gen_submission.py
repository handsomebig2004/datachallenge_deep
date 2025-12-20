from segnet import *

def main():
    ROOT = Path(".\\data")
    X_TEST_DIR  = ROOT / "X_test_xNbnvIa"
    train_size = (160,160)
    batch_size = 16
    collate_test  = partial(collate, train_size=train_size, is_train=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_te = WellDataset(X_TEST_DIR, y_df=None)
    print("X_TEST_DIR =", X_TEST_DIR, "exists =", Path(X_TEST_DIR).exists())
    test_files = list(Path(X_TEST_DIR).rglob("*.npy"))
    print("test npy found =", len(test_files))
    if len(test_files) > 0:
        print("first test file =", test_files[0])
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False,
                    num_workers=2, pin_memory=torch.cuda.is_available(),
                    collate_fn=collate_test)
    ckpt = torch.load("segnet_vgg.pth", map_location="cuda")
    num_classes = ckpt["num_classes"]
    model = SegNetVGG(in_channels=1, num_classes=num_classes)
    model.load_state_dict(ckpt["model"])
    model.cuda().eval()

    make_submission(model, dl_te, device, Path("model/submission_segnet_vgg.csv"), size_labels=272)

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()