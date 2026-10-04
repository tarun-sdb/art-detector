"""Finetune convnext-tiny, export ONNX. Colab T4-ready.
ponytail: one backbone, BCE 1-logit, no CLIP branch (add when cross-gen <70%).
Usage: pip install -r requirements.txt && python train.py --data ./data --epochs 5 --smoke to verify without data
"""
import argparse, io
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from sklearn.metrics import roc_auc_score

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def loaders(data, size, bs):
    tf = lambda train: transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip() if train else nn.Identity(),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    # test_holdout/fake has no real/ subfolder -> ImageFolder skips it; eval it separately below
    tr = DataLoader(datasets.ImageFolder(f"{data}/train", tf(True)), batch_size=bs, shuffle=True, num_workers=2)
    va = DataLoader(datasets.ImageFolder(f"{data}/val", tf(False)), batch_size=bs, num_workers=2)
    return tr, va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="./models")
    ap.add_argument("--smoke", action="store_true", help="1 synthetic batch, no data needed")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model = timm.create_model("convnext_tiny", pretrained=not args.smoke, num_classes=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    if args.smoke:  # ponytail: minimal runnable check
        x, y = torch.randn(4, 3, 64, 64).to(dev), torch.randint(0, 2, (4, 1)).float().to(dev)
        loss_fn(model(x), y).backward(); opt.step()
        torch.onnx.export(model.cpu().eval(), torch.randn(1, 3, 64, 64), str(out / "model_smoke.onnx"),
                          input_names=["input"], output_names=["logit"], opset_version=17)
        print("smoke ok ->", out / "model_smoke.onnx")
        return

    tr, va = loaders(args.data, args.size, args.bs)
    best = 0.0
    for ep in range(args.epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(dev), y.float().unsqueeze(1).to(dev)
            opt.zero_grad(); loss_fn(model(x), y).backward(); opt.step()
        # val
        model.eval(); ps, ys = [], []
        with torch.no_grad():
            for x, y in va:
                ps += torch.sigmoid(model(x.to(dev))).cpu().flatten().tolist(); ys += y.tolist()
        auc = roc_auc_score(ys, ps)
        print(f"epoch {ep+1}/{args.epochs} val_auc={auc:.4f}")
        if auc > best:
            best = auc
            torch.save(model.state_dict(), out / "best.pt")
            torch.onnx.export(model.cpu().eval(), torch.randn(1, 3, args.size, args.size),
                              str(out / "model.onnx"), input_names=["input"],
                              output_names=["logit"], opset_version=17,
                              dynamic_axes={"input": {0: "batch"}})
            model.to(dev)
    print(f"best_auc={best:.4f} -> {out}/model.onnx")


if __name__ == "__main__":
    main()
