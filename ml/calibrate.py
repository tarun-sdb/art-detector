"""Print val score ranges + midpoint threshold. Usage: python calibrate.py --data ./data --ckpt ./models/best.pt --size 64"""
import argparse
import torch, timm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--size", type=int, default=64)
a = ap.parse_args()
m = timm.create_model("convnext_tiny", pretrained=False, num_classes=1)
m.load_state_dict(torch.load(a.ckpt, map_location="cpu")); m.eval()
tf = transforms.Compose([transforms.Resize((a.size, a.size)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
ds = datasets.ImageFolder(f"{a.data}/val", tf)
rf, rl = [], []
with torch.no_grad():
    for x, y in DataLoader(ds, batch_size=32):
        for cls, p in zip(y.tolist(), torch.sigmoid(m(x)).flatten().tolist()):
            (rf if ds.classes[cls] == "fake" else rl).append(p)
print(f"fake_min={min(rf):.4f} real_max={max(rl):.4f} threshold={(max(rl) + min(rf)) / 2:.4f}")
