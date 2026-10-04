"""Build style-matched dataset from local real/fake art folders.
ponytail: stdlib + Pillow only, no new deps. Shuffle split, no stratify-by-style yet (add when style labels exist).
Usage:
  python download.py --real-dir /path/to/real_digital --fake-dir /path/to/ai_digital --out ./data --val 0.1 --test 0.15
Fetch data manually (digital first):
  real: WikiArt digital / ArtStation CC / Danbooru-safe sample
  fake: AI-ArtBench SD + Midjourney digital, DiffusionDB sample, custom SDXL prompts
Layout out: train/real, train/fake, val/real, val/fake, test_holdout/fake (+ manifest.csv)
"""
import argparse, csv, random, shutil
from pathlib import Path
from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def collect(d):
    p = Path(d)
    files = [f for f in p.rglob("*") if f.suffix.lower() in EXTS]
    if not files:
        raise SystemExit(f"no images in {d}")
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir", required=True)
    ap.add_argument("--fake-dir", required=True)
    ap.add_argument("--out", default="./data")
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--size", type=int, default=384, help="resize longest side, JPEG-match both classes")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    for s in ["train/real", "train/fake", "val/real", "val/fake", "test_holdout/fake"]:
        (out / s).mkdir(parents=True, exist_ok=True)

    real, fake = collect(args.real_dir), collect(args.fake_dir)
    rng.shuffle(real); rng.shuffle(fake)
    # holdout = last test-fraction of fake as unseen-generator proxy (ideally a different generator folder)
    n_hold = max(1, int(len(fake) * args.test))
    hold, fake = fake[:n_hold], fake[n_hold:]

    def split(files):
        n_v = max(1, int(len(files) * args.val))
        return files[n_v:], files[:n_v]  # train, val

    r_tr, r_va = split(real)
    f_tr, f_va = split(fake)
    rows = []
    for split_name, label, files in [("train", "real", r_tr), ("val", "real", r_va),
                                     ("train", "fake", f_tr), ("val", "fake", f_va),
                                     ("test_holdout", "fake", hold)]:
        for i, src in enumerate(files):
            dst = out / split_name / label / f"{label}_{i:06d}.jpg"
            try:
                im = Image.open(src).convert("RGB")
                im.thumbnail((args.size, args.size))
                im.save(dst, "JPEG", quality=92)  # same compression both classes
            except Exception as e:
                print(f"skip {src}: {e}")
                continue
            rows.append((str(dst), label, split_name, src.name))
    with open(out / "manifest.csv", "w", newline="") as f:
        csv.writer(f).writerows([("path", "label", "split", "src")] + rows)
    print(f"wrote {len(rows)} images -> {out} (train/val/test_holdout)")


if __name__ == "__main__":
    main()
