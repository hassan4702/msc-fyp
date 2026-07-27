"""Fine-tune an ImageNet-pretrained ResNet-18 on FER-2013, mapped to the 7 canonical labels.

Replaces the from-scratch CNN in `training/train_face.py`, which topped out at 0.580
accuracy / 0.540 macro-F1 (de-duplicated) — about the published ceiling for a small
CNN trained from scratch on 48x48 FER images.

Three things differ from the old recipe, and all three matter:

1. **Pretrained backbone.** 11.7M ImageNet-initialised parameters instead of 0.98M from
   scratch. Costs 7.7 ms/frame on CPU — less than face detection.
2. **The same detector at train and inference time.** Both sides run BlazeFace and the
   same margin (`backend/models/face_detect.py`), so the framing cannot drift. The old
   pipeline trained on raw thumbnails and inferred on Haar boxes, and lost 6pp to it.
3. **De-duplication.** 8% of FER-2013's test split is pixel-identical to training images.
   Those rows are dropped from val/test so the reported score is not part memorisation.

Run:

    python training/train_face_resnet.py --epochs 15
    FACE_MODEL_PATH=models/weights/face/resnet18.pt uvicorn backend.app:app
"""
from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.emotions import EMOTIONS  # noqa: E402
from training.fer2013 import FER_TO_CANONICAL  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Aaryan333/fer2013_train_publicTest_privateTest")
    parser.add_argument("--output", default="models/weights/face/resnet18.pt")
    parser.add_argument("--cache", default="data/fer_crops.npz", help="cached BlazeFace crops")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="subset train (0 = full); smoke tests")
    args = parser.parse_args()

    import cv2
    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from torch.utils.data import DataLoader, Dataset

    from backend.models.face_net import (
        IMAGENET_MEAN,
        IMAGENET_STD,
        RESNET_INPUT_SIZE,
        build_resnet18,
    )

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    splits = _load_crops(args.dataset, args.cache)

    # --- de-duplicate: drop val/test rows that are pixel-identical to a training image ---
    train_hashes = {hashlib.md5(a.tobytes()).hexdigest() for a in splits["train"][0]}
    for name in ("val", "test"):
        imgs, labels = splits[name]
        keep = [i for i, a in enumerate(imgs) if hashlib.md5(a.tobytes()).hexdigest() not in train_hashes]
        dropped = len(imgs) - len(keep)
        splits[name] = ([imgs[i] for i in keep], [labels[i] for i in keep])
        print(f"{name}: dropped {dropped}/{len(imgs)} images leaked from train ({dropped / len(imgs):.2%})")

    if args.limit:
        imgs, labels = splits["train"]
        splits["train"] = (imgs[:args.limit], labels[:args.limit])
        print(f"[smoke] limited train to {len(splits['train'][0])} examples")
    print({k: len(v[0]) for k, v in splits.items()})

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    class FER(Dataset):
        def __init__(self, imgs, labels, augment):
            self.imgs, self.labels, self.augment = imgs, labels, augment

        def __len__(self):
            return len(self.imgs)

        def __getitem__(self, i):
            img = self.imgs[i]
            if self.augment:
                if random.random() < 0.5:
                    img = np.ascontiguousarray(img[:, ::-1])
                if random.random() < 0.5:
                    angle = random.uniform(-12, 12)
                    h, w = img.shape
                    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                    # borderReplicate, not black: zero corners are far outside the
                    # normalised range and no real face produces them.
                    img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
                if random.random() < 0.5:
                    img = np.clip(img.astype(np.float32) * random.uniform(0.7, 1.3), 0, 255).astype(np.uint8)
            resized = cv2.resize(img, (RESNET_INPUT_SIZE, RESNET_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
            t = torch.from_numpy(resized.astype(np.float32) / 255.0).unsqueeze(0).repeat(3, 1, 1)
            return (t - mean) / std, self.labels[i]

    loaders = {
        name: DataLoader(FER(*splits[name], augment=(name == "train")),
                         batch_size=args.batch_size, shuffle=(name == "train"), num_workers=0)
        for name in splits
    }

    counts = Counter(splits["train"][1])
    total, n = sum(counts.values()), len(EMOTIONS)
    class_weights = torch.tensor([total / (n * counts.get(i, 1)) for i in range(n)], dtype=torch.float)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = build_resnet18().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Cosine decay: the old recipe held lr at 1e-3 for 30 epochs, the one real recipe gap.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.05)

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        preds, gold = [], []
        for x, y in loader:
            preds += model(x.to(device)).argmax(-1).cpu().tolist()
            gold += list(y)
        return f1_score(gold, preds, average="macro"), accuracy_score(gold, preds)

    best_f1, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in loaders["train"]:
            optimizer.zero_grad()
            loss = loss_fn(model(x.to(device)), torch.as_tensor(y).to(device))
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()
        val_f1, val_acc = evaluate(loaders["val"])
        print(f"epoch {epoch:2d} | loss {running / len(loaders['train']):.3f} | "
              f"val_f1 {val_f1:.4f} | val_acc {val_acc:.4f} | lr {scheduler.get_last_lr()[0]:.2e}")
        if val_f1 > best_f1:
            best_f1, best_state = val_f1, {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_f1, test_acc = evaluate(loaders["test"])
    print(f"TEST (de-duplicated) | macro_f1 {test_f1:.4f} | accuracy {test_acc:.4f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.output)
    print(f"Saved to {args.output}")


def _load_crops(dataset: str, cache: str):
    """BlazeFace-crop every FER image once and cache it; detection is too slow per-epoch."""
    import numpy as np
    from datasets import load_dataset

    path = Path(cache)
    if path.is_file():
        z = np.load(path, allow_pickle=True)
        print(f"loaded cached crops from {cache}")
        return {name: (list(z[f"{name}_x"]), list(z[f"{name}_y"])) for name in ("train", "val", "test")}

    from backend.models.face_detect import FaceDetector

    detector = FaceDetector()
    raw = load_dataset(dataset)
    names = raw["train"].features["label"].names
    source = {"train": raw["train"], "val": raw["publicTest"], "test": raw["privateTest"]}
    out, misses = {}, 0
    for name, hf in source.items():
        imgs, labels = [], []
        for i, ex in enumerate(hf):
            gray = np.asarray(ex["image"].convert("L"), dtype=np.uint8)
            crop = detector.crop(gray)
            if crop is None:
                crop = gray  # ponytail: keep the thumbnail; BlazeFace misses <1% of FER
                misses += 1
            imgs.append(crop)
            labels.append(EMOTIONS.index(FER_TO_CANONICAL[names[ex["label"]].lower()]))
            if (i + 1) % 5000 == 0:
                print(f"  {name}: cropped {i + 1}/{len(hf)}")
        out[name] = (imgs, labels)
        print(f"  {name}: {len(imgs)} crops")
    print(f"detector missed {misses} images (fell back to the raw thumbnail)")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f"{n}_x": np.array(v[0], dtype=object) for n, v in out.items()},
                        **{f"{n}_y": np.array(v[1]) for n, v in out.items()})
    print(f"cached crops to {cache}")
    return out


if __name__ == "__main__":
    main()
