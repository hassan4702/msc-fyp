"""Fine-tune the face model on MELD, so RQ1/RQ2 report in-domain rather than transfer numbers.

Everything in §9 of PROJECT.md is currently **cross-corpus**: the face model is trained
on FER-2013 (posed, cropped, grayscale stills) and applied to MELD (TV dialogue frames).
That domain gap, not the fusion design, is the main reason face-only macro-F1 sits near
the chance baseline. This script closes it by fine-tuning the FER-2013 model on MELD's
own train split, then re-scoring on MELD test.

An honest caveat, which belongs in the write-up: **MELD labels are utterance-level** and
were annotated from audio + text + video together. A single face frame often does not
display the labelled emotion at all (the speaker may be off-camera, mid-sentence, or
reacting to someone else). So this is a noisy-label problem by construction, and a
ceiling well below FER-2013 in-domain accuracy is expected, not a failure.

Run (after training/train_face_resnet.py):

    python training/finetune_meld.py --epochs 6
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.emotions import EMOTIONS  # noqa: E402
from evaluation.meld import meld_to_canonical  # noqa: E402


def _cache_crops(split: str, labels_dir: str, videos_dir: str, cache: str, frames_per_clip: int):
    """Detect and cache face crops from MELD videos once; detection is far too slow per-epoch."""
    import cv2
    import numpy as np

    path = Path(cache)
    if path.is_file():
        z = np.load(path, allow_pickle=True)
        print(f"loaded {split} crops from {cache}")
        return list(z["x"]), list(z["y"])

    from backend.models.face_detect import FaceDetector

    detector = FaceDetector()
    rows = list(csv.DictReader(open(Path(labels_dir) / f"{split}_sent_emo.csv",
                                    newline="", encoding="utf-8", errors="ignore")))
    imgs, labels, missing, no_face = [], [], 0, 0
    for i, r in enumerate(rows):
        video = Path(videos_dir) / f"{split}_dia{r['Dialogue_ID']}_utt{r['Utterance_ID']}.mp4"
        if not video.exists():
            missing += 1
            continue
        cap = cv2.VideoCapture(str(video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        # Spread the sampled frames across the clip: the emotion is not evenly expressed,
        # and several draws give the noisy utterance label more chances to be visible.
        picks = [max(0, int(total * f)) for f in (0.25, 0.5, 0.75)][:frames_per_clip] if total > 3 else [0]
        label = EMOTIONS.index(meld_to_canonical(r["Emotion"]))
        for idx in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            crop = detector.crop(frame)  # colour in, grayscale crop out
            if crop is None:
                no_face += 1
                continue
            imgs.append(cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA))
            labels.append(label)
        cap.release()
        if (i + 1) % 1000 == 0:
            print(f"  {split}: {i + 1}/{len(rows)} clips -> {len(imgs)} crops")
    print(f"  {split}: {len(imgs)} crops | {missing} videos missing | {no_face} frames with no face")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=np.array(imgs), y=np.array(labels))
    return imgs, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="models/weights/face/resnet18.pt")
    parser.add_argument("--output", default="models/weights/face/resnet18_meld.pt")
    parser.add_argument("--labels-dir", default="data/meld/labels")
    parser.add_argument("--videos-dir", default="data/meld/MELD_raw/videos")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-5)  # low: fine-tuning, noisy labels
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--frames-per-clip", type=int, default=3)
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

    data = {
        split: _cache_crops(split, args.labels_dir, args.videos_dir,
                            f"data/meld_crops_{split}.npz", args.frames_per_clip)
        for split in ("train", "dev", "test")
    }
    print({k: len(v[0]) for k, v in data.items()})

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    class Frames(Dataset):
        def __init__(self, imgs, labels, augment):
            self.imgs, self.labels, self.augment = imgs, labels, augment

        def __len__(self):
            return len(self.imgs)

        def __getitem__(self, i):
            img = self.imgs[i]
            if self.augment and random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])
            resized = cv2.resize(img, (RESNET_INPUT_SIZE, RESNET_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
            t = torch.from_numpy(resized.astype(np.float32) / 255.0).unsqueeze(0).repeat(3, 1, 1)
            return (t - mean) / std, int(self.labels[i])

    loaders = {
        s: DataLoader(Frames(*data[s], augment=(s == "train")),
                      batch_size=args.batch_size, shuffle=(s == "train"))
        for s in data
    }

    counts = Counter(int(v) for v in data["train"][1])
    total, n = sum(counts.values()), len(EMOTIONS)
    weights = torch.tensor([total / (n * counts.get(i, 1)) for i in range(n)], dtype=torch.float)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = build_resnet18(pretrained=False)
    model.load_state_dict(torch.load(args.base, map_location="cpu"))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.1)

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        preds, gold = [], []
        for x, y in loader:
            preds += model(x.to(device)).argmax(-1).cpu().tolist()
            gold += y.tolist()
        return f1_score(gold, preds, average="macro"), accuracy_score(gold, preds)

    base_f1, base_acc = evaluate(loaders["test"])
    print(f"BEFORE fine-tuning | MELD test frame-level macro_f1 {base_f1:.4f} | acc {base_acc:.4f}")

    best_f1, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in loaders["train"]:
            optimizer.zero_grad()
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()
        dev_f1, dev_acc = evaluate(loaders["dev"])
        print(f"epoch {epoch:2d} | loss {running / len(loaders['train']):.3f} | dev_f1 {dev_f1:.4f} | dev_acc {dev_acc:.4f}")
        if dev_f1 > best_f1:
            best_f1, best_state = dev_f1, {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_f1, test_acc = evaluate(loaders["test"])
    print(f"AFTER fine-tuning  | MELD test frame-level macro_f1 {test_f1:.4f} | acc {test_acc:.4f}")
    print(f"delta macro_f1 {test_f1 - base_f1:+.4f} | acc {test_acc - base_acc:+.4f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
