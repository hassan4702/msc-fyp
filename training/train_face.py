"""Train the FER-2013 face-emotion CNN, mapped to the 7 canonical labels.

Run on the M4 Max (MPS), then point the backend at the weights:

    python training/train_face.py --output models/weights/face/face_net.pt --epochs 30
    FACE_MODEL_PATH=models/weights/face/face_net.pt uvicorn backend.app:app

Labels are mapped by NAME from the dataset's own ClassLabel order (HF FER mirrors
disagree on index order). Augmentation (flip + small rotation) is done with PIL so
no torchvision dependency is needed. Loss is class-weighted (FER is "happy"-heavy).
"""
from __future__ import annotations

import argparse
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
    parser.add_argument("--output", default="models/weights/face/face_net.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=0, help="Subset train (0 = full); for smoke tests")
    args = parser.parse_args()

    import numpy as np
    import torch
    from datasets import load_dataset
    from PIL import Image
    from sklearn.metrics import accuracy_score, f1_score
    from torch.utils.data import DataLoader, Dataset

    from backend.models.face_net import INPUT_SIZE, NORM_MEAN, NORM_STD, FaceNet

    raw = load_dataset(args.dataset)
    names = raw["train"].features["label"].names
    print(f"dataset label order: {names}")

    def to_canonical(label_int: int) -> int:
        # .lower() because some mirrors capitalise the class names ("Angry").
        return EMOTIONS.index(FER_TO_CANONICAL[names[label_int].lower()])

    class FERDataset(Dataset):
        def __init__(self, hf, augment: bool):
            self.hf = hf
            self.augment = augment

        def __len__(self):
            return len(self.hf)

        def __getitem__(self, i):
            ex = self.hf[i]
            img = ex["image"].convert("L")
            if img.size != (INPUT_SIZE, INPUT_SIZE):
                img = img.resize((INPUT_SIZE, INPUT_SIZE))
            if self.augment:
                if random.random() < 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if random.random() < 0.5:
                    img = img.rotate(random.uniform(-12, 12))
            arr = np.asarray(img, dtype=np.float32)
            x = torch.from_numpy((arr / 255.0 - NORM_MEAN) / NORM_STD).unsqueeze(0)
            return x, to_canonical(ex["label"])

    # FER protocol: train -> publicTest (val) -> privateTest (test). Fall back to a
    # carved split / reuse for mirrors that only ship train+test or train only.
    available = set(raw.keys())
    train_hf = raw["train"]
    if "validation" in available:
        val_hf = raw["validation"]
    elif "publicTest" in available:
        val_hf = raw["publicTest"]
    else:
        carved = train_hf.train_test_split(test_size=args.val_split, seed=42)
        train_hf, val_hf = carved["train"], carved["test"]
    if "test" in available:
        test_hf = raw["test"]
    elif "privateTest" in available:
        test_hf = raw["privateTest"]
    else:
        test_hf = val_hf
    print(f"splits -> train {len(train_hf)} | val {len(val_hf)} | test {len(test_hf)}")

    if args.limit:
        train_hf = train_hf.select(range(min(args.limit, len(train_hf))))
        print(f"[smoke] limited train to {len(train_hf)} examples")

    train_loader = DataLoader(FERDataset(train_hf, augment=True), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(FERDataset(val_hf, augment=False), batch_size=args.batch_size)
    test_loader = DataLoader(FERDataset(test_hf, augment=False), batch_size=args.batch_size)

    counts = Counter(to_canonical(label) for label in train_hf["label"])
    total, n = sum(counts.values()), len(EMOTIONS)
    class_weights = torch.tensor([total / (n * counts.get(i, 1)) for i in range(n)], dtype=torch.float)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = FaceNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    @torch.no_grad()
    def evaluate(loader) -> tuple[float, float]:
        model.eval()
        preds, gold = [], []
        for x, y in loader:
            logits = model(x.to(device))
            preds += logits.argmax(-1).cpu().tolist()
            gold += y.tolist()
        return f1_score(gold, preds, average="macro"), accuracy_score(gold, preds)

    best_f1, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x.to(device)), torch.as_tensor(y).to(device))
            loss.backward()
            optimizer.step()
            running += loss.item()
        val_f1, val_acc = evaluate(val_loader)
        print(f"epoch {epoch:2d} | loss {running / len(train_loader):.3f} | val_f1 {val_f1:.4f} | val_acc {val_acc:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_f1, test_acc = evaluate(test_loader)
    print(f"TEST | macro_f1 {test_f1:.4f} | accuracy {test_acc:.4f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, args.output)
    print(f"Saved fine-tuned face model to {args.output}")


if __name__ == "__main__":
    main()
