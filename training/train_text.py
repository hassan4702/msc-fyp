"""Fine-tune DistilBERT on GoEmotions, mapped to the 7 canonical Ekman labels.

Run on the M4 Max (uses MPS automatically), then point the backend at the output:

    python training/train_text.py --output models/weights/text --epochs 3
    TEXT_MODEL_DIR=models/weights/text uvicorn backend.app:app

The 27->7 collapse uses the official Ekman mapping (see training/goemotions.py);
GoEmotions multi-label examples that span >1 Ekman bucket are dropped (logged).
Label index i == EMOTIONS[i], which is the ordering the backend wrapper assumes.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Allow running as a plain script (`python training/train_text.py`) by putting
# the repo root on the path so `backend` and `training` are importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.emotions import EMOTIONS  # noqa: E402
from training.goemotions import single_ekman_label  # noqa: E402


def _prepare(dataset, tokenizer, max_length: int):
    """Map GoEmotions -> single canonical label index, drop ambiguous, tokenize."""
    def to_single(example):
        label = single_ekman_label(example["labels"])
        example["label"] = EMOTIONS.index(label) if label is not None else -1
        return example

    before = {split: len(dataset[split]) for split in dataset}
    dataset = dataset.map(to_single)
    dataset = dataset.filter(lambda e: e["label"] != -1)
    for split in dataset:
        kept, total = len(dataset[split]), before[split]
        print(f"[{split}] kept {kept}/{total} ({100 * kept / total:.1f}%) after Ekman collapse")

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    dataset = dataset.map(tokenize, batched=True)
    # Drop everything except the model inputs + our single int label, otherwise the
    # collator tries to tensorize GoEmotions' original variable-length `labels` list.
    keep = {"input_ids", "attention_mask", "label"}
    drop = [c for c in dataset["train"].column_names if c not in keep]
    return dataset.remove_columns(drop)


def _class_weights(labels: list[int], torch):
    counts = Counter(labels)
    total = len(labels)
    n = len(EMOTIONS)
    weights = [total / (n * counts.get(i, 1)) for i in range(n)]
    return torch.tensor(weights, dtype=torch.float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--output", default="models/weights/text")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0, help="Subset each split (0 = full); for smoke tests")
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    raw = load_dataset("google-research-datasets/go_emotions", "simplified")
    data = _prepare(raw, tokenizer, args.max_length)

    if args.limit:
        for split in list(data.keys()):
            data[split] = data[split].select(range(min(args.limit, len(data[split]))))
        print(f"[smoke] limited each split to {args.limit} examples")

    class_weights = _class_weights(data["train"]["label"], torch)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(EMOTIONS),
        id2label={i: e for i, e in enumerate(EMOTIONS)},
        label2id={e: i for i, e in enumerate(EMOTIONS)},
    )

    def compute_metrics(pred):
        preds = pred.predictions.argmax(-1)
        return {
            "f1": f1_score(pred.label_ids, preds, average="macro"),
            "accuracy": accuracy_score(pred.label_ids, preds),
        }

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(outputs.logits.device))
            loss = loss_fct(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=100,
        report_to="none",
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    trainer.train()
    test_metrics = trainer.evaluate(data["test"])
    print("TEST:", test_metrics)

    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved fine-tuned text model to {args.output}")


if __name__ == "__main__":
    main()
