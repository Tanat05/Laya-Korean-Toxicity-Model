import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

import laya
from toxicity_model import CATEGORIES, LayaModerationClassifier


BENIGN_PATTERN = re.compile(
    r"(ㅈ[0-9\s_.-]*ㅂ|ㅜㅠ|ㅠㅠ|ㅜㅜ|[ㅜㅠ]{1,}|ㅈㅉ|ㅈㅅ|ㄹㅈㄷ|ㄱㄴ|ㄱㄷ|"
    r"ㅅㄱ|ㅂㅂ|ㅊㅋ|ㅎㅇ|ㅂㄷㅂㄷ|ㄹㅇ|ㅇㅈ|ㅇㅇ|ㄷㄷ|ㅋㅋ|ㅎㅎ|바보야|"
    r"시비 먼저|해킹범|뭐ㅈ|토로 울까바)"
)
PROFANITY_PATTERN = re.compile(
    r"(시발|씨발|ㅅㅂ|ㅆㅂ|시벌|시팔|스발|시2발|병신|ㅂㅅ|개새|새끼|새꺄|새기|"
    r"개같|존나|ㅈㄴ|좆|자지|보지|느금|니애미|애미|창녀|닥쳐|지랄|ㅈㄹ|꺼져|"
    r"뒤져|ㅗ|凸|따먹|성관계|자살|살해|패드립|앰티씨발|개패)"
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_records(data_dir):
    files = sorted(Path(data_dir).rglob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found under {data_dir}")

    records = []
    malformed = 0
    for path in files:
        with path.open(encoding="utf-8") as file:
            for line in file:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                if not isinstance(item, dict):
                    malformed += 1
                    continue
                text = item.get("text")
                if not isinstance(text, str) or not text.strip():
                    continue

                target = item.get("target") or {}
                if not isinstance(target, dict):
                    target = {}
                source_categories = target.get("categories") or {}
                if not isinstance(source_categories, dict):
                    source_categories = {}
                labels = {}
                for category in CATEGORIES:
                    value = source_categories.get(category)
                    if value is None:
                        value = source_categories.get(category.replace("_", "-"))
                    if value is None:
                        value = source_categories.get(category.replace("_", "/"), False)
                    labels[category] = int(bool(value))
                flagged = int(bool(target.get("flagged", False)))
                if BENIGN_PATTERN.search(text) and not PROFANITY_PATTERN.search(text):
                    flagged = 0
                records.append({"text": text, "labels": labels, "flagged": flagged})

    if not records:
        raise ValueError(f"No usable records found in {len(files)} JSONL file(s)")
    if malformed:
        print(f"Skipped {malformed} malformed JSONL line(s).")
    print(f"Loaded {len(records):,} records from {len(files)} JSONL file(s).")
    return records


def split_records(records, seed):
    rng = random.Random(seed)
    groups = {0: [], 1: []}
    for record in records:
        groups[record["flagged"]].append(record)

    splits = ([], [], [])
    for group in groups.values():
        rng.shuffle(group)
        train_end = int(len(group) * 0.70)
        validation_end = int(len(group) * 0.85)
        splits[0].extend(group[:train_end])
        splits[1].extend(group[train_end:validation_end])
        splits[2].extend(group[validation_end:])
    for split in splits:
        rng.shuffle(split)
    if any(not split for split in splits):
        raise ValueError("The dataset must contain enough examples for all splits.")
    return splits


class ToxicityDataset(Dataset):
    def __init__(self, records, tokenizer, max_length):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        encoded = self.tokenizer(
            record["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(
                [record["labels"][category] for category in CATEGORIES],
                dtype=torch.float32,
            ),
            "flagged": torch.tensor(record["flagged"], dtype=torch.float32),
        }


def macro_f1(targets, predictions):
    scores = []
    for index in range(len(CATEGORIES)):
        target = targets[:, index]
        prediction = predictions[:, index]
        true_positive = np.logical_and(target == 1, prediction == 1).sum()
        false_positive = np.logical_and(target == 0, prediction == 1).sum()
        false_negative = np.logical_and(target == 1, prediction == 0).sum()
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return float(np.mean(scores))


def evaluate(model, loader, device, category_loss, flagged_loss, amp_enabled):
    model.eval()
    total_loss = 0.0
    predictions = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            flagged = batch["flagged"].to(device)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                category_logits, flagged_logits = model(input_ids, attention_mask)
                loss = category_loss(category_logits, labels) + 0.5 * flagged_loss(
                    flagged_logits, flagged
                )
            total_loss += loss.item()
            predictions.append((torch.sigmoid(category_logits) >= 0.5).cpu().numpy())
            targets.append(labels.cpu().numpy())
    all_predictions = np.concatenate(predictions)
    all_targets = np.concatenate(targets)
    return total_loss / len(loader), macro_f1(all_targets, all_predictions)


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    print(f"Using device: {device}")

    records = read_records(args.data_dir)
    train_records, validation_records, test_records = split_records(records, args.seed)
    print(
        "Split sizes: "
        f"train={len(train_records):,}, validation={len(validation_records):,}, "
        f"test={len(test_records):,}"
    )

    base_agent = laya.load("convaiinnovations/laya", subfolder="multilingual")
    tokenizer = base_agent.tok
    model = LayaModerationClassifier(base_agent.model.encoder).to(device)
    train_loader = DataLoader(
        ToxicityDataset(train_records, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        ToxicityDataset(validation_records, tokenizer, args.max_length),
        batch_size=args.batch_size,
    )
    test_loader = DataLoader(
        ToxicityDataset(test_records, tokenizer, args.max_length),
        batch_size=args.batch_size,
    )

    positive_counts = np.array(
        [sum(record["labels"][category] for record in train_records) for category in CATEGORIES]
    )
    negative_counts = len(train_records) - positive_counts
    positive_weights = np.clip(
        np.sqrt(negative_counts / (positive_counts + 1e-5)), 1.0, 5.0
    )
    category_loss = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weights, dtype=torch.float32, device=device)
    )
    flagged_loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_weights = output_dir / ".best_model.pt"
    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            flagged = batch["flagged"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                category_logits, flagged_logits = model(input_ids, attention_mask)
                loss = category_loss(category_logits, labels) + 0.5 * flagged_loss(
                    flagged_logits, flagged
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item()

        validation_loss, validation_f1 = evaluate(
            model, validation_loader, device, category_loss, flagged_loss, amp_enabled
        )
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={running_loss / len(train_loader):.4f} "
            f"val_loss={validation_loss:.4f} val_macro_f1={validation_f1:.4f} "
            f"elapsed={time.time() - start_time:.1f}s"
        )
        if validation_f1 > best_f1:
            best_f1 = validation_f1
            torch.save(model.state_dict(), best_weights)

    model.load_state_dict(torch.load(best_weights, map_location=device, weights_only=True))
    test_loss, test_f1 = evaluate(
        model, test_loader, device, category_loss, flagged_loss, amp_enabled
    )
    best_weights.unlink(missing_ok=True)

    tokenizer.save_pretrained(output_dir / "tokenizer")
    model.encoder.save_pretrained(output_dir / "encoder")
    torch.save(model.state_dict(), output_dir / "model.pt")
    summary = {
        "model_type": "laya-moderation",
        "base_model": "convaiinnovations/laya-multilingual",
        "categories": list(CATEGORIES),
        "max_length": args.max_length,
        "best_val_macro_f1": best_f1,
        "test_loss": test_loss,
        "test_macro_f1": test_f1,
        "train_samples": len(train_records),
        "val_samples": len(validation_records),
        "benchmark_samples": len(test_records),
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    print(f"Test loss={test_loss:.4f}, test macro F1={test_f1:.4f}")
    print(f"Model artifacts saved to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Laya for Korean toxicity moderation.")
    parser.add_argument("--data-dir", default="data", help="Directory containing JSONL files.")
    parser.add_argument("--output-dir", default="laya-korean-toxicity-model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 1:
        parser.error("epochs, batch-size, and max-length must be positive.")
    return args


if __name__ == "__main__":
    train(parse_args())
