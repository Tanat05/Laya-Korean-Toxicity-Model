import argparse
import json
from pathlib import Path

import torch

from toxicity_model import CATEGORIES, load_model


def predict(model_dir, text, threshold=0.5, max_length=None):
    if not text.strip():
        raise ValueError("Text must not be empty.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")
    if max_length is None:
        summary_path = Path(model_dir) / "training_summary.json"
        if summary_path.is_file():
            with summary_path.open(encoding="utf-8") as file:
                max_length = json.load(file).get("max_length", 128)
        else:
            max_length = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(model_dir, device)
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        category_logits, flagged_logit = model(
            encoded["input_ids"], encoded["attention_mask"]
        )
        category_scores = torch.sigmoid(category_logits)[0].cpu().tolist()
        flagged_score = torch.sigmoid(flagged_logit)[0].item()

    category_results = dict(zip(CATEGORIES, category_scores))
    toxicity_score = max([flagged_score, *category_scores])
    return {
        "flagged": toxicity_score >= threshold,
        "toxicity_score": toxicity_score,
        "categories": category_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Classify Korean text with a fine-tuned Laya model.")
    parser.add_argument("model_dir", help="Path to the downloaded or locally trained model.")
    parser.add_argument("text", help="Text to classify.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-length", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            predict(args.model_dir, args.text, args.threshold, args.max_length),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
