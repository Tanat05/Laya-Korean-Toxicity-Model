import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


CATEGORIES = (
    "sexual",
    "sexual_minors",
    "hate",
    "harassment",
    "threatening",
    "violence",
    "illicit",
    "self_harm",
    "spam",
)


class LayaModerationClassifier(nn.Module):
    def __init__(self, encoder, num_classes=len(CATEGORIES), dropout=0.2):
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.category_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )
        self.flagged_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        pooled = self.dropout(pooled)
        return self.category_head(pooled), self.flagged_head(pooled).squeeze(-1)


def load_model(model_dir, device):
    model_dir = Path(model_dir)
    required_files = (model_dir / "model.pt", model_dir / "encoder", model_dir / "tokenizer")
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Model files are missing: {', '.join(missing)}")

    config_path = model_dir / "training_summary.json"
    if config_path.is_file():
        with config_path.open(encoding="utf-8") as file:
            config = json.load(file)
        categories = config.get("categories", list(CATEGORIES))
        if tuple(categories) != CATEGORIES:
            raise ValueError(f"Unsupported model categories: {categories}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    encoder = AutoModel.from_pretrained(model_dir / "encoder")
    model = LayaModerationClassifier(encoder).to(device)
    state_dict = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model, tokenizer
