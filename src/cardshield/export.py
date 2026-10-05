"""Export the champion model to artifacts/model/ (model.txt + metadata.json)."""

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb

from cardshield.config import MODEL_DIR

MODEL_FILE = "model.txt"
METADATA_FILE = "metadata.json"


def export_model(booster: lgb.Booster, metadata: dict[str, Any], out_dir: Path = MODEL_DIR) -> Path:
    """Write the booster and its metadata. `metadata["features"]` fixes serve-time order."""
    if list(booster.feature_name()) != list(metadata["features"]):
        raise ValueError("metadata feature order does not match the booster")
    out_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out_dir / MODEL_FILE))
    (out_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return out_dir


def load_metadata(model_dir: Path = MODEL_DIR) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((model_dir / METADATA_FILE).read_text())
    return data
