from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import json
import yaml


@dataclass
class ModelPaths:
    detection_ckpt: str
    generation_weights: str
    input_tokenizer: str
    target_tokenizer: str


@dataclass
class Config:
    device: str = "cpu"
    model_paths: ModelPaths = None
    max_gen_len: int = 20
    flirty_threshold: float = 0.5


def _load_raw(path: str) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix in {".yml", ".yaml"}:
        with path.open() as f:
            return yaml.safe_load(f)
    elif path.suffix == ".json":
        with path.open() as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")


def load_config(path: str) -> Config:
    raw = _load_raw(path)
    mp = raw.get("model_paths", {})

    model_paths = ModelPaths(
        detection_ckpt=mp["detection_ckpt"],
        generation_weights=mp["generation_weights"],
        input_tokenizer=mp["input_tokenizer"],
        target_tokenizer=mp["target_tokenizer"],
    )

    return Config(
        device=raw.get("device", "cpu"),
        model_paths=model_paths,
        max_gen_len=raw.get("max_gen_len", 20),
        flirty_threshold=raw.get("flirty_threshold", 0.5),
    )
