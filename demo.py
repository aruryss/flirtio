from pathlib import Path
import sys
import argparse
import torch

from utils.config import load_config
from utils.logger import setup_logger
from generation.generate import FlirtReplyGenerator
from flirt_detection.src.detection.model import FlirtDetectionModel, FlirtDetectionTokenizer


logger = setup_logger("demo")


def load_detection(cfg):
    project_root = Path(__file__).resolve().parent
    ckpt_path = project_root / cfg.model_paths.detection_ckpt

    device = torch.device(cfg.device)
    model = FlirtDetectionModel.load_model(str(ckpt_path), device=str(device))

    # from pathlib import Path
    import json

    best_cfg_path = project_root / "best_configuration.json"
    if best_cfg_path.exists():
        with open(best_cfg_path) as f:
            best = json.load(f)
        model_name = best.get("model_name", "distilbert-base-uncased")
        max_length = best.get("max_length", 128)
    else:
        model_name = "distilbert-base-uncased"
        max_length = 128

    tokenizer = FlirtDetectionTokenizer(model_name=model_name, max_length=max_length)

    def predict_proba(texts):
        enc = tokenizer.encode_batch(texts, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        probs = model.predict_proba(input_ids, attention_mask)
        return probs.cpu().numpy()

    return predict_proba


def interactive_demo(cfg_path: str):
    cfg = load_config(cfg_path)
    project_root = Path(__file__).resolve().parent

    logger.info("Loading detection and generation models...")
    detect_proba = load_detection(cfg)
    generator = FlirtReplyGenerator(cfg, project_root=project_root)

    print("=== FlirtIO demo ===")
    print("Type a message. 'quit' to exit.\n")

    while True:
        text = input("You: ").strip()
        if text.lower() in {"quit", "exit"}:
            break

        probs = detect_proba([text])[0]
        flirt_score = float(probs[1])   # class 1 = flirty
        print(f"(flirty prob: {flirt_score:.2f})")

        if flirt_score < cfg.flirty_threshold:
            print("Bot: Hmm, that doesn't sound like flirting 😅")
        else:
            reply = generator.generate(text, flirty=True)
            print(f"Bot: {reply}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config.yml")
    args = parser.parse_args()
    interactive_demo(args.config)
