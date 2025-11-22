from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

try:
    import sacrebleu
except ImportError:
    sacrebleu = None


def classification_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    """Return accuracy, macro precision/recall/F1 and confusion matrix."""
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "confusion_matrix": cm.tolist(),
    }


def bleu_score(references: List[str], hypotheses: List[str]) -> float:
    """
    Corpus BLEU using sacrebleu (if installed).
    references: list of gold replies
    hypotheses: list of generated replies
    """
    if sacrebleu is None:
        raise ImportError("Install sacrebleu to compute BLEU.")
    return float(sacrebleu.corpus_bleu(hypotheses, [references]).score)


def token_overlap(references: List[str], hypotheses: List[str]) -> float:
    """
    Very simple lexical overlap metric (for sanity checks / ablation).
    """
    scores = []
    for ref, hyp in zip(references, hypotheses):
        ref_set = set(ref.split())
        hyp_set = set(hyp.split())
        if not ref_set:
            scores.append(0.0)
        else:
            scores.append(len(ref_set & hyp_set) / len(ref_set))
    return float(np.mean(scores))
