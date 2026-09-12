import random
import numpy as np
import torch
from typing import Dict
import torch.nn.functional as F


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_class_weights(class_distribution: Dict[str, int]) -> torch.Tensor:
    """Compute class weights for imbalanced dataset."""
    total = sum(class_distribution.values())
    weights = []
    for class_name in ["NO_OIL_SPILL", "OIL_SPILL"]:
        count = class_distribution.get(class_name, 1)
        weights.append(total / (len(class_distribution) * count))
    return torch.tensor(weights, dtype=torch.float32)


def compute_metrics(y_true, y_pred, y_probs=None, class_names=None):
    """Compute classification metrics."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

    if class_names is None:
        class_names = ["NO_OIL_SPILL", "OIL_SPILL"]

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": acc,
        "precision_per_class": dict(zip(class_names, precision)),
        "recall_per_class": dict(zip(class_names, recall)),
        "f1_per_class": dict(zip(class_names, f1)),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }

    if y_probs is not None:
        try:
            from sklearn.metrics import roc_auc_score
            if y_probs.shape[1] == 2:
                metrics["roc_auc"] = roc_auc_score(y_true, y_probs[:, 1])
        except:
            pass

    return metrics


def print_metrics(metrics: dict) -> None:
    """Pretty print metrics."""
    print(f"\n{'='*50}")
    print(f"METRICS")
    print(f"{'='*50}")
    print(f"Accuracy:       {metrics['accuracy']:.4f}")
    print(f"Precision Macro: {metrics['precision_macro']:.4f}")
    print(f"Recall Macro:   {metrics['recall_macro']:.4f}")
    print(f"F1 Macro:       {metrics['f1_macro']:.4f}")
    print(f"Precision Wtd:  {metrics['precision_weighted']:.4f}")
    print(f"Recall Wtd:     {metrics['recall_weighted']:.4f}")
    print(f"F1 Wtd:         {metrics['f1_weighted']:.4f}")
    if "roc_auc" in metrics:
        print(f"ROC AUC:        {metrics['roc_auc']:.4f}")

    print(f"\nPer-class metrics:")
    for cls in metrics["class_names"]:
        print(f"  {cls}:")
        print(f"    Precision: {metrics['precision_per_class'][cls]:.4f}")
        print(f"    Recall:    {metrics['recall_per_class'][cls]:.4f}")
        print(f"    F1:        {metrics['f1_per_class'][cls]:.4f}")

    print(f"\nConfusion Matrix:")
    cm = metrics["confusion_matrix"]
    class_names = metrics["class_names"]
    print(f"                Predicted")
    print(f"Actual      " + "  ".join(f"{c:>12}" for c in class_names))
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>10}  " + "  ".join(f"{v:>12}" for v in row))
    print(f"{'='*50}\n")


def save_metrics(metrics: dict, path: str) -> None:
    """Save metrics to JSON file."""
    import json
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {path}")


def load_metrics(path: str) -> dict:
    """Load metrics from JSON file."""
    import json
    with open(path, "r") as f:
        return json.load(f)


def format_prediction(class_name: str, confidence: float) -> str:
    """Format prediction output."""
    return f"Prediction: {class_name}\nConfidence: {confidence:.1%}"