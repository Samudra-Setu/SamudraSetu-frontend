import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from src.data.dataset import create_data_loaders
from src.models.classifier import create_model, load_model, get_model_summary
from src.utils.helpers import set_seed, compute_metrics, print_metrics, save_metrics


def evaluate_model(model, loader, criterion, device, class_names, threshold=0.5):
    """Evaluate model on a data loader."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            
            # Use threshold for binary classification
            if probs.shape[1] == 2:
                predicted = (probs[:, 1] >= threshold).long()
            else:
                _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = running_loss / len(loader)
    metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
        class_names,
    )
    metrics["loss"] = avg_loss

    return metrics, np.array(all_preds), np.array(all_labels), np.array(all_probs)


def tune_threshold(probs, labels, class_names, thresholds=None):
    """Find optimal threshold for binary classification."""
    if thresholds is None:
        thresholds = np.arange(0.3, 0.95, 0.05)
    
    best_f1 = 0
    best_threshold = 0.5
    best_metrics = None
    
    print("\nThreshold tuning:")
    for t in thresholds:
        preds = (probs[:, 1] >= t).astype(int)
        metrics = compute_metrics(labels, preds, probs, class_names)
        f1 = metrics["f1_per_class"]["OIL_SPILL"]
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        print(f"  Thresh={t:.2f}: F1(OIL)={f1:.4f}, FP={fp}, FN={fn}")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_metrics = metrics
    
    return best_threshold, best_f1, best_metrics


def main(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"\nLoading data from {args.data_dir}...")
    train_loader, val_loader, test_loader, class_info = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        image_size=args.image_size,
        num_workers=args.num_workers,
        seed=args.seed,
        preprocessing=args.preprocessing,
        single_channel=args.single_channel,
    )

    print(f"\nLoading model from {args.model_path}...")
    model = load_model(
        args.model_path,
        num_classes=len(class_info["class_names"]),
        model_name=args.model,
        device=device,
    )
    print(get_model_summary(model))

    criterion = nn.CrossEntropyLoss()

    print(f"\nEvaluating on test set ({class_info['test_size']} samples)...")
    
    # First evaluate with default threshold to get probabilities
    metrics, preds, labels, probs = evaluate_model(
        model, test_loader, criterion, device, class_info["class_names"], threshold=0.5
    )
    
    # Tune threshold on validation set
    print("\nTuning threshold on validation set...")
    val_probs = []
    val_labels = []
    model.eval()
    with torch.no_grad():
        for images, labels_batch in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs_batch = torch.softmax(outputs, dim=1)
            val_probs.extend(probs_batch.cpu().numpy())
            val_labels.extend(labels_batch.numpy())
    
    val_probs = np.array(val_probs)
    val_labels = np.array(val_labels)
    
    best_threshold, best_f1, _ = tune_threshold(val_probs, val_labels, class_info["class_names"])
    
    # Re-evaluate test set with best threshold
    print(f"\nRe-evaluating test set with optimal threshold: {best_threshold:.2f}")
    metrics, preds, labels, probs = evaluate_model(
        model, test_loader, criterion, device, class_info["class_names"], threshold=best_threshold
    )

    print_metrics(metrics)
    print(f"Optimal threshold: {best_threshold:.2f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics with threshold info
    metrics["optimal_threshold"] = float(best_threshold)
    save_metrics(metrics, output_dir / "test_metrics.json")

    np.save(output_dir / "test_predictions.npy", preds)
    np.save(output_dir / "test_labels.npy", labels)
    np.save(output_dir / "test_probabilities.npy", probs)
    print(f"Predictions saved to {output_dir}")

    if args.save_confusion_matrix:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(8, 6))
        cm = metrics["confusion_matrix"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_info["class_names"],
                    yticklabels=class_info["class_names"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - Test Set (threshold={best_threshold:.2f})")
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
        print(f"Confusion matrix plot saved to {output_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate oil spill classifier")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/raw/archive/kaggle/data",
                        help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="results/metrics",
                        help="Directory to save evaluation results")
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "efficientnet_b0", "mobilenet_v3_small"],
                        help="Model architecture")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Input image size")
    parser.add_argument("--val_split", type=float, default=0.15,
                        help="Validation split ratio")
    parser.add_argument("--test_split", type=float, default=0.15,
                        help="Test split ratio")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of data loader workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--save_confusion_matrix", action="store_true", default=True,
                        help="Save confusion matrix plot")
    parser.add_argument("--preprocessing", type=str, default="imagenet",
                        choices=["imagenet", "dataset_stats", "sar_log"],
                        help="Preprocessing mode (must match training)")
    parser.add_argument("--single_channel", action="store_true",
                        help="Use single-channel input (grayscale)")

    args = parser.parse_args()
    main(args)