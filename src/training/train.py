import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

from src.data.dataset import create_data_loaders
from src.models.classifier import create_model, save_checkpoint, get_model_summary
from src.utils.helpers import set_seed, compute_class_weights


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance and hard examples."""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """CrossEntropyLoss with label smoothing."""
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        self.confidence = 1.0 - smoothing

    def forward(self, inputs, targets):
        log_probs = nn.functional.log_softmax(inputs, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        
        if self.weight is not None:
            nll_loss = nll_loss * self.weight[targets]
        
        smooth_loss = -log_probs.mean(dim=-1)
        
        if self.weight is not None:
            smooth_loss = smooth_loss * self.weight[targets]
        
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx % 20 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%")

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


def main(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")

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

    class_weights = compute_class_weights(class_info["class_distribution"]).to(device)
    print(f"Class weights: {class_weights.cpu().numpy()}")

    model = create_model(
        num_classes=len(class_info["class_names"]),
        model_name=args.model,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        dropout_rate=args.dropout,
    ).to(device)

    print(f"\nModel: {args.model}")
    print(get_model_summary(model))

    # Loss function selection
    if args.loss == "crossentropy":
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif args.loss == "focal":
        criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
    elif args.loss == "label_smoothing":
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing, weight=class_weights)
    else:
        raise ValueError(f"Unknown loss: {args.loss}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Loss function: {args.loss}")
    print(f"Preprocessing: {args.preprocessing}")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
              f"Time: {epoch_time:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                {"val_acc": val_acc, "train_acc": train_acc},
                output_dir / "best_model.pth",
                class_info["class_names"],
            )
            print(f"  -> New best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                {"val_acc": val_acc, "train_acc": train_acc},
                output_dir / f"checkpoint_epoch_{epoch}.pth",
                class_info["class_names"],
            )

        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}, Best validation accuracy: {best_val_acc:.2f}%")

    final_model_path = output_dir / "final_model.pth"
    save_checkpoint(
        model, optimizer, epoch, val_loss,
        {"val_acc": val_acc, "train_acc": train_acc, "history": history},
        final_model_path,
        class_info["class_names"],
    )

    return model, class_info, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train oil spill classifier")
    parser.add_argument("--data_dir", type=str, default="data/raw/archive/kaggle/data",
                        help="Path to dataset directory containing Class_0 and Class_1")
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Directory to save model checkpoints")
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "efficientnet_b0", "mobilenet_v3_small"],
                        help="Model architecture")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use ImageNet pretrained weights")
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze backbone layers")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate")
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
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--preprocessing", type=str, default="imagenet",
                        choices=["imagenet", "dataset_stats", "sar_log"],
                        help="Preprocessing mode")
    parser.add_argument("--single_channel", action="store_true",
                        help="Use single-channel input (grayscale)")
    parser.add_argument("--loss", type=str, default="crossentropy",
                        choices=["crossentropy", "focal", "label_smoothing"],
                        help="Loss function")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal loss gamma parameter")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing factor")

    args = parser.parse_args()
    main(args)