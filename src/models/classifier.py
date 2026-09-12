import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


def create_model(
    num_classes: int = 2,
    model_name: str = "resnet18",
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout_rate: float = 0.3,
) -> nn.Module:
    """
    Create a classification model with transfer learning.

    Args:
        num_classes: Number of output classes (2 for binary)
        model_name: Backbone architecture ('resnet18', 'resnet34', 'efficientnet_b0', 'mobilenet_v3_small')
        pretrained: Whether to use ImageNet pretrained weights
        freeze_backbone: Whether to freeze backbone layers
        dropout_rate: Dropout rate for classifier head

    Returns:
        PyTorch model
    """
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes),
        )
    elif model_name == "resnet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet34(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes),
        )
    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes),
        )
    elif model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes),
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "fc" not in name and "classifier" not in name:
                param.requires_grad = False
        print(f"Backbone frozen. Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    else:
        print(f"All params trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    return model


def load_model(
    model_path: str,
    num_classes: int = 2,
    model_name: str = "resnet18",
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Load a trained model from checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(num_classes=num_classes, model_name=model_name, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: dict,
    path: str,
    class_names: list,
) -> None:
    """Save model checkpoint."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "metrics": metrics,
        "class_names": class_names,
    }, path)
    print(f"Checkpoint saved to {path}")


def get_model_summary(model: nn.Module) -> str:
    """Get a string summary of the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total params: {total_params:,} | Trainable: {trainable_params:,}"