import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from src.data.dataset import create_inference_loader
from src.models.classifier import load_model
from src.utils.helpers import format_prediction


def predict_image(model_path: str, image_path: str, model_name: str = "resnet18",
                  image_size: int = 224, class_names=None, device=None,
                  preprocessing: str = "imagenet", single_channel: bool = False,
                  threshold: float = 0.5):
    """
    Predict class for a single image.

    Args:
        model_path: Path to trained model checkpoint
        image_path: Path to input image
        model_name: Model architecture
        image_size: Input image size
        class_names: List of class names
        device: Torch device
        preprocessing: Preprocessing mode ("imagenet", "dataset_stats", "sar_log")
        single_channel: Use single-channel input
        threshold: Decision threshold for OIL_SPILL class

    Returns:
        dict with prediction results
    """
    if class_names is None:
        class_names = ["NO_OIL_SPILL", "OIL_SPILL"]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(model_path, num_classes=len(class_names), model_name=model_name, device=device)

    loader = create_inference_loader(image_path, image_size=image_size, batch_size=1,
                                    preprocessing=preprocessing, single_channel=single_channel)

    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            
            # Use threshold for binary classification
            oil_prob = probs[0, 1].item()
            predicted = 1 if oil_prob >= threshold else 0
            confidence = oil_prob if predicted == 1 else (1 - oil_prob)

            pred_class = class_names[predicted]
            conf_value = confidence

            return {
                "class_name": pred_class,
                "confidence": conf_value,
                "class_index": predicted,
                "oil_probability": oil_prob,
                "threshold": threshold,
                "all_probabilities": {class_names[i]: probs[0][i].item() for i in range(len(class_names))},
            }

    return None


def main(args):
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        return

    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        return

    result = predict_image(
        model_path=args.model,
        image_path=args.image,
        model_name=args.model_name,
        image_size=args.image_size,
        preprocessing=args.preprocessing,
        single_channel=args.single_channel,
        threshold=args.threshold,
    )

    if result:
        output = format_prediction(result["class_name"], result["confidence"])
        print(output)
        print(f"Oil probability: {result['oil_probability']:.4f}")
        print(f"Threshold: {result['threshold']:.2f}")

        if args.verbose:
            print(f"\nAll probabilities:")
            for cls, prob in result["all_probabilities"].items():
                print(f"  {cls}: {prob:.4f}")

        if args.output:
            import json
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\nResult saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict oil spill class for an image")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--model", type=str, default="models/best_model.pth",
                        help="Path to trained model checkpoint")
    parser.add_argument("--model_name", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "efficientnet_b0", "mobilenet_v3_small"],
                        help="Model architecture")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Input image size")
    parser.add_argument("--preprocessing", type=str, default="imagenet",
                        choices=["imagenet", "dataset_stats", "sar_log"],
                        help="Preprocessing mode (must match training)")
    parser.add_argument("--single_channel", action="store_true",
                        help="Use single-channel input (grayscale)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold for OIL_SPILL class")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show all class probabilities")
    parser.add_argument("--output", "-o", type=str,
                        help="Save result to JSON file")

    args = parser.parse_args()
    main(args)