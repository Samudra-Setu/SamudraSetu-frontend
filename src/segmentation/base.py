"""
Base segmentation interface for Samudra Setu.

This module defines the abstract interface that all segmentation models must implement.
The actual trained segmentation model (U-Net, DeepLabV3+, etc.) will be plugged in here
when a segmentation dataset becomes available.

IMPORTANT: The current CSIRO dataset contains ONLY image-level labels (Class_0/Class_1),
NOT segmentation masks. This interface is designed for future use with proper
segmentation training data.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import torch
from PIL import Image


@dataclass
class SegmentationResult:
    """Result of oil spill segmentation."""
    mask: np.ndarray                    # Binary mask (H, W), 1 = oil spill, 0 = background
    confidence_map: Optional[np.ndarray] = None  # Probability map (H, W), 0-1
    class_name: str = "OIL_SPILL"       # Class label
    metadata: Optional[Dict[str, Any]] = None    # Additional info (model name, thresholds, etc.)


class BaseSegmenter(ABC):
    """
    Abstract base class for oil spill segmentation models.
    
    All segmentation models (U-Net, DeepLabV3+, SegFormer, etc.) should inherit from this
    and implement the required methods.
    """
    
    def __init__(self, model_name: str, device: Optional[torch.device] = None):
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.input_size = (256, 256)  # Default, override in subclasses
    
    @abstractmethod
    def load_model(self, checkpoint_path: str) -> None:
        """Load trained model weights from checkpoint."""
        pass
    
    @abstractmethod
    def predict(self, image: np.ndarray) -> SegmentationResult:
        """
        Run segmentation on a single image.
        
        Args:
            image: Input image as numpy array (H, W, 3) or (H, W), range 0-255 or 0-1
            
        Returns:
            SegmentationResult with mask and optional confidence map
        """
        pass
    
    @abstractmethod
    def predict_batch(self, images: List[np.ndarray]) -> List[SegmentationResult]:
        """Run segmentation on a batch of images."""
        pass
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input. Override if needed."""
        # Default: resize, normalize to [0,1], convert to tensor
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        # Resize if needed
        if image.shape[:2] != self.input_size:
            from PIL import Image
            pil_img = Image.fromarray((image * 255).astype(np.uint8))
            pil_img = pil_img.resize(self.input_size, Image.BILINEAR)
            image = np.array(pil_img) / 255.0
        
        # Convert to tensor: (H, W, C) -> (1, C, H, W)
        if image.ndim == 2:
            image = image[..., None]
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
        return tensor.to(self.device)
    
    def postprocess(self, output: torch.Tensor, original_size: Tuple[int, int]) -> SegmentationResult:
        """Postprocess model output to segmentation mask. Override if needed."""
        # Default: sigmoid + threshold at 0.5
        probs = torch.sigmoid(output).squeeze().cpu().numpy()
        mask = (probs > 0.5).astype(np.uint8)
        
        # Resize back to original size
        if mask.shape != original_size:
            from PIL import Image
            mask_pil = Image.fromarray(mask * 255)
            mask_pil = mask_pil.resize(original_size[::-1], Image.NEAREST)
            mask = (np.array(mask_pil) > 127).astype(np.uint8)
            
            probs_pil = Image.fromarray(probs)
            probs_pil = probs_pil.resize(original_size[::-1], Image.BILINEAR)
            probs = np.array(probs_pil)
        
        return SegmentationResult(
            mask=mask,
            confidence_map=probs,
            metadata={"model": self.model_name, "threshold": 0.5}
        )


class PrototypeSegmenter(BaseSegmenter):
    """
    PROTOTYPE segmenter for demonstration purposes ONLY.
    
    This is NOT a trained segmentation model. It uses the trained classifier
    to create a pseudo-segmentation by:
    1. Running classifier on sliding windows / superpixels
    2. Or using classifier activation maps (CAM/Grad-CAM)
    3. Or simple thresholding on SAR backscatter (for demo)
    
    DO NOT use for production. Real segmentation requires:
    - Pixel-level annotated training data
    - Proper segmentation architecture (U-Net, DeepLabV3+, etc.)
    - Training on segmentation loss (Dice, IoU, etc.)
    """
    
    def __init__(
        self,
        classifier_path: str,
        classifier_model: str = "mobilenet_v3_small",
        image_size: int = 128,
        device: Optional[torch.device] = None,
        method: str = "sar_threshold"
    ):
        """
        Args:
            classifier_path: Path to trained classification model
            classifier_model: Architecture name
            image_size: Input size for classifier
            device: Torch device
            method: Prototype method - "sar_threshold" (simple SAR backscatter thresholding)
        """
        super().__init__(f"PrototypeSegmenter({method})", device)
        self.classifier_path = classifier_path
        self.classifier_model = classifier_model
        self.image_size = image_size
        self.method = method
        self.classifier = None
        
        # Only load classifier for sliding window method
        if self.method == "classifier_sliding":
            self._load_classifier()
    
    def _load_classifier(self):
        """Load the trained classifier for prototype segmentation."""
        from src.models.classifier import load_model
        self.classifier = load_model(
            self.classifier_path,
            num_classes=2,
            model_name=self.classifier_model,
            device=self.device
        )
        self.classifier.eval()
    
    def load_model(self, checkpoint_path: str) -> None:
        """Load a real segmentation model checkpoint (when available)."""
        # Placeholder for future real segmentation model loading
        raise NotImplementedError(
            "Real segmentation model loading not implemented. "
            "This is a prototype segmenter. Use a trained U-Net/DeepLab checkpoint instead."
        )
    
    def predict(self, image: np.ndarray) -> SegmentationResult:
        """
        PROTOTYPE segmentation using simple SAR backscatter thresholding.
        
        This method exploits the fact that oil spills appear as DARK regions
        in SAR imagery (low backscatter). It applies adaptive thresholding
        on the grayscale SAR image.
        
        WARNING: This is a heuristic, NOT a trained segmentation model.
        Results are approximate and for demonstration only.
        """
        original_size = image.shape[:2]
        
        # Convert to grayscale if needed
        if image.ndim == 3:
            gray = np.mean(image, axis=2).astype(np.float32)
        else:
            gray = image.astype(np.float32)
        
        # Normalize to 0-1
        if gray.max() > 1.0:
            gray = gray / 255.0
        
        if self.method == "sar_threshold":
            mask, confidence = self._sar_threshold_segmentation(gray)
        elif self.method == "classifier_sliding":
            mask, confidence = self._classifier_sliding_window(image)
        else:
            raise ValueError(f"Unknown prototype method: {self.method}")
        
        return SegmentationResult(
            mask=mask,
            confidence_map=confidence,
            class_name="OIL_SPILL",
            metadata={
                "model": f"PrototypeSegmenter({self.method})",
                "warning": "PROTOTYPE - Not a trained segmentation model. "
                          "Requires pixel-level annotated data for real segmentation."
            }
        )
    
    def _sar_threshold_segmentation(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple SAR backscatter thresholding for oil spill detection.
        
        Oil appears as dark (low backscatter) regions in SAR.
        Uses adaptive thresholding (Otsu or percentile-based).
        """
        from skimage.filters import threshold_otsu
        from skimage.morphology import remove_small_objects, remove_small_holes
        
        # Invert so oil (dark) becomes bright
        inverted = 1.0 - gray
        
        # Apply Otsu threshold on inverted image
        try:
            thresh = threshold_otsu(inverted)
            mask = (inverted > thresh).astype(np.uint8)
        except:
            # Fallback: percentile-based
            thresh = np.percentile(inverted, 85)
            mask = (inverted > thresh).astype(np.uint8)
        
        # Clean up: remove small noise
        mask = remove_small_objects(mask.astype(bool), min_size=50)
        mask = remove_small_holes(mask.astype(bool), area_threshold=50)
        mask = mask.astype(np.uint8)
        
        # Confidence map: distance from threshold
        confidence = np.clip((inverted - thresh) / (1.0 - thresh + 1e-6), 0, 1)
        confidence = confidence * mask  # Only confident where mask is 1
        
        return mask, confidence
    
    def _classifier_sliding_window(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        PROTOTYPE: Sliding window classification to build pseudo-segmentation.
        
        Runs classifier on overlapping patches to create a coarse probability map.
        Very slow and low resolution - for demonstration only.
        """
        from src.data.dataset import get_transforms
        from PIL import Image
        
        h, w = image.shape[:2]
        patch_size = self.image_size
        stride = patch_size // 2
        
        prob_map = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)
        
        transform = get_transforms(self.image_size, augment=False)
        
        self.classifier.eval()
        with torch.no_grad():
            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):
                    patch = image[y:y+patch_size, x:x+patch_size]
                    pil_patch = Image.fromarray(patch.astype(np.uint8))
                    tensor = transform(pil_patch).unsqueeze(0).to(self.device)
                    
                    output = self.classifier(tensor)
                    probs = torch.softmax(output, dim=1)
                    oil_prob = probs[0, 1].item()  # Class 1 = OIL_SPILL
                    
                    prob_map[y:y+patch_size, x:x+patch_size] += oil_prob
                    count_map[y:y+patch_size, x:x+patch_size] += 1
        
        # Average overlapping predictions
        avg_prob = np.divide(prob_map, count_map, out=np.zeros_like(prob_map), where=count_map>0)
        mask = (avg_prob > 0.5).astype(np.uint8)
        
        return mask, avg_prob
    
    def predict_batch(self, images: List[np.ndarray]) -> List[SegmentationResult]:
        """Run prototype segmentation on batch."""
        return [self.predict(img) for img in images]


def create_segmenter(
    segmenter_type: str = "prototype",
    **kwargs
) -> BaseSegmenter:
    """
    Factory function to create segmenter instances.
    
    Args:
        segmenter_type: "prototype" (only option currently)
        **kwargs: Arguments passed to segmenter constructor
        
    Returns:
        BaseSegmenter instance
    """
    if segmenter_type == "prototype":
        return PrototypeSegmenter(**kwargs)
    else:
        raise ValueError(f"Unknown segmenter type: {segmenter_type}. "
                         "Only 'prototype' is available. "
                         "Real segmentation models (unet, deeplab) require training data.")


# Future: Real segmentation model implementations
# class UNetSegmenter(BaseSegmenter): ...
# class DeepLabV3Segmenter(BaseSegmenter): ...
# class SegFormerSegmenter(BaseSegmenter): ...