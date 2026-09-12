"""
End-to-end segmentation + geospatial pipeline for Samudra Setu.

Combines:
1. Segmentation model (prototype or real)
2. Geospatial analysis (mask -> centroid -> coordinates -> area)

This is the main entry point for the segmentation stage of the pipeline.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List, Union
import numpy as np
from PIL import Image
from pathlib import Path

from src.segmentation.base import BaseSegmenter, SegmentationResult, PrototypeSegmenter, create_segmenter
from src.geospatial.pipeline import (
    GeospatialResult, SpillProperties,
    analyze_spills, format_spill_report, segmentation_to_geospatial
)
from src.data.geotiff_loader import load_geotiff_for_pipeline, GeoTIFFLoadError, check_geotiff_georeferencing


@dataclass
class SegmentationGeospatialResult:
    """Complete result from segmentation + geospatial analysis."""
    segmentation: SegmentationResult
    geospatial: GeospatialResult
    processing_time_ms: float


class SegmentationGeospatialPipeline:
    """
    End-to-end pipeline: Image -> Segmentation -> Geospatial properties.
    
    Usage:
        pipeline = SegmentationGeospatialPipeline(
            segmenter_type="prototype",
            classifier_path="models/best_model.pth",
            pixel_size_m=10.0
        )
        result = pipeline.process(image)
        print(format_spill_report(result.geospatial))
    """
    
    def __init__(
        self,
        segmenter_type: str = "prototype",
        segmenter_kwargs: Optional[Dict[str, Any]] = None,
        pixel_size_m: float = 10.0,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        min_area_px: int = 10,
        geotransform: Optional[Tuple[float, ...]] = None,
        crs: Optional[str] = "EPSG:4326"
    ):
        """
        Args:
            segmenter_type: Type of segmenter ("prototype" for now)
            segmenter_kwargs: Arguments for segmenter constructor
            pixel_size_m: Ground sampling distance in meters/pixel
            bounds: Geographic bounds (min_lon, min_lat, max_lon, max_lat) for georeferencing
            min_area_px: Minimum spill area in pixels
            geotransform: GDAL geotransform (alternative to bounds)
            crs: Coordinate reference system
        """
        self.segmenter = create_segmenter(segmenter_type, **(segmenter_kwargs or {}))
        self.pixel_size_m = pixel_size_m
        self.bounds = bounds
        self.min_area_px = min_area_px
        self.geotransform = geotransform
        self.crs = crs
        
        # Validate bounds/geotransform
        if bounds is not None and geotransform is not None:
            raise ValueError("Provide either bounds or geotransform, not both")
    
    def process(self, image: np.ndarray, image_metadata: Optional[Dict[str, Any]] = None) -> SegmentationGeospatialResult:
        """
        Run complete pipeline on a single image.
        
        Args:
            image: Input image (H, W, 3) or (H, W), range 0-255 or 0-1
            image_metadata: Optional metadata dict
            
        Returns:
            SegmentationGeospatialResult with segmentation and geospatial analysis
            
        Raises:
            ValueError: If require_georeferencing=True and no geotransform/bounds available
        """
        import time
        start_time = time.time()
        
        # Check if georeferencing is required (set by create_pipeline or process_from_file)
        require_georeferencing = getattr(self, 'require_georeferencing', False)
        
        # Step 1: Segmentation
        seg_result = self.segmenter.predict(image)
        
        # Determine geotransform to use
        geotransform = self.geotransform
        if geotransform is None and self.bounds is not None:
            from src.geospatial.pipeline import create_geotransform_from_bounds
            geotransform = create_geotransform_from_bounds(self.bounds, image.shape[:2])
        
        # Step 2: Geospatial analysis (single call with resolved geotransform)
        geo_result = analyze_spills(
            mask=seg_result.mask,
            pixel_size_m=self.pixel_size_m,
            geotransform=geotransform,
            crs=self.crs,
            min_area_px=self.min_area_px,
            image_metadata=image_metadata,
            require_georeferencing=require_georeferencing
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return SegmentationGeospatialResult(
            segmentation=seg_result,
            geospatial=geo_result,
            processing_time_ms=processing_time
        )
    
    def process_batch(
        self,
        images: List[np.ndarray],
        image_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[SegmentationGeospatialResult]:
        """Process multiple images."""
        if image_metadata is None:
            image_metadata = [None] * len(images)
        return [self.process(img, meta) for img, meta in zip(images, image_metadata)]
    
    def process_from_file(self, image_path: str, require_georeferencing: bool = True) -> SegmentationGeospatialResult:
        """Load image from file (supports GeoTIFF with georeferencing) and process.
        
        Args:
            image_path: Path to image file
            require_georeferencing: If True, raise error if GeoTIFF lacks CRS/geotransform
            
        Returns:
            SegmentationGeospatialResult
            
        Raises:
            GeoTIFFLoadError: If require_georeferencing=True and file lacks georeferencing
        """
        path = Path(image_path)
        
        # Check if it's a GeoTIFF
        if path.suffix.lower() in ('.tif', '.tiff'):
            # Use rasterio to load with geospatial metadata
            image, geotransform, crs, pixel_size_m, bounds, is_georeferenced = load_geotiff_for_pipeline(
                image_path, require_georeferencing=require_georeferencing
            )
            
            # Override pipeline settings with file metadata if available
            if geotransform is not None:
                self.geotransform = geotransform
            if crs is not None:
                self.crs = crs
            if pixel_size_m is not None:
                self.pixel_size_m = pixel_size_m
            # Only use file bounds if the file is actually georeferenced
            # Otherwise, keep the manual bounds from the sidebar/pipeline config
            if is_georeferenced and bounds is not None:
                self.bounds = bounds
        else:
            # Regular image (JPG, PNG) - no georeferencing
            image = np.array(Image.open(image_path).convert("RGB"))
            if require_georeferencing:
                raise GeoTIFFLoadError(
                    f"File {image_path} is not a GeoTIFF and has no georeferencing. "
                    f"Provide bounds manually or use a properly georeferenced Sentinel-1 GeoTIFF."
                )
        
        return self.process(image)


def create_pipeline(
    classifier_path: str = "models/best_model.pth",
    classifier_model: str = "mobilenet_v3_small",
    image_size: int = 128,
    pixel_size_m: float = 10.0,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    prototype_method: str = "sar_threshold",
    require_georeferencing: bool = True
) -> SegmentationGeospatialPipeline:
    """
    Factory function to create a prototype segmentation + geospatial pipeline.
    
    Args:
        classifier_path: Path to trained classification model
        classifier_model: Classifier architecture
        image_size: Classifier input size
        pixel_size_m: Ground sampling distance (meters/pixel)
        bounds: Geographic bounds for georeferencing
        prototype_method: "sar_threshold" or "classifier_sliding"
        require_georeferencing: If True, fail when input GeoTIFF lacks CRS/geotransform
        
    Returns:
        Configured SegmentationGeospatialPipeline
    """
    pipeline = SegmentationGeospatialPipeline(
        segmenter_type="prototype",
        segmenter_kwargs={
            "classifier_path": classifier_path,
            "classifier_model": classifier_model,
            "image_size": image_size,
            "method": prototype_method
        },
        pixel_size_m=pixel_size_m,
        bounds=bounds
    )
    pipeline.require_georeferencing = require_georeferencing
    return pipeline


# Future: Real segmentation pipeline
# def create_unet_pipeline(checkpoint_path: str, ...): ...
# def create_deeplab_pipeline(checkpoint_path: str, ...): ...