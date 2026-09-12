"""
Geospatial pipeline for Samudra Setu.

Pipeline: segmentation mask -> spill pixels -> centroid -> latitude/longitude -> estimated area

This module converts binary segmentation masks into geospatial features:
- Centroid (pixel coordinates)
- Centroid (geographic coordinates - lat/lon)  
- Area estimation (square kilometers)
- Bounding box
- Spill shape properties (compactness, elongation, etc.)

NOTE: The current CSIRO dataset does NOT contain geospatial metadata (geotransform, CRS, etc.)
for individual image chips. This pipeline requires Sentinel-1 GRD/SLC products with proper
georeferencing. The functions here are designed to work with real Sentinel-1 data when available.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from scipy import ndimage
from skimage.measure import regionprops, label
import math


@dataclass
class SpillProperties:
    """Geospatial properties of a detected oil spill."""
    # Pixel coordinates
    centroid_px: Tuple[float, float]          # (row, col) in image coordinates
    bbox_px: Tuple[int, int, int, int]        # (min_row, min_col, max_row, max_col)
    area_px: int                              # Area in pixels
    perimeter_px: float                       # Perimeter in pixels
    
    # Shape descriptors
    eccentricity: float                       # 0=circle, 1=line
    solidity: float                           # Area / convex hull area
    extent: float                             # Area / bounding box area
    major_axis_length: float                  # Length of major axis (px)
    minor_axis_length: float                  # Length of minor axis (px)
    orientation: float                        # Angle of major axis (radians)
    
    # Geographic coordinates (when geotransform available)
    centroid_lat: Optional[float] = None      # Latitude (WGS84)
    centroid_lon: Optional[float] = None      # Longitude (WGS84)
    bbox_latlon: Optional[Tuple[float, float, float, float]] = None  # (min_lat, min_lon, max_lat, max_lon)
    area_sqkm: Optional[float] = None         # Area in square kilometers
    
    # Metadata
    image_shape: Tuple[int, int] = (0, 0)     # (H, W)
    pixel_size_m: Optional[float] = None      # Ground sampling distance (meters/pixel)
    crs: Optional[str] = None                 # Coordinate reference system


@dataclass
class GeospatialResult:
    """Complete geospatial analysis result for one or more spills."""
    spills: List[SpillProperties]
    image_metadata: Dict[str, Any]
    total_spill_area_px: int
    total_spill_area_sqkm: Optional[float]
    spill_count: int


def extract_spill_properties(
    mask: np.ndarray,
    pixel_size_m: Optional[float] = None,
    geotransform: Optional[Tuple[float, ...]] = None,
    crs: Optional[str] = None,
    min_area_px: int = 10,
    require_georeferencing: bool = False
) -> List[SpillProperties]:
    """
    Extract geospatial properties from a binary segmentation mask.
    
    Args:
        mask: Binary mask (H, W), 1 = oil spill, 0 = background
        pixel_size_m: Ground sampling distance in meters/pixel (e.g., 10m for Sentinel-1 IW GRD)
        geotransform: GDAL-style geotransform (6-tuple) for pixel->geo conversion
        crs: Coordinate reference system string (e.g., "EPSG:4326")
        min_area_px: Minimum area in pixels to consider as valid spill
        require_georeferencing: If True, raise error when geotransform is missing
        
    Returns:
        List of SpillProperties for each connected component
        
    Raises:
        ValueError: If require_georeferencing=True and geotransform is None
    """
    # Validate georeferencing requirement
    if require_georeferencing and geotransform is None:
        raise ValueError(
            "Georeferencing required but no geotransform provided. "
            "Provide a geotransform from a properly georeferenced Sentinel-1 GeoTIFF, "
            "or use bounds to create one. The current image has no CRS/geotransform."
        )
    
    # Label connected components
    labeled = label(mask, connectivity=2)
    regions = regionprops(labeled)
    
    spills = []
    for region in regions:
        if region.area < min_area_px:
            continue
        
        # Pixel coordinates (row, col) = (y, x)
        centroid_px = region.centroid
        bbox_px = region.bbox  # (min_row, min_col, max_row, max_col)
        area_px = region.area
        perimeter_px = region.perimeter
        
        # Shape descriptors
        eccentricity = region.eccentricity
        solidity = region.solidity
        extent = region.extent
        major_axis = region.major_axis_length
        minor_axis = region.minor_axis_length
        orientation = region.orientation
        
        spill = SpillProperties(
            centroid_px=centroid_px,
            bbox_px=bbox_px,
            area_px=area_px,
            perimeter_px=perimeter_px,
            eccentricity=eccentricity,
            solidity=solidity,
            extent=extent,
            major_axis_length=major_axis,
            minor_axis_length=minor_axis,
            orientation=orientation,
            image_shape=mask.shape,
            pixel_size_m=pixel_size_m,
            crs=crs
        )
        
        # Convert to geographic coordinates if geotransform provided
        if geotransform is not None and pixel_size_m is not None:
            spill = _pixel_to_geo(spill, geotransform)
        elif pixel_size_m is not None:
            # Calculate area in sqkm even without geotransform
            spill.area_sqkm = estimate_spill_area_sqkm(spill.area_px, pixel_size_m)
        
        spills.append(spill)
    
    return spills


def _pixel_to_geo(spill: SpillProperties, geotransform: Tuple[float, ...]) -> SpillProperties:
    """
    Convert pixel coordinates to geographic coordinates using GDAL geotransform.
    
    Geotransform: (ulx, xres, xskew, uly, yskew, yres)
    - ulx, uly: Upper-left corner coordinates (longitude, latitude)
    - xres: Pixel width (degrees or meters)
    - yres: Pixel height (negative for north-up images)
    - xskew, yskew: Rotation (usually 0 for north-up)
    """
    ulx, xres, xskew, uly, yskew, yres = geotransform
    
    # Centroid: (row, col) -> (y, x)
    row, col = spill.centroid_px
    
    # Apply affine transformation
    lon = ulx + col * xres + row * xskew
    lat = uly + col * yskew + row * yres
    
    spill.centroid_lat = lat
    spill.centroid_lon = lon
    spill.crs = "EPSG:4326"  # Assuming WGS84
    
    # Bounding box corners
    min_row, min_col, max_row, max_col = spill.bbox_px
    
    # Four corners of bbox
    corners = [
        (min_row, min_col),  # top-left
        (min_row, max_col),  # top-right
        (max_row, min_col),  # bottom-left
        (max_row, max_col),  # bottom-right
    ]
    
    lons = []
    lats = []
    for r, c in corners:
        lons.append(ulx + c * xres + r * xskew)
        lats.append(uly + c * yskew + r * yres)
    
    spill.bbox_latlon = (min(lats), min(lons), max(lats), max(lons))
    
    # Area in square kilometers
    if spill.pixel_size_m is not None:
        area_sqm = spill.area_px * (spill.pixel_size_m ** 2)
        spill.area_sqkm = area_sqm / 1_000_000  # Convert to km²
    
    return spill


def analyze_spills(
    mask: np.ndarray,
    pixel_size_m: Optional[float] = None,
    geotransform: Optional[Tuple[float, ...]] = None,
    crs: Optional[str] = None,
    min_area_px: int = 10,
    image_metadata: Optional[Dict[str, Any]] = None,
    require_georeferencing: bool = False
) -> GeospatialResult:
    """
    Complete geospatial analysis of segmentation mask.
    
    Args:
        mask: Binary segmentation mask
        pixel_size_m: Ground sampling distance (meters/pixel)
        geotransform: GDAL geotransform tuple
        crs: Coordinate reference system
        min_area_px: Minimum spill size in pixels
        image_metadata: Additional image metadata
        require_georeferencing: If True, raise error when geotransform is missing
        
    Returns:
        GeospatialResult with all spill properties and summary statistics
        
    Raises:
        ValueError: If require_georeferencing=True and geotransform is None
    """
    spills = extract_spill_properties(
        mask, pixel_size_m, geotransform, crs, min_area_px, require_georeferencing
    )
    
    total_area_px = sum(s.area_px for s in spills)
    total_area_sqkm = None
    if pixel_size_m is not None:
        total_area_sqkm = sum(s.area_sqkm or 0 for s in spills)
    
    return GeospatialResult(
        spills=spills,
        image_metadata=image_metadata or {},
        total_spill_area_px=total_area_px,
        total_spill_area_sqkm=total_area_sqkm,
        spill_count=len(spills)
    )


def create_geotransform_from_bounds(
    bounds: Tuple[float, float, float, float],
    image_shape: Tuple[int, int]
) -> Tuple[float, ...]:
    """
    Create GDAL geotransform from geographic bounds and image shape.
    
    Args:
        bounds: (min_lon, min_lat, max_lon, max_lat) in WGS84
        image_shape: (height, width) in pixels
        
    Returns:
        GDAL geotransform tuple (ulx, xres, 0, uly, 0, yres)
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    h, w = image_shape
    
    # Upper-left corner
    ulx = min_lon
    uly = max_lat  # North-up: max latitude is top
    
    # Pixel resolution in degrees
    xres = (max_lon - min_lon) / w
    yres = -(max_lat - min_lat) / h  # Negative for north-up
    
    return (ulx, xres, 0.0, uly, 0.0, yres)


def estimate_spill_area_sqkm(
    area_px: int,
    pixel_size_m: float
) -> float:
    """Convert pixel area to square kilometers."""
    area_sqm = area_px * (pixel_size_m ** 2)
    return area_sqm / 1_000_000


def get_spill_summary(spills: List[SpillProperties]) -> Dict[str, Any]:
    """Generate human-readable summary of spill properties."""
    if not spills:
        return {"spill_count": 0, "message": "No spills detected"}
    
    total_area_px = sum(s.area_px for s in spills)
    total_area_sqkm = sum(s.area_sqkm or 0 for s in spills)
    
    summary = {
        "spill_count": len(spills),
        "total_area_pixels": total_area_px,
        "total_area_sqkm": round(total_area_sqkm, 4) if total_area_sqkm else None,
        "spills": []
    }
    
    for i, s in enumerate(spills):
        spill_info = {
            "id": i + 1,
            "centroid_pixel": (round(s.centroid_px[0], 1), round(s.centroid_px[1], 1)),
            "area_pixels": s.area_px,
            "area_sqkm": round(s.area_sqkm, 4) if s.area_sqkm else None,
            "centroid_latlon": (round(s.centroid_lat, 6), round(s.centroid_lon, 6)) 
                              if s.centroid_lat is not None else None,
            "bbox_pixel": s.bbox_px,
            "bbox_latlon": tuple(round(c, 6) for c in s.bbox_latlon) 
                          if s.bbox_latlon else None,
            "shape": {
                "eccentricity": round(s.eccentricity, 3),
                "solidity": round(s.solidity, 3),
                "extent": round(s.extent, 3),
                "major_axis_px": round(s.major_axis_length, 1),
                "minor_axis_px": round(s.minor_axis_length, 1),
                "orientation_deg": round(math.degrees(s.orientation), 1)
            }
        }
        summary["spills"].append(spill_info)
    
    return summary


def format_spill_report(result: GeospatialResult) -> str:
    """Format geospatial result as human-readable report."""
    lines = [
        "=" * 60,
        "SAMUDRA SETU - OIL SPILL GEOSPATIAL ANALYSIS",
        "=" * 60,
        f"Spills detected: {result.spill_count}",
        f"Total spill area: {result.total_spill_area_px} pixels",
    ]
    
    if result.total_spill_area_sqkm is not None:
        lines.append(f"Total spill area: {result.total_spill_area_sqkm:.4f} km²")
    
    lines.append("")
    
    for i, spill in enumerate(result.spills):
        lines.append(f"--- Spill #{i+1} ---")
        lines.append(f"  Centroid (px):     ({spill.centroid_px[0]:.1f}, {spill.centroid_px[1]:.1f})")
        
        if spill.centroid_lat is not None:
            lines.append(f"  Centroid (lat/lon): ({spill.centroid_lat:.6f}, {spill.centroid_lon:.6f})")
        
        lines.append(f"  Area:              {spill.area_px} px")
        if spill.area_sqkm is not None:
            lines.append(f"  Area:              {spill.area_sqkm:.4f} km²")
        
        lines.append(f"  Bbox (px):         {spill.bbox_px}")
        if spill.bbox_latlon:
            lines.append(f"  Bbox (lat/lon):    ({spill.bbox_latlon[0]:.6f}, {spill.bbox_latlon[1]:.6f}) - ({spill.bbox_latlon[2]:.6f}, {spill.bbox_latlon[3]:.6f})")
        
        lines.append(f"  Shape:")
        lines.append(f"    Eccentricity:    {spill.eccentricity:.3f}")
        lines.append(f"    Solidity:        {spill.solidity:.3f}")
        lines.append(f"    Extent:          {spill.extent:.3f}")
        lines.append(f"    Major axis:      {spill.major_axis_length:.1f} px")
        lines.append(f"    Minor axis:      {spill.minor_axis_length:.1f} px")
        lines.append(f"    Orientation:     {math.degrees(spill.orientation):.1f}°")
        lines.append("")
    
    lines.append("=" * 60)
    return "\n".join(lines)


# Convenience function for end-to-end pipeline
def segmentation_to_geospatial(
    mask: np.ndarray,
    pixel_size_m: float = 10.0,  # Sentinel-1 IW GRD default ~10m
    bounds: Optional[Tuple[float, float, float, float]] = None,
    image_metadata: Optional[Dict[str, Any]] = None,
    require_georeferencing: bool = False
) -> GeospatialResult:
    """
    Convenience function: segmentation mask -> geospatial properties.
    
    Args:
        mask: Binary segmentation mask (H, W)
        pixel_size_m: Ground sampling distance (default 10m for Sentinel-1)
        bounds: Optional (min_lon, min_lat, max_lon, max_lat) for georeferencing
        image_metadata: Additional metadata
        require_georeferencing: If True, raise error when geotransform is missing
        
    Returns:
        GeospatialResult
        
    Raises:
        ValueError: If require_georeferencing=True and bounds/geotransform not provided
    """
    geotransform = None
    if bounds is not None:
        geotransform = create_geotransform_from_bounds(bounds, mask.shape)
    
    return analyze_spills(
        mask=mask,
        pixel_size_m=pixel_size_m,
        geotransform=geotransform,
        crs="EPSG:4326",
        image_metadata=image_metadata,
        require_georeferencing=require_georeferencing
    )