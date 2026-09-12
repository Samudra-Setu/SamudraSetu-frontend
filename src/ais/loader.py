"""
AIS Data Loader and Validator for Samudra Setu.

Handles loading, validation, and preprocessing of AIS (Automatic Identification System) data.
Designed to work with standard AIS CSV formats including fields:
- MMSI, IMO, VesselName, VesselType
- BaseDateTime (ISO 8601 timestamp)
- Latitude, Longitude
- SOG (Speed Over Ground), COG (Course Over Ground), Heading
- Length, Width, Draft
- Status
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import warnings
import logging
import glob

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s - %(name)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# Expected column names (flexible matching)
COLUMN_MAPPING = {
    'mmsi': ['MMSI', 'mmsi', 'Mmsi'],
    'imo': ['IMO', 'imo', 'Imo'],
    'vessel_name': ['VesselName', 'vessel_name', 'Vessel Name', 'ShipName', 'Name'],
    'vessel_type': ['VesselType', 'vessel_type', 'Vessel Type', 'ShipType', 'Type'],
    'timestamp': ['BaseDateTime', 'base_datetime', 'Timestamp', 'timestamp', 'DateTime', 'datetime'],
    'latitude': ['Latitude', 'latitude', 'Lat', 'lat', 'LAT'],
    'longitude': ['Longitude', 'longitude', 'Lon', 'lon', 'LON', 'Long', 'long'],
    'sog': ['SOG', 'sog', 'Speed', 'speed', 'SpeedOverGround'],
    'cog': ['COG', 'cog', 'Course', 'course', 'CourseOverGround'],
    'heading': ['Heading', 'heading', 'Head', 'head', 'TrueHeading'],
    'length': ['Length', 'length', 'Len', 'len', 'ShipLength'],
    'width': ['Width', 'width', 'Beam', 'beam', 'ShipWidth'],
    'draft': ['Draft', 'draft', 'Draught', 'draught', 'ShipDraft'],
    'status': ['Status', 'status', 'NavStatus', 'nav_status', 'NavigationStatus'],
}


@dataclass
class AISRecord:
    """Single AIS record with validated fields."""
    mmsi: str
    imo: Optional[str]
    vessel_name: Optional[str]
    vessel_type: Optional[str]
    timestamp: datetime
    latitude: float
    longitude: float
    sog: Optional[float]
    cog: Optional[float]
    heading: Optional[float]
    length: Optional[float]
    width: Optional[float]
    draft: Optional[float]
    status: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'mmsi': self.mmsi,
            'imo': self.imo,
            'vessel_name': self.vessel_name,
            'vessel_type': self.vessel_type,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'sog': self.sog,
            'cog': self.cog,
            'heading': self.heading,
            'length': self.length,
            'width': self.width,
            'draft': self.draft,
            'status': self.status,
        }


@dataclass
class VesselInfo:
    """Aggregated vessel information from multiple AIS records."""
    mmsi: str
    imo: Optional[str]
    vessel_name: Optional[str]
    vessel_type: Optional[str]
    records: List[AISRecord]
    first_seen: datetime
    last_seen: datetime
    total_records: int
    avg_sog: Optional[float]
    avg_cog: Optional[float]
    bounding_box: Tuple[float, float, float, float]  # min_lat, min_lon, max_lat, max_lon
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'mmsi': self.mmsi,
            'imo': self.imo,
            'vessel_name': self.vessel_name,
            'vessel_type': self.vessel_type,
            'first_seen': self.first_seen.isoformat() if self.first_seen else None,
            'last_seen': self.last_seen.isoformat() if self.last_seen else None,
            'total_records': self.total_records,
            'avg_sog': self.avg_sog,
            'avg_cog': self.avg_cog,
            'bounding_box': self.bounding_box,
        }


class AISDataValidator:
    """Validates AIS data quality and consistency."""
    
    # Valid ranges for AIS fields
    LAT_RANGE = (-90.0, 90.0)
    LON_RANGE = (-180.0, 180.0)
    SOG_MAX = 102.2  # knots (AIS max)
    COG_RANGE = (0.0, 360.0)
    HEADING_RANGE = (0.0, 360.0)  # 511 = not available
    MMSI_LENGTH = 9
    IMO_LENGTH = 7
    
    def __init__(self, strict: bool = False):
        self.strict = strict
        self.errors = []
        self.warnings = []
    
    def validate_record(self, record: AISRecord) -> bool:
        """Validate a single AIS record. Returns True if valid."""
        valid = True
        
        # MMSI validation
        if not record.mmsi or len(str(record.mmsi)) != self.MMSI_LENGTH:
            self._add_error(f"Invalid MMSI: {record.mmsi} (must be 9 digits)")
            valid = False
        
        # IMO validation (optional)
        if record.imo and len(str(record.imo)) != self.IMO_LENGTH:
            self._add_warning(f"Invalid IMO: {record.imo} (should be 7 digits)")
        
        # Latitude validation
        if not self.LAT_RANGE[0] <= record.latitude <= self.LAT_RANGE[1]:
            self._add_error(f"Invalid latitude: {record.latitude} (must be {self.LAT_RANGE})")
            valid = False
        
        # Longitude validation
        if not self.LON_RANGE[0] <= record.longitude <= self.LON_RANGE[1]:
            self._add_error(f"Invalid longitude: {record.longitude} (must be {self.LON_RANGE})")
            valid = False
        
        # SOG validation
        if record.sog is not None and (record.sog < 0 or record.sog > self.SOG_MAX):
            self._add_warning(f"Unusual SOG: {record.sog} knots (max {self.SOG_MAX})")
        
        # COG validation
        if record.cog is not None and not (self.COG_RANGE[0] <= record.cog <= self.COG_RANGE[1]):
            self._add_warning(f"Invalid COG: {record.cog} (must be {self.COG_RANGE})")
        
        # Heading validation (511 = not available per AIS spec)
        if record.heading is not None and record.heading != 511 and not (self.HEADING_RANGE[0] <= record.heading <= self.HEADING_RANGE[1]):
            self._add_warning(f"Invalid heading: {record.heading} (must be {self.HEADING_RANGE} or 511)")
        
        # Timestamp validation
        if record.timestamp is None:
            self._add_error("Missing timestamp")
            valid = False
        
        return valid
    
    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Validate entire DataFrame and return clean DataFrame with stats."""
        self.errors = []
        self.warnings = []
        
        initial_count = len(df)
        
        # Check required columns
        required_cols = ['mmsi', 'timestamp', 'latitude', 'longitude']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['mmsi', 'timestamp', 'latitude', 'longitude'])
        
        # Remove records with missing required fields
        df = df.dropna(subset=required_cols)
        
        # Validate coordinate ranges
        lat_mask = (df['latitude'] >= self.LAT_RANGE[0]) & (df['latitude'] <= self.LAT_RANGE[1])
        lon_mask = (df['longitude'] >= self.LON_RANGE[0]) & (df['longitude'] <= self.LON_RANGE[1])
        df = df[lat_mask & lon_mask]
        
        # Validate SOG
        if 'sog' in df.columns:
            df = df[(df['sog'].isna()) | ((df['sog'] >= 0) & (df['sog'] <= self.SOG_MAX))]
        
        # Validate COG
        if 'cog' in df.columns:
            df = df[(df['cog'].isna()) | ((df['cog'] >= 0) & (df['cog'] <= 360))]
        
        # Validate heading
        if 'heading' in df.columns:
            df = df[(df['heading'].isna()) | (df['heading'] == 511) | ((df['heading'] >= 0) & (df['heading'] <= 360))]
        
        # Sort by MMSI and timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values(['mmsi', 'timestamp']).reset_index(drop=True)
        
        final_count = len(df)
        
        stats = {
            'initial_records': initial_count,
            'valid_records': final_count,
            'removed_records': initial_count - final_count,
            'unique_vessels': df['mmsi'].nunique() if 'mmsi' in df.columns else 0,
            'date_range': (
                df['timestamp'].min().isoformat() if 'timestamp' in df.columns and len(df) > 0 else None,
                df['timestamp'].max().isoformat() if 'timestamp' in df.columns and len(df) > 0 else None
            ),
            'errors': self.errors,
            'warnings': self.warnings,
        }
        
        return df, stats
    
    def _add_error(self, msg: str):
        self.errors.append(msg)
        if self.strict:
            raise ValueError(msg)
    
    def _add_warning(self, msg: str):
        self.warnings.append(msg)
        warnings.warn(msg)


class AISDataLoader:
    """Loads and preprocesses AIS data from various sources."""
    
    def __init__(self, validator: Optional[AISDataValidator] = None):
        self.validator = validator or AISDataValidator(strict=False)
    
    def load_csv(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Load AIS data from CSV file with automatic column detection."""
        df = pd.read_csv(filepath, **kwargs)
        return self._standardize_columns(df)
    
    def load_multiple_csv(self, filepaths: List[str], **kwargs) -> pd.DataFrame:
        """Load and concatenate multiple CSV files."""
        dfs = [self.load_csv(fp, **kwargs) for fp in filepaths]
        return pd.concat(dfs, ignore_index=True)
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map various column name conventions to standard names."""
        df = df.copy()
        
        # Build reverse mapping
        col_map = {}
        for std_name, variants in COLUMN_MAPPING.items():
            for variant in variants:
                if variant in df.columns:
                    col_map[variant] = std_name
                    break
        
        # Rename columns
        df = df.rename(columns=col_map)
        
        # Ensure required columns exist
        for col in ['mmsi', 'timestamp', 'latitude', 'longitude']:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found. Available: {list(df.columns)}")
        
        # Convert MMSI to string
        df['mmsi'] = df['mmsi'].astype(str)
        
        # Convert IMO to string if present
        if 'imo' in df.columns:
            df['imo'] = df['imo'].astype(str)
        
        # Parse timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        
        # Convert numeric columns
        numeric_cols = ['latitude', 'longitude', 'sog', 'cog', 'heading', 'length', 'width', 'draft']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle heading = 511 (not available)
        if 'heading' in df.columns:
            df.loc[df['heading'] == 511, 'heading'] = np.nan
        
        return df
    
    def load_and_validate(self, filepath: str, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load CSV and validate in one step."""
        df = self.load_csv(filepath, **kwargs)
        return self.validator.validate_dataframe(df)
    
    def to_records(self, df: pd.DataFrame) -> List[AISRecord]:
        """Convert DataFrame to list of AISRecord objects."""
        records = []
        for _, row in df.iterrows():
            try:
                record = AISRecord(
                    mmsi=str(row['mmsi']),
                    imo=str(row['imo']) if pd.notna(row.get('imo')) else None,
                    vessel_name=row['vessel_name'] if pd.notna(row.get('vessel_name')) else None,
                    vessel_type=row['vessel_type'] if pd.notna(row.get('vessel_type')) else None,
                    timestamp=row['timestamp'] if pd.notna(row.get('timestamp')) else None,
                    latitude=float(row['latitude']),
                    longitude=float(row['longitude']),
                    sog=float(row['sog']) if pd.notna(row.get('sog')) else None,
                    cog=float(row['cog']) if pd.notna(row.get('cog')) else None,
                    heading=float(row['heading']) if pd.notna(row.get('heading')) else None,
                    length=float(row['length']) if pd.notna(row.get('length')) else None,
                    width=float(row['width']) if pd.notna(row.get('width')) else None,
                    draft=float(row['draft']) if pd.notna(row.get('draft')) else None,
                    status=row['status'] if pd.notna(row.get('status')) else None,
                )
                records.append(record)
            except Exception as e:
                self.validator._add_warning(f"Failed to parse record: {e}")
        return records
    
    def aggregate_by_vessel(self, records: List[AISRecord]) -> Dict[str, VesselInfo]:
        """Aggregate records by vessel (MMSI)."""
        vessels = {}
        
        for record in records:
            mmsi = record.mmsi
            if mmsi not in vessels:
                vessels[mmsi] = {
                    'mmsi': mmsi,
                    'imo': record.imo,
                    'vessel_name': record.vessel_name,
                    'vessel_type': record.vessel_type,
                    'records': [],
                    'timestamps': [],
                    'sogs': [],
                    'cogs': [],
                    'lats': [],
                    'lons': [],
                }
            
            v = vessels[mmsi]
            v['records'].append(record)
            v['timestamps'].append(record.timestamp)
            v['lats'].append(record.latitude)
            v['lons'].append(record.longitude)
            
            if record.sog is not None:
                v['sogs'].append(record.sog)
            if record.cog is not None:
                v['cogs'].append(record.cog)
            
            # Update vessel info with latest non-null values
            if record.imo and not v['imo']:
                v['imo'] = record.imo
            if record.vessel_name and not v['vessel_name']:
                v['vessel_name'] = record.vessel_name
            if record.vessel_type and not v['vessel_type']:
                v['vessel_type'] = record.vessel_type
        
        # Build VesselInfo objects
        result = {}
        for mmsi, v in vessels.items():
            result[mmsi] = VesselInfo(
                mmsi=mmsi,
                imo=v['imo'],
                vessel_name=v['vessel_name'],
                vessel_type=v['vessel_type'],
                records=v['records'],
                first_seen=min(v['timestamps']),
                last_seen=max(v['timestamps']),
                total_records=len(v['records']),
                avg_sog=np.mean(v['sogs']) if v['sogs'] else None,
                avg_cog=np.mean(v['cogs']) if v['cogs'] else None,
                bounding_box=(
                    min(v['lats']), min(v['lons']),
                    max(v['lats']), max(v['lons'])
                )
            )
        
        return result


def load_ais_data(
    filepath: str,
    validate: bool = True,
    strict: bool = False
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Convenience function to load and optionally validate AIS data.
    
    Args:
        filepath: Path to AIS CSV file
        validate: Whether to validate the data
        strict: Whether to raise errors on validation failures
        
    Returns:
        DataFrame and validation stats (if validate=True)
    """
    loader = AISDataLoader(AISDataValidator(strict=strict))
    if validate:
        return loader.load_and_validate(filepath)
    else:
        df = loader.load_csv(filepath)
        return df, None


def get_ais_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate summary statistics for AIS dataset."""
    if len(df) == 0:
        return {'error': 'Empty dataframe'}
    
    summary = {
        'total_records': len(df),
        'unique_vessels': df['mmsi'].nunique() if 'mmsi' in df.columns else 0,
        'unique_imos': df['imo'].nunique() if 'imo' in df.columns else 0,
        'vessel_types': df['vessel_type'].value_counts().to_dict() if 'vessel_type' in df.columns else {},
        'date_range': {
            'start': df['timestamp'].min().isoformat() if 'timestamp' in df.columns and df['timestamp'].notna().any() else None,
            'end': df['timestamp'].max().isoformat() if 'timestamp' in df.columns and df['timestamp'].notna().any() else None,
        },
        'spatial_bounds': {
            'min_lat': df['latitude'].min() if 'latitude' in df.columns else None,
            'max_lat': df['latitude'].max() if 'latitude' in df.columns else None,
            'min_lon': df['longitude'].min() if 'longitude' in df.columns else None,
            'max_lon': df['longitude'].max() if 'longitude' in df.columns else None,
        },
        'sog_stats': {
            'mean': df['sog'].mean() if 'sog' in df.columns else None,
            'std': df['sog'].std() if 'sog' in df.columns else None,
            'min': df['sog'].min() if 'sog' in df.columns else None,
            'max': df['sog'].max() if 'sog' in df.columns else None,
        },
        'missing_values': df.isnull().sum().to_dict(),
    }
    
    return summary


def load_ais_from_directory(
    directory: str,
    pattern: str = "*.csv",
    validate: bool = True,
    strict: bool = False,
    **kwargs
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Load all AIS CSV files from a directory.
    
    Args:
        directory: Path to directory containing AIS CSV files
        pattern: Glob pattern for CSV files (default: *.csv)
        validate: Whether to validate the data
        strict: Whether to raise errors on validation failures
        **kwargs: Additional arguments passed to pd.read_csv
        
    Returns:
        Combined DataFrame and validation stats (if validate=True)
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    # Find all CSV files
    csv_files = sorted(glob.glob(str(dir_path / pattern)))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory} matching pattern {pattern}")
    
    logger.info(f"Found {len(csv_files)} CSV files in {directory}")
    
    loader = AISDataLoader(AISDataValidator(strict=strict))
    all_dfs = []
    load_stats = {
        'files_found': len(csv_files),
        'files_loaded': 0,
        'files_failed': 0,
        'total_rows': 0,
        'file_details': []
    }
    
    for filepath in csv_files:
        try:
            logger.info(f"Loading {filepath}...")
            df = loader.load_csv(filepath, **kwargs)
            rows = len(df)
            load_stats['total_rows'] += rows
            load_stats['files_loaded'] += 1
            load_stats['file_details'].append({
                'file': Path(filepath).name,
                'rows': rows,
                'status': 'success'
            })
            logger.info(f"  Loaded {rows} rows")
            all_dfs.append(df)
        except Exception as e:
            load_stats['files_failed'] += 1
            load_stats['file_details'].append({
                'file': Path(filepath).name,
                'rows': 0,
                'status': 'failed',
                'error': str(e)
            })
            logger.error(f"  Failed to load {filepath}: {e}")
            if strict:
                raise
    
    if not all_dfs:
        raise ValueError("No CSV files were successfully loaded")
    
    # Concatenate all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Combined DataFrame: {len(combined_df)} total rows from {load_stats['files_loaded']} files")
    
    # Validate if requested
    if validate:
        validated_df, stats = loader.validator.validate_dataframe(combined_df)
        # Merge load stats with validation stats
        stats['load_stats'] = load_stats
        return validated_df, stats
    else:
        return combined_df, load_stats


def load_ais_data_auto(
    path: str,
    validate: bool = True,
    strict: bool = False,
    **kwargs
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Automatically load AIS data from a file or directory.
    
    Args:
        path: Path to CSV file or directory containing CSV files
        validate: Whether to validate the data
        strict: Whether to raise errors on validation failures
        **kwargs: Additional arguments passed to pd.read_csv
        
    Returns:
        DataFrame and validation stats (if validate=True)
    """
    p = Path(path)
    if p.is_file():
        return load_ais_data(path, validate=validate, strict=strict, **kwargs)
    elif p.is_dir():
        return load_ais_from_directory(path, validate=validate, strict=strict, **kwargs)
    else:
        raise FileNotFoundError(f"Path not found: {path}")


def log_ais_debug_info(df: pd.DataFrame, label: str = "AIS Data") -> Dict[str, Any]:
    """
    Log comprehensive debug information about AIS DataFrame.
    
    Args:
        df: AIS DataFrame
        label: Label for logging
        
    Returns:
        Dictionary with debug information
    """
    if len(df) == 0:
        logger.warning(f"{label}: EMPTY DATAFRAME")
        return {'error': 'Empty dataframe', 'label': label}
    
    info = {
        'label': label,
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'unique_vessels': df['mmsi'].nunique() if 'mmsi' in df.columns else 0,
        'unique_imos': df['imo'].nunique() if 'imo' in df.columns else 0,
        'vessel_types': df['vessel_type'].value_counts().to_dict() if 'vessel_type' in df.columns else {},
    }
    
    # Required columns check
    required = ['mmsi', 'timestamp', 'latitude', 'longitude']
    for col in required:
        if col in df.columns:
            info[f'{col}_non_null'] = df[col].notna().sum()
            info[f'{col}_null'] = df[col].isna().sum()
        else:
            info[f'{col}_missing'] = True
    
    # Timestamp info
    if 'timestamp' in df.columns and df['timestamp'].notna().any():
        info['timestamp_min'] = df['timestamp'].min().isoformat()
        info['timestamp_max'] = df['timestamp'].max().isoformat()
        info['timestamp_range_hours'] = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
    
    # Spatial bounds
    if 'latitude' in df.columns and 'longitude' in df.columns:
        info['lat_min'] = float(df['latitude'].min())
        info['lat_max'] = float(df['latitude'].max())
        info['lon_min'] = float(df['longitude'].min())
        info['lon_max'] = float(df['longitude'].max())
    
    # Detected column mapping
    detected_cols = {}
    for std_name, variants in COLUMN_MAPPING.items():
        for variant in variants:
            if variant in df.columns:
                detected_cols[std_name] = variant
                break
    info['detected_columns'] = detected_cols
    
    # Log summary
    logger.info(f"=== {label} Debug Info ===")
    logger.info(f"  Rows: {info['total_rows']}, Columns: {info['total_columns']}")
    logger.info(f"  Unique vessels: {info['unique_vessels']}")
    logger.info(f"  Vessel types: {info['vessel_types']}")
    logger.info(f"  Detected columns: {detected_cols}")
    if 'timestamp_min' in info:
        logger.info(f"  Time range: {info['timestamp_min']} to {info['timestamp_max']} ({info['timestamp_range_hours']:.1f} hours)")
    if 'lat_min' in info:
        logger.info(f"  Spatial bounds: lat[{info['lat_min']:.4f}, {info['lat_max']:.4f}], lon[{info['lon_min']:.4f}, {info['lon_max']:.4f}]")
    logger.info(f"  Null counts: mmsi={info.get('mmsi_null', 'N/A')}, timestamp={info.get('timestamp_null', 'N/A')}, lat={info.get('latitude_null', 'N/A')}, lon={info.get('longitude_null', 'N/A')}")
    
    return info