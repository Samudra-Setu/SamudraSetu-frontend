"""
AIS Vessel Proximity Query for Samudra Setu.

Finds vessels within a configurable radius of an oil spill location.
Returns vessel information with distance, bearing, and temporal proximity.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
from math import radians, degrees, sin, cos, sqrt, atan2, asin

from src.ais.loader import AISRecord, AISDataLoader, load_ais_data


@dataclass
class ProximityResult:
    """Result of a vessel proximity query."""
    mmsi: str
    imo: Optional[str]
    vessel_name: Optional[str]
    vessel_type: Optional[str]
    distance_km: float
    bearing_deg: float  # Bearing from spill to vessel (0=N, 90=E, etc.)
    timestamp: datetime
    time_diff_hours: float  # Hours between spill time and vessel record
    sog: Optional[float]
    cog: Optional[float]
    heading: Optional[float]
    latitude: float
    longitude: float
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
            'distance_km': round(self.distance_km, 3),
            'bearing_deg': round(self.bearing_deg, 1),
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'time_diff_hours': round(self.time_diff_hours, 2),
            'sog': self.sog,
            'cog': self.cog,
            'heading': self.heading,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'length': self.length,
            'width': self.width,
            'draft': self.draft,
            'status': self.status,
        }


@dataclass
class SpillVesselProximity:
    """Complete proximity analysis for one spill location."""
    spill_lat: float
    spill_lon: float
    spill_time: Optional[datetime]
    search_radius_km: float
    time_window_hours: Optional[float]
    vessels_found: List[ProximityResult]
    total_vessels_in_dataset: int
    query_time: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'spill_location': {'lat': self.spill_lat, 'lon': self.spill_lon},
            'spill_time': self.spill_time.isoformat() if self.spill_time else None,
            'search_radius_km': self.search_radius_km,
            'time_window_hours': self.time_window_hours,
            'vessels_found': len(self.vessels_found),
            'total_vessels_in_dataset': self.total_vessels_in_dataset,
            'query_time': self.query_time.isoformat(),
            'vessels': [v.to_dict() for v in self.vessels_found],
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of nearby vessels."""
        if not self.vessels_found:
            return {
                'vessels_found': 0,
                'vessel_types': {},
                'min_distance_km': None,
                'max_distance_km': None,
                'avg_distance_km': None,
            }
        
        vessel_types = {}
        distances = [v.distance_km for v in self.vessels_found]
        time_diffs = [v.time_diff_hours for v in self.vessels_found]
        
        for v in self.vessels_found:
            vt = v.vessel_type or 'Unknown'
            vessel_types[vt] = vessel_types.get(vt, 0) + 1
        
        return {
            'vessels_found': len(self.vessels_found),
            'vessel_types': vessel_types,
            'min_distance_km': round(min(distances), 3),
            'max_distance_km': round(max(distances), 3),
            'avg_distance_km': round(np.mean(distances), 3),
            'min_time_diff_hours': round(min(time_diffs), 2),
            'max_time_diff_hours': round(max(time_diffs), 2),
            'avg_time_diff_hours': round(np.mean(time_diffs), 2),
        }


# Earth radius in kilometers
EARTH_RADIUS_KM = 6371.0


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance between two points using Haversine formula.
    
    Args:
        lat1, lon1: First point coordinates (degrees)
        lat2, lon2: Second point coordinates (degrees)
        
    Returns:
        Distance in kilometers
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    return EARTH_RADIUS_KM * c


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate initial bearing from point 1 to point 2.
    
    Returns bearing in degrees (0=N, 90=E, 180=S, 270=W).
    """
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlon = lon2 - lon1
    
    y = sin(dlon) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    
    bearing = atan2(y, x)
    bearing = degrees(bearing)
    bearing = (bearing + 360) % 360
    
    return bearing


def find_vessels_near_spill(
    ais_data: pd.DataFrame,
    spill_lat: float,
    spill_lon: float,
    spill_time: Optional[datetime] = None,
    radius_km: float = 50.0,
    time_window_hours: Optional[float] = None,
    min_sog: Optional[float] = None,
    vessel_types: Optional[List[str]] = None,
    max_results: Optional[int] = None,
) -> SpillVesselProximity:
    """
    Find vessels within radius of a spill location.
    
    Args:
        ais_data: DataFrame with AIS records (must have mmsi, latitude, longitude, timestamp)
        spill_lat: Spill latitude (degrees)
        spill_lon: Spill longitude (degrees)
        spill_time: Spill timestamp for temporal filtering (optional)
        radius_km: Search radius in kilometers (default 50)
        time_window_hours: Only include vessels within this many hours of spill_time (optional)
        min_sog: Minimum speed over ground to include (optional)
        vessel_types: List of vessel types to include (optional)
        max_results: Maximum number of results to return (optional)
        
    Returns:
        SpillVesselProximity with all matching vessels
    """
    query_time = datetime.now(timezone.utc)
    
    if ais_data.empty:
        return SpillVesselProximity(
            spill_lat=spill_lat,
            spill_lon=spill_lon,
            spill_time=spill_time,
            search_radius_km=radius_km,
            time_window_hours=time_window_hours,
            vessels_found=[],
            total_vessels_in_dataset=0,
            query_time=query_time,
        )
    
    # Ensure required columns
    required = ['mmsi', 'latitude', 'longitude', 'timestamp']
    for col in required:
        if col not in ais_data.columns:
            raise ValueError(f"Required column '{col}' not in AIS data")
    
    # Make a copy to avoid modifying original
    df = ais_data.copy()
    
    # Ensure timestamp is timezone-aware
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    
    if spill_time is not None and spill_time.tzinfo is None:
        spill_time = spill_time.replace(tzinfo=timezone.utc)
    
    # Calculate distances
    df['distance_km'] = df.apply(
        lambda row: haversine_distance(spill_lat, spill_lon, row['latitude'], row['longitude']),
        axis=1
    )
    
    # Filter by radius
    df = df[df['distance_km'] <= radius_km]
    
    # Filter by time window
    if time_window_hours is not None and spill_time is not None:
        time_diff = (df['timestamp'] - spill_time).dt.total_seconds() / 3600
        df = df[time_diff.abs() <= time_window_hours]
    
    # Filter by minimum SOG
    if min_sog is not None and 'sog' in df.columns:
        df = df[(df['sog'].isna()) | (df['sog'] >= min_sog)]
    
    # Filter by vessel type
    if vessel_types is not None and 'vessel_type' in df.columns:
        df = df[df['vessel_type'].isin(vessel_types)]
    
    # Calculate bearing from spill to vessel
    df['bearing_deg'] = df.apply(
        lambda row: calculate_bearing(spill_lat, spill_lon, row['latitude'], row['longitude']),
        axis=1
    )
    
    # Calculate time difference from spill
    if spill_time is not None:
        df['time_diff_hours'] = (df['timestamp'] - spill_time).dt.total_seconds() / 3600
    else:
        df['time_diff_hours'] = 0.0
    
    # Sort by distance
    df = df.sort_values('distance_km')
    
    # Limit results
    if max_results is not None:
        df = df.head(max_results)
    
    # Build results
    vessels_found = []
    for _, row in df.iterrows():
        result = ProximityResult(
            mmsi=str(row['mmsi']),
            imo=str(row['imo']) if pd.notna(row.get('imo')) else None,
            vessel_name=row['vessel_name'] if pd.notna(row.get('vessel_name')) else None,
            vessel_type=row['vessel_type'] if pd.notna(row.get('vessel_type')) else None,
            distance_km=row['distance_km'],
            bearing_deg=row['bearing_deg'],
            timestamp=row['timestamp'].to_pydatetime() if pd.notna(row['timestamp']) else None,
            time_diff_hours=row['time_diff_hours'],
            sog=row['sog'] if pd.notna(row.get('sog')) else None,
            cog=row['cog'] if pd.notna(row.get('cog')) else None,
            heading=row['heading'] if pd.notna(row.get('heading')) else None,
            latitude=row['latitude'],
            longitude=row['longitude'],
            length=row['length'] if pd.notna(row.get('length')) else None,
            width=row['width'] if pd.notna(row.get('width')) else None,
            draft=row['draft'] if pd.notna(row.get('draft')) else None,
            status=row['status'] if pd.notna(row.get('status')) else None,
        )
        vessels_found.append(result)
    
    # Count unique vessels in original dataset
    total_vessels = ais_data['mmsi'].nunique() if 'mmsi' in ais_data.columns else 0
    
    return SpillVesselProximity(
        spill_lat=spill_lat,
        spill_lon=spill_lon,
        spill_time=spill_time,
        search_radius_km=radius_km,
        time_window_hours=time_window_hours,
        vessels_found=vessels_found,
        total_vessels_in_dataset=total_vessels,
        query_time=query_time,
    )


def find_vessels_near_spills_batch(
    ais_data: pd.DataFrame,
    spill_locations: List[Tuple[float, float, Optional[datetime]]],
    radius_km: float = 50.0,
    time_window_hours: Optional[float] = None,
    **kwargs
) -> List[SpillVesselProximity]:
    """
    Find vessels near multiple spill locations.
    
    Args:
        ais_data: AIS DataFrame
        spill_locations: List of (lat, lon, timestamp) tuples
        radius_km: Search radius
        time_window_hours: Temporal window
        **kwargs: Additional args passed to find_vessels_near_spill
        
    Returns:
        List of SpillVesselProximity results
    """
    results = []
    for lat, lon, timestamp in spill_locations:
        result = find_vessels_near_spill(
            ais_data=ais_data,
            spill_lat=lat,
            spill_lon=lon,
            spill_time=timestamp,
            radius_km=radius_km,
            time_window_hours=time_window_hours,
            **kwargs
        )
        results.append(result)
    return results


def get_vessel_trajectory(
    ais_data: pd.DataFrame,
    mmsi: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Get trajectory (time-ordered positions) for a specific vessel.
    
    Args:
        ais_data: AIS DataFrame
        mmsi: Vessel MMSI
        start_time: Start of time range (optional)
        end_time: End of time range (optional)
        
    Returns:
        DataFrame with vessel's positions sorted by time
    """
    vessel_data = ais_data[ais_data['mmsi'] == str(mmsi)].copy()
    
    if vessel_data.empty:
        return pd.DataFrame()
    
    if start_time is not None:
        vessel_data = vessel_data[vessel_data['timestamp'] >= start_time]
    if end_time is not None:
        vessel_data = vessel_data[vessel_data['timestamp'] <= end_time]
    
    return vessel_data.sort_values('timestamp').reset_index(drop=True)


def calculate_closest_approach(
    ais_data: pd.DataFrame,
    spill_lat: float,
    spill_lon: float,
    spill_time: datetime,
    mmsi: str,
    time_window_hours: float = 24.0
) -> Optional[ProximityResult]:
    """
    Find the closest approach of a specific vessel to a spill location
    within a time window.
    
    Returns the record with minimum distance, or None if vessel not found.
    """
    vessel_data = get_vessel_trajectory(
        ais_data, mmsi,
        start_time=spill_time - timedelta(hours=time_window_hours),
        end_time=spill_time + timedelta(hours=time_window_hours)
    )
    
    if vessel_data.empty:
        return None
    
    # Calculate distances
    vessel_data['distance_km'] = vessel_data.apply(
        lambda row: haversine_distance(spill_lat, spill_lon, row['latitude'], row['longitude']),
        axis=1
    )
    
    # Find closest
    closest = vessel_data.loc[vessel_data['distance_km'].idxmin()]
    
    return ProximityResult(
        mmsi=str(closest['mmsi']),
        imo=str(closest['imo']) if pd.notna(closest.get('imo')) else None,
        vessel_name=closest['vessel_name'] if pd.notna(closest.get('vessel_name')) else None,
        vessel_type=closest['vessel_type'] if pd.notna(closest.get('vessel_type')) else None,
        distance_km=closest['distance_km'],
        bearing_deg=calculate_bearing(spill_lat, spill_lon, closest['latitude'], closest['longitude']),
        timestamp=closest['timestamp'].to_pydatetime() if pd.notna(closest['timestamp']) else None,
        time_diff_hours=(closest['timestamp'] - spill_time).total_seconds() / 3600,
        sog=closest['sog'] if pd.notna(closest.get('sog')) else None,
        cog=closest['cog'] if pd.notna(closest.get('cog')) else None,
        heading=closest['heading'] if pd.notna(closest.get('heading')) else None,
        latitude=closest['latitude'],
        longitude=closest['longitude'],
        length=closest['length'] if pd.notna(closest.get('length')) else None,
        width=closest['width'] if pd.notna(closest.get('width')) else None,
        draft=closest['draft'] if pd.notna(closest.get('draft')) else None,
        status=closest['status'] if pd.notna(closest.get('status')) else None,
    )


def format_proximity_report(proximity: SpillVesselProximity) -> str:
    """Format proximity result as human-readable report."""
    lines = [
        "=" * 70,
        "SAMUDRA SETU - AIS VESSEL PROXIMITY ANALYSIS",
        "=" * 70,
        f"Spill Location:     ({proximity.spill_lat:.6f}, {proximity.spill_lon:.6f})",
        f"Spill Time:         {proximity.spill_time.isoformat() if proximity.spill_time else 'N/A'}",
        f"Search Radius:      {proximity.search_radius_km} km",
        f"Time Window:        {proximity.time_window_hours} hours" if proximity.time_window_hours else "Time Window:        No temporal filter",
        f"Vessels in Dataset: {proximity.total_vessels_in_dataset}",
        f"Vessels Found:      {len(proximity.vessels_found)}",
        "",
    ]
    
    if not proximity.vessels_found:
        lines.append("No vessels found within search criteria.")
    else:
        # Sort by distance
        sorted_vessels = sorted(proximity.vessels_found, key=lambda v: v.distance_km)
        
        lines.append(f"{'#':>3} {'MMSI':>12} {'Name':<20} {'Type':<12} {'Dist(km)':>8} {'Bearing':>7} {'TimeDiff(h)':>10} {'SOG':>6} {'COG':>6}")
        lines.append("-" * 110)
        
        for i, v in enumerate(sorted_vessels, 1):
            name = (v.vessel_name or 'N/A')[:18]
            vtype = (v.vessel_type or 'N/A')[:10]
            sog = f"{v.sog:.1f}" if v.sog is not None else "N/A"
            cog = f"{v.cog:.0f}" if v.cog is not None else "N/A"
            time_diff = f"{v.time_diff_hours:+.2f}" if v.time_diff_hours is not None else "N/A"
            
            lines.append(
                f"{i:>3} {v.mmsi:>12} {name:<20} {vtype:<12} "
                f"{v.distance_km:>8.3f} {v.bearing_deg:>7.1f} "
                f"{time_diff:>10} {sog:>6} {cog:>6}"
            )
    
    lines.append("=" * 70)
    return "\n".join(lines)