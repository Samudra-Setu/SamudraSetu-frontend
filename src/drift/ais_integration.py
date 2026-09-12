"""
Drift-AIS Integration for Samudra Setu.

Connects drift/backtracking results to AIS vessel proximity queries.
Implements the data flow:
  Spill observation → Backtracked origin → Candidate vessels → Distance/temporal correlation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta, timezone
import numpy as np

from src.drift.base import (
    DriftResult, DriftStep, OilSpillDriftQuery, DriftDirection,
    BaseDriftModel, haversine_distance, calculate_bearing,
)
from src.drift.prototype import PrototypeConstantCurrentModel, create_prototype_model
from src.ais.loader import AISRecord, VesselInfo
from src.ais.proximity import (
    ProximityResult, SpillVesselProximity,
    find_vessels_near_spill, find_vessels_near_spills_batch,
    get_vessel_trajectory, format_proximity_report,
)


@dataclass
class DriftAISCandidate:
    """
    A vessel candidate from drift-AIS correlation.
    
    Combines drift backtracking with AIS proximity analysis.
    """
    # Vessel identification
    mmsi: str
    imo: Optional[str]
    vessel_name: Optional[str]
    vessel_type: Optional[str]
    
    # Drift information
    backtracked_origin_lat: float
    backtracked_origin_lon: float
    backtrack_time: datetime
    
    # AIS proximity at backtracked time
    distance_to_origin_km: float
    bearing_from_origin: float
    time_diff_hours: float
    
    # Vessel state at that time
    vessel_lat: float
    vessel_lon: float
    vessel_time: datetime
    sog: Optional[float]
    cog: Optional[float]
    heading: Optional[float]
    
    # Correlation scores (0-1, placeholder for future)
    distance_score: Optional[float] = None
    temporal_score: Optional[float] = None
    trajectory_score: Optional[float] = None
    
    # Combined score (placeholder)
    combined_score: Optional[float] = None
    
    # Metadata
    drift_model: str = "unknown"
    assumptions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'mmsi': self.mmsi,
            'imo': self.imo,
            'vessel_name': self.vessel_name,
            'vessel_type': self.vessel_type,
            'backtracked_origin': {'lat': self.backtracked_origin_lat, 'lon': self.backtracked_origin_lon},
            'backtrack_time': self.backtrack_time.isoformat() if self.backtrack_time else None,
            'distance_to_origin_km': self.distance_to_origin_km,
            'bearing_from_origin': self.bearing_from_origin,
            'time_diff_hours': self.time_diff_hours,
            'vessel_location': {'lat': self.vessel_lat, 'lon': self.vessel_lon},
            'vessel_time': self.vessel_time.isoformat() if self.vessel_time else None,
            'sog': self.sog,
            'cog': self.cog,
            'heading': self.heading,
            'distance_score': self.distance_score,
            'temporal_score': self.temporal_score,
            'trajectory_score': self.trajectory_score,
            'combined_score': self.combined_score,
            'drift_model': self.drift_model,
            'assumptions': self.assumptions,
            'warnings': self.warnings,
        }


@dataclass
class DriftAISAnalysis:
    """
    Complete drift-AIS correlation analysis for one spill.
    """
    # Spill information
    spill_latitude: float
    spill_longitude: float
    spill_time: datetime
    
    # Drift results
    drift_result: DriftResult
    backtracked_origin_lat: float
    backtracked_origin_lon: float
    backtracked_time: datetime
    
    # AIS candidates
    candidates: List[DriftAISCandidate]
    
    # Search parameters
    search_radius_km: float
    time_window_hours: float
    drift_duration_hours: float
    
    # Summary
    total_candidates: int
    candidates_by_type: Dict[str, int]
    
    # Metadata
    drift_model: str
    assumptions: List[str]
    warnings: List[str]
    analysis_time: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'spill_location': {'lat': self.spill_latitude, 'lon': self.spill_longitude},
            'spill_time': self.spill_time.isoformat() if self.spill_time else None,
            'backtracked_origin': {'lat': self.backtracked_origin_lat, 'lon': self.backtracked_origin_lon},
            'backtracked_time': self.backtracked_time.isoformat() if self.backtracked_time else None,
            'drift_result': self.drift_result.to_dict(),
            'candidates': [c.to_dict() for c in self.candidates],
            'search_radius_km': self.search_radius_km,
            'time_window_hours': self.time_window_hours,
            'drift_duration_hours': self.drift_duration_hours,
            'total_candidates': self.total_candidates,
            'candidates_by_type': self.candidates_by_type,
            'drift_model': self.drift_model,
            'assumptions': self.assumptions,
            'warnings': self.warnings,
            'analysis_time': self.analysis_time.isoformat() if self.analysis_time else None,
        }
    
    def get_top_candidates(self, n: int = 5) -> List[DriftAISCandidate]:
        """Get top N candidates by distance."""
        return sorted(self.candidates, key=lambda c: c.distance_to_origin_km)[:n]
    
    def get_candidates_by_type(self, vessel_type: str) -> List[DriftAISCandidate]:
        """Filter candidates by vessel type."""
        return [c for c in self.candidates if c.vessel_type == vessel_type]


def run_drift_ais_analysis(
    ais_data,  # DataFrame
    spill_latitude: float,
    spill_longitude: float,
    spill_time: datetime,
    drift_duration_hours: float = 24.0,
    drift_time_step_hours: float = 1.0,
    search_radius_km: float = 50.0,
    time_window_hours: float = 6.0,
    drift_model: Optional[BaseDriftModel] = None,
    current_u: float = 0.1,
    current_v: float = 0.05,
    vessel_types: Optional[List[str]] = None,
    min_sog: Optional[float] = None,
) -> DriftAISAnalysis:
    """
    Run complete drift-AIS correlation analysis.
    
    Data flow:
    1. Spill observation (lat, lon, time)
    2. Backtrack drift to probable origin
    3. Search AIS for vessels near backtracked origin at that time
    4. Return correlated candidates
    
    Args:
        ais_data: AIS DataFrame from load_ais_data()
        spill_latitude: Spill observation latitude
        spill_longitude: Spill observation longitude
        spill_time: Spill detection time
        drift_duration_hours: How far back to drift
        drift_time_step_hours: Integration time step
        search_radius_km: AIS search radius around backtracked origin
        time_window_hours: AIS temporal window around backtracked time
        drift_model: Optional pre-configured drift model
        current_u: Eastward current (m/s) for prototype
        current_v: Northward current (m/s) for prototype
        vessel_types: Filter by vessel type
        min_sog: Minimum SOG filter
        
    Returns:
        DriftAISAnalysis with candidates and metadata
    """
    import time
    analysis_start = time.time()
    
    # Create drift model if not provided
    if drift_model is None:
        drift_model = PrototypeConstantCurrentModel(
            u_velocity=current_u,
            v_velocity=current_v,
        )
    
    # Step 1: Run drift backtracking
    drift_query = OilSpillDriftQuery(
        spill_latitude=spill_latitude,
        spill_longitude=spill_longitude,
        spill_time=spill_time,
        backtrack_hours=drift_duration_hours,
        time_step_hours=drift_time_step_hours,
        current_source=drift_model.current_source,
    )
    
    drift_result = drift_model.simulate(drift_query)
    
    # Step 2: Get backtracked origin (end of backward trajectory)
    backtracked_lat = drift_result.end_latitude
    backtracked_lon = drift_result.end_longitude
    backtracked_time = drift_result.end_time
    
    # Step 3: AIS proximity search at backtracked location/time
    proximity_result = find_vessels_near_spill(
        ais_data=ais_data,
        spill_lat=backtracked_lat,
        spill_lon=backtracked_lon,
        spill_time=backtracked_time,
        radius_km=search_radius_km,
        time_window_hours=time_window_hours,
        vessel_types=vessel_types,
        min_sog=min_sog,
    )
    
    # Step 4: Build candidate objects
    candidates = []
    for prox in proximity_result.vessels_found:
        candidate = DriftAISCandidate(
            mmsi=prox.mmsi,
            imo=prox.imo,
            vessel_name=prox.vessel_name,
            vessel_type=prox.vessel_type,
            backtracked_origin_lat=backtracked_lat,
            backtracked_origin_lon=backtracked_lon,
            backtrack_time=backtracked_time,
            distance_to_origin_km=prox.distance_km,
            bearing_from_origin=prox.bearing_deg,
            time_diff_hours=prox.time_diff_hours,
            vessel_lat=prox.latitude,
            vessel_lon=prox.longitude,
            vessel_time=prox.timestamp,
            sog=prox.sog,
            cog=prox.cog,
            heading=prox.heading,
            drift_model=drift_model.model_name,
            assumptions=drift_model.assumptions.copy(),
            warnings=drift_model.warnings.copy(),
        )
        candidates.append(candidate)
    
    # Summary by type
    candidates_by_type = {}
    for c in candidates:
        vt = c.vessel_type or "Unknown"
        candidates_by_type[vt] = candidates_by_type.get(vt, 0) + 1
    
    return DriftAISAnalysis(
        spill_latitude=spill_latitude,
        spill_longitude=spill_longitude,
        spill_time=spill_time,
        drift_result=drift_result,
        backtracked_origin_lat=backtracked_lat,
        backtracked_origin_lon=backtracked_lon,
        backtracked_time=backtracked_time,
        candidates=candidates,
        search_radius_km=search_radius_km,
        time_window_hours=time_window_hours,
        drift_duration_hours=drift_duration_hours,
        total_candidates=len(candidates),
        candidates_by_type=candidates_by_type,
        drift_model=drift_model.model_name,
        assumptions=drift_model.assumptions,
        warnings=drift_model.warnings,
        analysis_time=datetime.now(timezone.utc),
    )


def run_multi_spill_drift_ais(
    ais_data,
    spills: List[Tuple[float, float, datetime]],
    **kwargs
) -> List[DriftAISAnalysis]:
    """Run drift-AIS analysis for multiple spills."""
    results = []
    for lat, lon, timestamp in spills:
        result = run_drift_ais_analysis(
            ais_data=ais_data,
            spill_latitude=lat,
            spill_longitude=lon,
            spill_time=timestamp,
            **kwargs
        )
        results.append(result)
    return results


def format_drift_ais_report(analysis: DriftAISAnalysis) -> str:
    """Format drift-AIS analysis as human-readable report."""
    lines = [
        "=" * 80,
        "SAMUDRA SETU - DRIFT-AIS CORRELATION ANALYSIS",
        "=" * 80,
        f"Spill Observation:  ({analysis.spill_latitude:.6f}, {analysis.spill_longitude:.6f}) at {analysis.spill_time.isoformat()}",
        f"Backtracked Origin: ({analysis.backtracked_origin_lat:.6f}, {analysis.backtracked_origin_lon:.6f}) at {analysis.backtracked_time.isoformat()}",
        f"Drift Duration:     {analysis.drift_duration_hours} hours",
        f"Search Radius:      {analysis.search_radius_km} km",
        f"Time Window:        ±{analysis.time_window_hours} hours",
        f"Drift Model:        {analysis.drift_model}",
        "",
        "DRIFT TRAJECTORY:",
        f"  Total Distance:    {analysis.drift_result.total_distance_km:.2f} km",
        f"  Net Displacement:  {analysis.drift_result.net_displacement_km:.2f} km",
        f"  Avg Current Speed: {analysis.drift_result.avg_current_speed*100:.1f} cm/s",
        f"  Steps:             {len(analysis.drift_result.trajectory)}",
        "",
        f"CANDIDATES FOUND:   {analysis.total_candidates}",
    ]
    
    if analysis.candidates_by_type:
        lines.append("  By Type:")
        for vtype, count in analysis.candidates_by_type.items():
            lines.append(f"    {vtype}: {count}")
    
    lines.append("")
    
    if analysis.candidates:
        lines.append(f"{'#':>3} {'MMSI':>12} {'Name':<18} {'Type':<10} {'Dist(km)':>8} {'Bear':>6} {'TimeDiff(h)':>10} {'SOG':>5} {'COG':>5}")
        lines.append("-" * 100)
        
        for i, c in enumerate(sorted(analysis.candidates, key=lambda x: x.distance_to_origin_km), 1):
            name = (c.vessel_name or 'N/A')[:16]
            vtype = (c.vessel_type or 'N/A')[:8]
            sog = f"{c.sog:.1f}" if c.sog else "N/A"
            cog = f"{c.cog:.0f}" if c.cog else "N/A"
            td = f"{c.time_diff_hours:+.2f}" if c.time_diff_hours is not None else "N/A"
            
            lines.append(
                f"{i:>3} {c.mmsi:>12} {name:<18} {vtype:<10} "
                f"{c.distance_to_origin_km:>8.3f} {c.bearing_from_origin:>6.1f} "
                f"{td:>10} {sog:>5} {cog:>5}"
            )
    else:
        lines.append("  No candidates found within search criteria.")
    
    lines.append("")
    lines.append("ASSUMPTIONS:")
    for a in analysis.assumptions:
        lines.append(f"  - {a}")
    
    if analysis.warnings:
        lines.append("")
        lines.append("WARNINGS:")
        for w in analysis.warnings:
            lines.append(f"  ! {w}")
    
    lines.append("=" * 80)
    return "\n".join(lines)


def get_vessel_drift_history(
    ais_data,
    mmsi: str,
    drift_model: BaseDriftModel,
    reference_time: datetime,
    lookback_hours: float = 24.0,
    time_step_hours: float = 1.0,
) -> List[DriftStep]:
    """
    Simulate forward drift from vessel's historical positions.
    
    For each AIS position of the vessel, run forward drift to reference_time
    to see if it could reach the spill location.
    
    This is an alternative approach: instead of backtracking from spill,
    forward-drift from vessel positions.
    """
    # Get vessel trajectory
    trajectory = get_vessel_trajectory(ais_data, mmsi)
    
    if trajectory.empty:
        return []
    
    # Filter to lookback window
    start_time = reference_time - timedelta(hours=lookback_hours)
    trajectory = trajectory[
        (trajectory['timestamp'] >= start_time) & 
        (trajectory['timestamp'] <= reference_time)
    ]
    
    if trajectory.empty:
        return []
    
    # For each position, run forward drift to reference_time
    all_drift_steps = []
    
    for _, row in trajectory.iterrows():
        vessel_time = row['timestamp']
        vessel_lat = row['latitude']
        vessel_lon = row['longitude']
        
        # Time remaining to reference
        remaining_hours = (reference_time - vessel_time).total_seconds() / 3600
        
        if remaining_hours <= 0:
            continue
        
        # Forward drift query
        query = OilSpillDriftQuery(
            spill_latitude=vessel_lat,
            spill_longitude=vessel_lon,
            spill_time=vessel_time,
            backtrack_hours=remaining_hours,
            time_step_hours=time_step_hours,
        )
        
        drift_result = drift_model.simulate(query)
        
        # Check if drift reaches near spill
        for step in drift_result.trajectory:
            all_drift_steps.append(step)
    
    return all_drift_steps