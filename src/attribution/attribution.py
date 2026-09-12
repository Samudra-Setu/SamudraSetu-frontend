"""
Main attribution pipeline for Samudra Setu.

Orchestrates the complete vessel attribution workflow:
1. Spill observation → drift backtracking
2. AIS candidate search around backtracked origin
3. Component scoring for each candidate
3. Weighted final score and ranking
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timezone, timedelta
import pandas as pd

from src.attribution.models import (
    AttributionCandidate, AttributionResult, AttributionWeights,
    ComponentScores, VesselTypeCategory, categorize_vessel_type,
)
from src.attribution.scoring import (
    compute_all_scores,
    calculate_final_score,
    DistanceScoringConfig,
    TemporalScoringConfig,
    DriftScoringConfig,
    VesselTypeScoringConfig,
    DataQualityConfig,
)
from src.drift.ais_integration import (
    run_drift_ais_analysis,
    DriftAISAnalysis,
    DriftAISCandidate,
)
from src.ais.loader import load_ais_data, AISDataLoader
from src.ais.proximity import get_vessel_trajectory
from src.drift.base import DriftResult
from src.drift.prototype import create_prototype_model, PrototypeConstantCurrentModel


def build_attribution_candidate(
    drift_candidate: DriftAISCandidate,
    spill_lat: float,
    spill_lon: float,
    spill_time: datetime,
    backtracked_origin_lat: float,
    backtracked_origin_lon: float,
    backtracked_time: datetime,
) -> AttributionCandidate:
    """
    Convert a drift-AIS candidate to an attribution candidate.
    
    Args:
        drift_candidate: Candidate from drift-AIS analysis
        spill_lat: Spill observation latitude
        spill_lon: Spill observation longitude
        spill_time: Spill detection time
        backtracked_origin_lat: Backtracked origin latitude
        backtracked_origin_lon: Backtracked origin longitude
        backtracked_time: Backtracked origin time
        
    Returns:
        AttributionCandidate ready for scoring
    """
    return AttributionCandidate(
        mmsi=drift_candidate.mmsi,
        imo=drift_candidate.imo,
        vessel_name=drift_candidate.vessel_name,
        vessel_type=drift_candidate.vessel_type,
        vessel_type_category=categorize_vessel_type(drift_candidate.vessel_type),
        spill_latitude=spill_lat,
        spill_longitude=spill_lon,
        spill_time=spill_time,
        backtracked_origin_lat=backtracked_origin_lat,
        backtracked_origin_lon=backtracked_origin_lon,
        backtracked_time=backtracked_time,
        vessel_lat=drift_candidate.vessel_lat,
        vessel_lon=drift_candidate.vessel_lon,
        vessel_time=drift_candidate.vessel_time,
        distance_to_origin_km=drift_candidate.distance_to_origin_km,
        bearing_from_origin=drift_candidate.bearing_from_origin,
        time_diff_hours=drift_candidate.time_diff_hours,
        sog=drift_candidate.sog,
        cog=drift_candidate.cog,
        heading=drift_candidate.heading,
    )


def get_vessel_trajectory_points(
    ais_data: pd.DataFrame,
    mmsi: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> List[Tuple[float, float, datetime]]:
    """
    Extract vessel trajectory as list of (lat, lon, timestamp) tuples.
    
    Args:
        ais_data: AIS DataFrame
        mmsi: Vessel MMSI
        start_time: Optional start time filter
        end_time: Optional end time filter
        
    Returns:
        List of (latitude, longitude, timestamp) tuples sorted by time
    """
    vessel_data = ais_data[ais_data['mmsi'] == str(mmsi)].copy()
    
    if vessel_data.empty:
        return []
    
    # Ensure timestamp is timezone-aware
    if vessel_data['timestamp'].dt.tz is None:
        vessel_data['timestamp'] = vessel_data['timestamp'].dt.tz_localize('UTC')
    
    if start_time is not None:
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        vessel_data = vessel_data[vessel_data['timestamp'] >= start_time]
    
    if end_time is not None:
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)
        vessel_data = vessel_data[vessel_data['timestamp'] <= end_time]
    
    vessel_data = vessel_data.sort_values('timestamp')
    
    trajectory = []
    for _, row in vessel_data.iterrows():
        lat = row.get('latitude')
        lon = row.get('longitude')
        ts = row.get('timestamp')
        
        if pd.notna(lat) and pd.notna(lon) and pd.notna(ts):
            trajectory.append((float(lat), float(lon), ts.to_pydatetime()))
    
    return trajectory


def run_attribution_analysis(
    ais_data: pd.DataFrame,
    spill_latitude: float,
    spill_longitude: float,
    spill_time: datetime,
    drift_duration_hours: float = 24.0,
    drift_time_step_hours: float = 1.0,
    search_radius_km: float = 50.0,
    time_window_hours: float = 6.0,
    current_u: float = 0.1,
    current_v: float = 0.05,
    scoring_weights: Optional[AttributionWeights] = None,
    distance_config: Optional[DistanceScoringConfig] = None,
    temporal_config: Optional[TemporalScoringConfig] = None,
    drift_config: Optional[DriftScoringConfig] = None,
    vessel_type_config: Optional[VesselTypeScoringConfig] = None,
    data_quality_config: Optional[DataQualityConfig] = None,
    vessel_types: Optional[List[str]] = None,
    min_sog: Optional[float] = None,
    drift_model: Optional[PrototypeConstantCurrentModel] = None,
) -> AttributionResult:
    """
    Run complete vessel attribution analysis for a spill event.
    
    This is the main entry point for the attribution pipeline.
    
    Args:
        ais_data: AIS DataFrame from load_ais_data()
        spill_latitude: Spill observation latitude
        spill_longitude: Spill observation longitude
        spill_time: Spill detection time (timezone-aware)
        drift_duration_hours: How far back to drift
        drift_time_step_hours: Integration time step
        search_radius_km: AIS search radius around backtracked origin
        time_window_hours: AIS temporal window around backtracked time
        current_u: Eastward current (m/s) for prototype
        current_v: Northward current (m/s) for prototype
        scoring_weights: Weight configuration for final score
        *_config: Optional scoring component configurations
        vessel_types: Filter by vessel type
        min_sog: Minimum SOG filter
        drift_model: Optional pre-configured drift model
        
    Returns:
        AttributionResult with ranked candidates
    """
    # Default weights
    if scoring_weights is None:
        scoring_weights = AttributionWeights()
    scoring_weights = scoring_weights.normalized()
    
    # Create drift model if not provided
    if drift_model is None:
        drift_model = create_prototype_model(
            "constant",
            u_velocity=current_u,
            v_velocity=current_v,
        )
    
    # Step 1: Run drift-AIS analysis (drift backtrack + AIS search)
    drift_analysis = run_drift_ais_analysis(
        ais_data=ais_data,
        spill_latitude=spill_latitude,
        spill_longitude=spill_longitude,
        spill_time=spill_time,
        drift_duration_hours=drift_duration_hours,
        drift_time_step_hours=drift_time_step_hours,
        search_radius_km=search_radius_km,
        time_window_hours=time_window_hours,
        drift_model=drift_model,
        vessel_types=vessel_types,
        min_sog=min_sog,
    )
    
    # Step 2: Build attribution candidates from drift-AIS results
    candidates = []
    for drift_cand in drift_analysis.candidates:
        attr_candidate = build_attribution_candidate(
            drift_candidate=drift_cand,
            spill_lat=spill_latitude,
            spill_lon=spill_longitude,
            spill_time=spill_time,
            backtracked_origin_lat=drift_analysis.backtracked_origin_lat,
            backtracked_origin_lon=drift_analysis.backtracked_origin_lon,
            backtracked_time=drift_analysis.backtracked_time,
        )
        
        # Get vessel trajectory for trajectory scoring
        vessel_traj = get_vessel_trajectory_points(
            ais_data=ais_data,
            mmsi=drift_cand.mmsi,
            start_time=drift_analysis.backtracked_time - timedelta(hours=drift_duration_hours),
            end_time=spill_time,
        )
        
        # Compute all component scores
        component_scores = compute_all_scores(
            candidate=attr_candidate,
            drift_result=drift_analysis.drift_result,
            vessel_trajectory=vessel_traj,
            distance_config=distance_config,
            temporal_config=temporal_config,
            drift_config=drift_config,
            vessel_type_config=vessel_type_config,
            data_quality_config=data_quality_config,
        )
        
        attr_candidate.component_scores = component_scores
        attr_candidate.scoring_weights = scoring_weights
        attr_candidate.drift_model = drift_model.model_name
        attr_candidate.assumptions = drift_analysis.assumptions.copy()
        attr_candidate.warnings = drift_analysis.warnings.copy()
        
        # Calculate final weighted score
        attr_candidate.final_score = calculate_final_score(component_scores, scoring_weights)
        
        candidates.append(attr_candidate)
    
    # Create result
    result = AttributionResult(
        spill_latitude=spill_latitude,
        spill_longitude=spill_longitude,
        spill_time=spill_time,
        backtracked_origin_lat=drift_analysis.backtracked_origin_lat,
        backtracked_origin_lon=drift_analysis.backtracked_origin_lon,
        backtracked_time=drift_analysis.backtracked_time,
        drift_duration_hours=drift_duration_hours,
        drift_model=drift_model.model_name,
        candidates=candidates,
        search_radius_km=search_radius_km,
        time_window_hours=time_window_hours,
        scoring_weights=scoring_weights,
        assumptions=drift_analysis.assumptions.copy(),
        warnings=drift_analysis.warnings.copy(),
    )
    
    # Attribution-specific warnings
    result.warnings.extend([
        "Attribution scores are MODEL SCORES based on prototype assumptions.",
        "Scores do NOT establish legal or scientific proof of responsibility.",
        "Ocean drift model uses prototype constant current assumptions.",
        "Vessel type weights are PROTOTYPE assumptions.",
    ])
    
    return result


def format_attribution_report(result: AttributionResult, top_n: int = 10) -> str:
    """
    Format attribution result as human-readable report.
    
    Args:
        result: AttributionResult
        top_n: Number of top candidates to display
        
    Returns:
        Formatted string report
    """
    lines = [
        "=" * 80,
        "SAMUDRA SETU — VESSEL ATTRIBUTION ANALYSIS",
        "=" * 80,
        f"Spill Observation:  ({result.spill_latitude:.6f}, {result.spill_longitude:.6f}) at {result.spill_time.isoformat()}",
        f"Backtracked Origin: ({result.backtracked_origin_lat:.6f}, {result.backtracked_origin_lon:.6f}) at {result.backtracked_time.isoformat()}",
        f"Drift Duration:     {result.drift_duration_hours} hours",
        f"Drift Model:        {result.drift_model}",
        f"Search Radius:      {result.search_radius_km} km",
        f"Time Window:        ±{result.time_window_hours} hours",
        "",
        "SCORING WEIGHTS:",
        f"  Distance:         {result.scoring_weights.distance:.0%}",
        f"  Temporal:         {result.scoring_weights.temporal:.0%}",
        f"  Drift/Trajectory: {result.scoring_weights.drift:.0%}",
        f"  Vessel Type:      {result.scoring_weights.vessel_type:.0%}",
        f"  Data Quality:     {result.scoring_weights.data_quality:.0%}",
        "",
        f"CANDIDATES FOUND:   {result.total_candidates}",
    ]
    
    if result.candidates_by_type:
        lines.append("  By Type:")
        for vtype, count in result.candidates_by_type.items():
            lines.append(f"    {vtype}: {count}")
    
    lines.append("")
    
    # Table header
    lines.append(f"{'Rank':>4} {'MMSI':>12} {'Name':<18} {'Type':<10} {'Dist(km)':>8} {'dT(h)':>6} {'D_Score':>7} {'T_Score':>7} {'Dr_Score':>7} {'VT_Score':>7} {'DQ_Score':>7} {'Final':>6}")
    lines.append("-" * 120)
    
    # Display top N candidates
    display_candidates = result.get_top_n(top_n)
    for c in display_candidates:
        name = (c.vessel_name or 'N/A')[:16]
        vtype = (c.vessel_type or 'N/A')[:8]
        dist = f"{c.distance_to_origin_km:.2f}"
        tdiff = f"{c.time_diff_hours:+.2f}" if c.time_diff_hours else "N/A"
        ds = f"{c.component_scores.distance_score:.3f}"
        ts = f"{c.component_scores.temporal_score:.3f}"
        drs = f"{c.component_scores.drift_score:.3f}"
        vts = f"{c.component_scores.vessel_type_score:.3f}"
        dqs = f"{c.component_scores.data_quality_score:.3f}"
        fs = f"{c.final_score:.3f}"
        
        lines.append(
            f"{c.rank:>4} {c.mmsi:>12} {name:<18} {vtype:<10} "
            f"{dist:>8} {tdiff:>6} {ds:>7} {ts:>7} {drs:>7} {vts:>7} {dqs:>7} {fs:>6}"
        )
    
    lines.append("")
    
    # Evidence explanations for top 3
    lines.append("EVIDENCE SUMMARIES:")
    for c in result.get_top_n(3):
        lines.append(f"  Rank {c.rank} [{c.mmsi}] {c.vessel_name or 'N/A'} ({c.vessel_type or 'N/A'}):")
        lines.append(f"    {c.evidence_summary}")
    
    lines.append("")
    lines.append("ASSUMPTIONS:")
    for a in result.assumptions:
        lines.append(f"  - {a}")
    
    if result.warnings:
        lines.append("")
        lines.append("WARNINGS:")
        for w in result.warnings:
            lines.append(f"  ! {w}")
    
    lines.append("=" * 80)
    return "\n".join(lines)


def run_multi_spill_attribution(
    ais_data: pd.DataFrame,
    spills: List[Tuple[float, float, datetime]],
    **kwargs
) -> List[AttributionResult]:
    """
    Run attribution analysis for multiple spills.
    
    Args:
        ais_data: AIS DataFrame
        spills: List of (lat, lon, timestamp) tuples
        **kwargs: Additional args passed to run_attribution_analysis
        
    Returns:
        List of AttributionResult
    """
    results = []
    for lat, lon, timestamp in spills:
        result = run_attribution_analysis(
            ais_data=ais_data,
            spill_latitude=lat,
            spill_longitude=lon,
            spill_time=timestamp,
            **kwargs
        )
        results.append(result)
    return results