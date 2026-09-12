"""
Scoring components for vessel attribution.

Each scoring function computes a normalized [0, 1] score for a specific
evidence dimension. All scores are deterministic and reproducible.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from datetime import datetime, timezone, timedelta
import math
import numpy as np

from src.attribution.models import (
    AttributionCandidate, ComponentScores, AttributionWeights,
    VesselTypeCategory, categorize_vessel_type,
)
from src.drift.base import haversine_distance, DriftResult, DriftStep
from src.ais.proximity import ProximityResult, SpillVesselProximity


# ============================================================================
# DISTANCE SCORING
# ============================================================================

@dataclass
class DistanceScoringConfig:
    """Configuration for distance scoring."""
    max_distance_km: float = 100.0
    decay_function: str = "exponential"
    half_distance_km: float = 25.0


def score_distance(
    candidate: AttributionCandidate,
    config: Optional[DistanceScoringConfig] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute distance score based on vessel's proximity to backtracked origin.
    
    Score: 1.0 at 0 km, decaying to 0.0 at max_distance_km.
    """
    if config is None:
        config = DistanceScoringConfig()
    
    dist = candidate.distance_to_origin_km
    
    if dist <= 0:
        score = 1.0
    elif dist >= config.max_distance_km:
        score = 0.0
    else:
        if config.decay_function == "exponential":
            score = math.exp(-dist / config.half_distance_km)
        elif config.decay_function == "linear":
            score = 1.0 - (dist / config.max_distance_km)
        elif config.decay_function == "gaussian":
            sigma = config.max_distance_km / 3.0
            score = math.exp(-0.5 * (dist / sigma) ** 2)
        else:
            raise ValueError(f"Unknown decay function: {config.decay_function}")
        
        score = max(0.0, min(1.0, score))
    
    evidence = {
        "distance_km": round(dist, 3),
        "max_distance_km": config.max_distance_km,
        "decay_function": config.decay_function,
        "score": round(score, 4),
        "interpretation": (
            "Very close to origin" if score > 0.8
            else "Close to origin" if score > 0.5
            else "Moderate distance" if score > 0.2
            else "Far from origin"
        ),
    }
    
    return score, evidence


# ============================================================================
# TEMPORAL SCORING
# ============================================================================

@dataclass
class TemporalScoringConfig:
    """Configuration for temporal scoring."""
    max_time_diff_hours: float = 48.0
    optimal_window_hours: float = 6.0
    decay_function: str = "gaussian"


def score_temporal(
    candidate: AttributionCandidate,
    config: Optional[TemporalScoringConfig] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute temporal score based on alignment between vessel observation
    and backtracked origin time.
    """
    if config is None:
        config = TemporalScoringConfig()
    
    time_diff = abs(candidate.time_diff_hours)
    
    if time_diff <= config.optimal_window_hours:
        score = 1.0
    elif time_diff >= config.max_time_diff_hours:
        score = 0.0
    else:
        if config.decay_function == "gaussian":
            sigma = (config.max_time_diff_hours - config.optimal_window_hours) / 2.0
            x = time_diff - config.optimal_window_hours
            score = math.exp(-0.5 * (x / sigma) ** 2)
        elif config.decay_function == "linear":
            score = 1.0 - ((time_diff - config.optimal_window_hours) / 
                          (config.max_time_diff_hours - config.optimal_window_hours))
        elif config.decay_function == "uniform":
            score = 1.0
        else:
            raise ValueError(f"Unknown decay function: {config.decay_function}")
        
        score = max(0.0, min(1.0, score))
    
    evidence = {
        "time_diff_hours": round(candidate.time_diff_hours, 2),
        "abs_time_diff_hours": round(time_diff, 2),
        "optimal_window_hours": config.optimal_window_hours,
        "max_time_diff_hours": config.max_time_diff_hours,
        "decay_function": config.decay_function,
        "score": round(score, 4),
        "interpretation": (
            "Strong temporal alignment" if score > 0.8
            else "Good temporal alignment" if score > 0.5
            else "Moderate temporal alignment" if score > 0.2
            else "Poor temporal alignment"
        ),
    }
    
    return score, evidence


# ============================================================================
# DRIFT/TRAJECTORY CORRELATION SCORING
# ============================================================================

@dataclass
class DriftScoringConfig:
    """Configuration for drift/trajectory correlation scoring."""
    max_trajectory_distance_km: float = 50.0
    min_trajectory_points: int = 3
    weight_spatial: float = 0.6
    weight_temporal: float = 0.4


def score_drift_correlation(
    candidate: AttributionCandidate,
    drift_result: Optional[DriftResult] = None,
    vessel_trajectory: Optional[List[Tuple[float, float, datetime]]] = None,
    config: Optional[DriftScoringConfig] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute drift/trajectory correlation score.
    """
    if config is None:
        config = DriftScoringConfig()
    
    score = 0.0
    evidence = {
        "has_drift_result": drift_result is not None,
        "has_vessel_trajectory": vessel_trajectory is not None,
        "drift_trajectory_points": len(drift_result.trajectory) if drift_result else 0,
        "vessel_trajectory_points": len(vessel_trajectory) if vessel_trajectory else 0,
    }
    
    if not drift_result or not vessel_trajectory:
        evidence["score"] = 0.0
        evidence["interpretation"] = "Insufficient data for trajectory comparison"
        return 0.0, evidence
    
    if len(vessel_trajectory) < config.min_trajectory_points:
        evidence["score"] = 0.0
        evidence["interpretation"] = f"Vessel trajectory has fewer than {config.min_trajectory_points} points"
        return 0.0, evidence
    
    drift_positions = [(step.latitude, step.longitude, step.timestamp) 
                       for step in drift_result.trajectory]
    
    spatial_matches = 0
    total_vessel_points = len(vessel_trajectory)
    
    for v_lat, v_lon, v_time in vessel_trajectory:
        best_dist = float('inf')
        best_time_diff = float('inf')
        
        for d_lat, d_lon, d_time in drift_positions:
            dist = haversine_distance(v_lat, v_lon, d_lat, d_lon)
            time_diff = abs((v_time - d_time).total_seconds() / 3600.0)
            
            if dist < best_dist:
                best_dist = dist
                best_time_diff = time_diff
        
        if best_dist <= config.max_trajectory_distance_km and best_time_diff <= 6.0:
            spatial_matches += 1
    
    spatial_score = spatial_matches / total_vessel_points if total_vessel_points > 0 else 0.0
    
    drift_start = drift_result.start_time
    drift_end = drift_result.end_time
    
    vessel_times = [t for _, _, t in vessel_trajectory]
    temporal_overlap = sum(
        1 for t in vessel_times 
        if drift_start <= t <= drift_end
    )
    temporal_score = temporal_overlap / total_vessel_points if total_vessel_points > 0 else 0.0
    
    score = (config.weight_spatial * spatial_score + 
             config.weight_temporal * temporal_score)
    score = max(0.0, min(1.0, score))
    
    evidence.update({
        "spatial_matches": spatial_matches,
        "total_vessel_points": total_vessel_points,
        "spatial_score": round(spatial_score, 4),
        "temporal_overlap": temporal_overlap,
        "temporal_score": round(temporal_score, 4),
        "score": round(score, 4),
        "interpretation": (
            "Strong trajectory correlation" if score > 0.7
            else "Moderate trajectory correlation" if score > 0.4
            else "Weak trajectory correlation" if score > 0.2
            else "No significant trajectory correlation"
        ),
    })
    
    return score, evidence


# ============================================================================
# VESSEL TYPE SCORING
# ============================================================================

@dataclass
class VesselTypeScoringConfig:
    """Configuration for vessel type scoring."""
    type_scores: dict = field(default_factory=lambda: {
        "Tanker": 1.0,
        "Cargo": 0.6,
        "Fishing": 0.3,
        "Passenger": 0.2,
        "Tug": 0.4,
        "Offshore": 0.5,
        "Unknown": 0.1,
    })
    use_category: bool = True


def score_vessel_type(
    candidate: AttributionCandidate,
    config: Optional[VesselTypeScoringConfig] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute vessel type relevance score.
    
    Note: This is a PROTOTYPE assumption. Vessel type alone does not
    indicate responsibility for a spill. Weights are configurable.
    """
    if config is None:
        config = VesselTypeScoringConfig()
    
    vessel_type = candidate.vessel_type or "Unknown"
    category = candidate.vessel_type_category
    
    if config.use_category:
        score = config.type_scores.get(category.value, 0.1)
    else:
        score = config.type_scores.get(vessel_type, 0.1)
    
    score = max(0.0, min(1.0, score))
    
    evidence = {
        "vessel_type": vessel_type,
        "vessel_type_category": category.value,
        "type_score": round(score, 4),
        "interpretation": (
            "High relevance type" if score > 0.7
            else "Moderate relevance type" if score > 0.4
            else "Low relevance type" if score > 0.1
            else "Unknown/low relevance type"
        ),
    }
    
    return score, evidence


# ============================================================================
# DATA QUALITY SCORING
# ============================================================================

@dataclass
class DataQualityConfig:
    """Configuration for data quality scoring."""
    require_mmsi: bool = True
    require_imo: bool = False
    require_name: bool = False
    require_type: bool = True
    require_position: bool = True
    require_time: bool = True
    require_sog_cog: bool = False
    min_trajectory_points: int = 3


def score_data_quality(
    candidate: AttributionCandidate,
    vessel_trajectory: Optional[List[Tuple[float, float, datetime]]] = None,
    config: Optional[DataQualityConfig] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute data quality score based on completeness of AIS information.
    """
    if config is None:
        config = DataQualityConfig()
    
    checks = []
    details = {}
    
    if config.require_mmsi:
        has_mmsi = bool(candidate.mmsi and candidate.mmsi.strip())
        checks.append(has_mmsi)
        details["has_mmsi"] = has_mmsi
    
    if config.require_imo:
        has_imo = bool(candidate.imo and candidate.imo.strip())
        checks.append(has_imo)
        details["has_imo"] = has_imo
    
    if config.require_name:
        has_name = bool(candidate.vessel_name and candidate.vessel_name.strip())
        checks.append(has_name)
        details["has_name"] = has_name
    
    if config.require_type:
        has_type = bool(candidate.vessel_type and candidate.vessel_type.strip())
        checks.append(has_type)
        details["has_type"] = has_type
    
    if config.require_position:
        has_pos = (candidate.vessel_lat != 0.0 and candidate.vessel_lon != 0.0)
        checks.append(has_pos)
        details["has_position"] = has_pos
    
    if config.require_time:
        has_time = candidate.vessel_time is not None
        checks.append(has_time)
        details["has_time"] = has_time
    
    if config.require_sog_cog:
        has_sog = candidate.sog is not None
        has_cog = candidate.cog is not None
        checks.append(has_sog)
        checks.append(has_cog)
        details["has_sog"] = has_sog
        details["has_cog"] = has_cog
    
    if vessel_trajectory:
        traj_points = len(vessel_trajectory)
        traj_ok = traj_points >= config.min_trajectory_points
        checks.append(traj_ok)
        details["trajectory_points"] = traj_points
        details["trajectory_ok"] = traj_ok
    else:
        checks.append(False)
        details["trajectory_points"] = 0
        details["trajectory_ok"] = False
    
    score = sum(checks) / len(checks) if checks else 0.0
    score = max(0.0, min(1.0, score))
    
    evidence = {
        "checks_passed": sum(checks),
        "total_checks": len(checks),
        "details": details,
        "score": round(score, 4),
        "interpretation": (
            "High data quality" if score > 0.8
            else "Good data quality" if score > 0.6
            else "Moderate data quality" if score > 0.4
            else "Poor data quality"
        ),
    }
    
    return score, evidence


# ============================================================================
# COMPOSITE SCORING
# ============================================================================

def compute_all_scores(
    candidate: AttributionCandidate,
    drift_result: Optional[DriftResult] = None,
    vessel_trajectory: Optional[List[Tuple[float, float, datetime]]] = None,
    distance_config: Optional[DistanceScoringConfig] = None,
    temporal_config: Optional[TemporalScoringConfig] = None,
    drift_config: Optional[DriftScoringConfig] = None,
    vessel_type_config: Optional[VesselTypeScoringConfig] = None,
    data_quality_config: Optional[DataQualityConfig] = None,
) -> ComponentScores:
    """
    Compute all component scores for a candidate.
    """
    distance_score, dist_evidence = score_distance(candidate, distance_config)
    temporal_score, temp_evidence = score_temporal(candidate, temporal_config)
    drift_score, drift_evidence = score_drift_correlation(
        candidate, drift_result, vessel_trajectory, drift_config
    )
    vessel_type_score, vt_evidence = score_vessel_type(candidate, vessel_type_config)
    data_quality_score, dq_evidence = score_data_quality(
        candidate, vessel_trajectory, data_quality_config
    )
    
    candidate.evidence_details = {
        "distance": dist_evidence,
        "temporal": temp_evidence,
        "drift": drift_evidence,
        "vessel_type": vt_evidence,
        "data_quality": dq_evidence,
    }
    
    parts = []
    if distance_score > 0.7:
        parts.append("strong spatial consistency")
    elif distance_score > 0.4:
        parts.append("moderate spatial consistency")
    else:
        parts.append("weak spatial consistency")
    
    if temporal_score > 0.7:
        parts.append("strong temporal alignment")
    elif temporal_score > 0.4:
        parts.append("moderate temporal alignment")
    else:
        parts.append("weak temporal alignment")
    
    if drift_score > 0.5:
        parts.append("trajectory consistent with drift")
    elif drift_score > 0.2:
        parts.append("partial trajectory correlation")
    else:
        parts.append("no trajectory correlation")
    
    candidate.evidence_summary = (
        f"Candidate {candidate.mmsi}: " + "; ".join(parts) + "."
    )
    
    return ComponentScores(
        distance_score=distance_score,
        temporal_score=temporal_score,
        drift_score=drift_score,
        vessel_type_score=vessel_type_score,
        data_quality_score=data_quality_score,
    )


def calculate_final_score(
    component_scores: ComponentScores,
    weights: AttributionWeights,
) -> float:
    """
    Calculate final weighted attribution score.
    """
    normalized_weights = weights.normalized()
    return component_scores.weighted_sum(normalized_weights)