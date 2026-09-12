"""
Data models for vessel attribution scoring.

These models define the structure for attribution candidates, component scores,
and final ranked results. All scores are normalized to [0, 1] range.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
from enum import Enum


class VesselTypeCategory(Enum):
    """Vessel type categories for scoring."""
    TANKER = "Tanker"
    CARGO = "Cargo"
    FISHING = "Fishing"
    PASSENGER = "Passenger"
    TUG = "Tug"
    OFFSHORE = "Offshore"
    UNKNOWN = "Unknown"


@dataclass
class AttributionWeights:
    """
    Weights for the weighted attribution scoring system.
    
    All weights must be non-negative and will be normalized to sum to 1.0.
    Default values are PROTOTYPE assumptions - not scientifically validated.
    """
    distance: float = 0.30          # Distance score weight
    temporal: float = 0.25          # Temporal correlation score weight
    drift: float = 0.30             # Drift/trajectory correlation score weight
    vessel_type: float = 0.10       # Vessel type score weight
    data_quality: float = 0.05      # Data quality score weight

    def __post_init__(self):
        """Validate weights are non-negative."""
        for name, value in [
            ("distance", self.distance),
            ("temporal", self.temporal),
            ("drift", self.drift),
            ("vessel_type", self.vessel_type),
            ("data_quality", self.data_quality),
        ]:
            if value < 0:
                raise ValueError(f"Weight '{name}' must be non-negative, got {value}")
        
        total = self.distance + self.temporal + self.drift + self.vessel_type + self.data_quality
        if total == 0:
            raise ValueError("At least one weight must be positive")

    def normalized(self) -> "AttributionWeights":
        """Return normalized weights that sum to 1.0."""
        total = self.distance + self.temporal + self.drift + self.vessel_type + self.data_quality
        return AttributionWeights(
            distance=self.distance / total,
            temporal=self.temporal / total,
            drift=self.drift / total,
            vessel_type=self.vessel_type / total,
            data_quality=self.data_quality / total,
        )

    def as_dict(self) -> Dict[str, float]:
        return {
            "distance": self.distance,
            "temporal": self.temporal,
            "drift": self.drift,
            "vessel_type": self.vessel_type,
            "data_quality": self.data_quality,
        }


@dataclass
class ComponentScores:
    """Individual component scores for a single vessel candidate."""
    distance_score: float = 0.0           # [0, 1] - closer to origin = higher
    temporal_score: float = 0.0           # [0, 1] - better temporal alignment = higher
    drift_score: float = 0.0              # [0, 1] - better trajectory/drift match = higher
    vessel_type_score: float = 0.0        # [0, 1] - type relevance = higher
    data_quality_score: float = 0.0       # [0, 1] - better data completeness = higher

    def __post_init__(self):
        """Validate scores are in [0, 1]."""
        for name, value in [
            ("distance_score", self.distance_score),
            ("temporal_score", self.temporal_score),
            ("drift_score", self.drift_score),
            ("vessel_type_score", self.vessel_type_score),
            ("data_quality_score", self.data_quality_score),
        ]:
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"Score '{name}' must be in [0, 1], got {value}")

    def as_dict(self) -> Dict[str, float]:
        return {
            "distance_score": self.distance_score,
            "temporal_score": self.temporal_score,
            "drift_score": self.drift_score,
            "vessel_type_score": self.vessel_type_score,
            "data_quality_score": self.data_quality_score,
        }

    def weighted_sum(self, weights: AttributionWeights) -> float:
        """Calculate weighted final score."""
        return (
            self.distance_score * weights.distance +
            self.temporal_score * weights.temporal +
            self.drift_score * weights.drift +
            self.vessel_type_score * weights.vessel_type +
            self.data_quality_score * weights.data_quality
        )


@dataclass
class AttributionCandidate:
    """
    Complete attribution result for a single vessel candidate.
    Contains all component scores, final score, and supporting evidence.
    """
    # Vessel identification
    mmsi: str
    imo: Optional[str] = None
    vessel_name: Optional[str] = None
    vessel_type: Optional[str] = None
    vessel_type_category: VesselTypeCategory = VesselTypeCategory.UNKNOWN

    # Spill and drift context
    spill_latitude: float = 0.0
    spill_longitude: float = 0.0
    spill_time: Optional[datetime] = None
    backtracked_origin_lat: float = 0.0
    backtracked_origin_lon: float = 0.0
    backtracked_time: Optional[datetime] = None

    # Vessel observation at closest approach
    vessel_lat: float = 0.0
    vessel_lon: float = 0.0
    vessel_time: Optional[datetime] = None
    distance_to_origin_km: float = 0.0
    bearing_from_origin: float = 0.0
    time_diff_hours: float = 0.0
    sog: Optional[float] = None
    cog: Optional[float] = None
    heading: Optional[float] = None

    # Component scores
    component_scores: ComponentScores = field(default_factory=ComponentScores)

    # Final weighted score and rank
    final_score: float = 0.0
    rank: int = 0

    # Evidence and explanation
    evidence_summary: str = ""
    evidence_details: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    drift_model: str = "unknown"
    scoring_weights: Optional[AttributionWeights] = None
    assumptions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    scored_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """Validate final score is in [0, 1]."""
        if not (0.0 <= self.final_score <= 1.0):
            raise ValueError(f"Final score must be in [0, 1], got {self.final_score}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "rank": self.rank,
            "mmsi": self.mmsi,
            "imo": self.imo,
            "vessel_name": self.vessel_name,
            "vessel_type": self.vessel_type,
            "vessel_type_category": self.vessel_type_category.value,
            "spill_location": {
                "lat": self.spill_latitude,
                "lon": self.spill_longitude,
            },
            "spill_time": self.spill_time.isoformat() if self.spill_time else None,
            "backtracked_origin": {
                "lat": self.backtracked_origin_lat,
                "lon": self.backtracked_origin_lon,
            },
            "backtracked_time": self.backtracked_time.isoformat() if self.backtracked_time else None,
            "vessel_location": {
                "lat": self.vessel_lat,
                "lon": self.vessel_lon,
            },
            "vessel_time": self.vessel_time.isoformat() if self.vessel_time else None,
            "distance_km": self.distance_to_origin_km,
            "bearing_deg": self.bearing_from_origin,
            "time_diff_hours": self.time_diff_hours,
            "sog": self.sog,
            "cog": self.cog,
            "heading": self.heading,
            "component_scores": self.component_scores.as_dict(),
            "final_score": self.final_score,
            "rank": self.rank,
            "evidence_summary": self.evidence_summary,
            "evidence_details": self.evidence_details,
            "drift_model": self.drift_model,
            "scoring_weights": self.scoring_weights.as_dict() if self.scoring_weights else None,
            "assumptions": self.assumptions,
            "warnings": self.warnings,
            "scored_at": self.scored_at.isoformat() if self.scored_at else None,
        }

    def to_row(self) -> Dict[str, Any]:
        """Flattened row for tabular display (dashboard/CSV)."""
        return {
            "rank": self.rank,
            "mmsi": self.mmsi,
            "imo": self.imo,
            "vessel_name": self.vessel_name,
            "vessel_type": self.vessel_type,
            "distance_km": round(self.distance_to_origin_km, 3),
            "time_diff_hours": round(self.time_diff_hours, 2) if self.time_diff_hours else None,
            "distance_score": round(self.component_scores.distance_score, 3),
            "temporal_score": round(self.component_scores.temporal_score, 3),
            "drift_score": round(self.component_scores.drift_score, 3),
            "vessel_type_score": round(self.component_scores.vessel_type_score, 3),
            "data_quality_score": round(self.component_scores.data_quality_score, 3),
            "final_score": round(self.final_score, 3),
        }


@dataclass
class AttributionResult:
    """
    Complete attribution analysis result for a spill event.
    Contains ranked candidates and summary statistics.
    """
    # Spill information
    spill_latitude: float
    spill_longitude: float
    spill_time: datetime

    # Drift context
    backtracked_origin_lat: float
    backtracked_origin_lon: float
    backtracked_time: datetime
    drift_duration_hours: float
    drift_model: str

    # Ranked candidates
    candidates: List[AttributionCandidate]

    # Search parameters
    search_radius_km: float
    time_window_hours: float
    drift_duration_hours: float

    # Scoring configuration
    scoring_weights: AttributionWeights

    # Summary
    total_candidates: int = 0
    candidates_by_type: Dict[str, int] = field(default_factory=dict)

    # Metadata
    assumptions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    analysis_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """Validate and sort candidates by final score descending."""
        self.candidates.sort(key=lambda c: c.final_score, reverse=True)
        for i, candidate in enumerate(self.candidates, 1):
            candidate.rank = i
        self.total_candidates = len(self.candidates)

        # Count by type
        self.candidates_by_type = {}
        for c in self.candidates:
            vt = c.vessel_type or "Unknown"
            self.candidates_by_type[vt] = self.candidates_by_type.get(vt, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spill_location": {
                "lat": self.spill_latitude,
                "lon": self.spill_longitude,
            },
            "spill_time": self.spill_time.isoformat() if self.spill_time else None,
            "backtracked_origin": {
                "lat": self.backtracked_origin_lat,
                "lon": self.backtracked_origin_lon,
            },
            "backtracked_time": self.backtracked_time.isoformat() if self.backtracked_time else None,
            "drift_duration_hours": self.drift_duration_hours,
            "drift_model": self.drift_model,
            "search_radius_km": self.search_radius_km,
            "time_window_hours": self.time_window_hours,
            "drift_duration_hours": self.drift_duration_hours,
            "scoring_weights": self.scoring_weights.as_dict(),
            "total_candidates": self.total_candidates,
            "candidates_by_type": self.candidates_by_type,
            "candidates": [c.to_dict() for c in self.candidates],
            "assumptions": self.assumptions,
            "warnings": self.warnings,
            "analysis_time": self.analysis_time.isoformat() if self.analysis_time else None,
        }

    def to_table(self) -> List[Dict[str, Any]]:
        """Return flattened table for CSV/dashboard."""
        return [c.to_row() for c in self.candidates]

    def get_top_n(self, n: int) -> List[AttributionCandidate]:
        """Get top N candidates by score."""
        return self.candidates[:n]


def categorize_vessel_type(vessel_type: Optional[str]) -> VesselTypeCategory:
    """Map vessel type string to category enum."""
    if not vessel_type:
        return VesselTypeCategory.UNKNOWN
    
    vt_lower = vessel_type.lower()
    if "tanker" in vt_lower:
        return VesselTypeCategory.TANKER
    elif "cargo" in vt_lower or "freight" in vt_lower or "container" in vt_lower:
        return VesselTypeCategory.CARGO
    elif "fish" in vt_lower:
        return VesselTypeCategory.FISHING
    elif "passenger" in vt_lower or "ferry" in vt_lower or "cruise" in vt_lower:
        return VesselTypeCategory.PASSENGER
    elif "tug" in vt_lower:
        return VesselTypeCategory.TUG
    elif "offshore" in vt_lower or "supply" in vt_lower:
        return VesselTypeCategory.OFFSHORE
    return VesselTypeCategory.UNKNOWN