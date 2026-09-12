"""
Base classes and interfaces for Ocean Drift Modeling in Samudra Setu.

This module defines the abstract interface that all drift models must implement.
Real ocean-current-based models (HYCOM, Copernicus, etc.) should inherit from BaseDriftModel.

IMPORTANT: The prototype implementation in prototype.py uses simplified assumptions
and is NOT a validated ocean prediction model. It is for demonstration only.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Union
from datetime import datetime, timedelta, timezone
from enum import Enum
import numpy as np


class DriftDirection(Enum):
    """Direction of drift simulation."""
    FORWARD = "forward"      # Spill origin -> future position
    BACKWARD = "backward"    # Spill observation -> probable origin


class CurrentSource(Enum):
    """Source of ocean current data."""
    HYCOM = "hycom"
    COPERNICUS = "copernicus"
    NCEP = "ncep"
    OSCAR = "oscar"
    PROTOTYPE_CONSTANT = "prototype_constant"
    PROTOTYPE_UNIFORM = "prototype_uniform"
    USER_PROVIDED = "user_provided"


@dataclass
class CurrentVector:
    """Ocean current velocity vector at a location and time."""
    latitude: float
    longitude: float
    u_velocity: float      # Eastward component (m/s)
    v_velocity: float      # Northward component (m/s)
    timestamp: datetime
    depth: float = 0.0     # Depth in meters (0 = surface)
    source: CurrentSource = CurrentSource.USER_PROVIDED
    
    @property
    def speed(self) -> float:
        """Current speed in m/s."""
        return np.sqrt(self.u_velocity**2 + self.v_velocity**2)
    
    @property
    def direction(self) -> float:
        """Current direction in degrees (0=N, 90=E, 180=S, 270=W)."""
        return (np.degrees(np.arctan2(self.u_velocity, self.v_velocity)) + 360) % 360
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'latitude': self.latitude,
            'longitude': self.longitude,
            'u_velocity': self.u_velocity,
            'v_velocity': self.v_velocity,
            'speed': self.speed,
            'direction': self.direction,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'depth': self.depth,
            'source': self.source.value,
        }


@dataclass
class DriftStep:
    """Single step in a drift trajectory."""
    step: int
    latitude: float
    longitude: float
    timestamp: datetime
    current_u: float
    current_v: float
    current_speed: float
    current_direction: float
    displacement_km: float
    cumulative_distance_km: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'step': self.step,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'current_u': self.current_u,
            'current_v': self.current_v,
            'current_speed': self.current_speed,
            'current_direction': self.current_direction,
            'displacement_km': self.displacement_km,
            'cumulative_distance_km': self.cumulative_distance_km,
        }


@dataclass
class DriftResult:
    """Complete result of a drift simulation."""
    # Input parameters
    start_latitude: float
    start_longitude: float
    start_time: datetime
    direction: DriftDirection
    duration_hours: float
    time_step_hours: float
    
    # Output trajectory
    trajectory: List[DriftStep]
    end_latitude: float
    end_longitude: float
    end_time: datetime
    
    # Summary statistics
    total_distance_km: float
    net_displacement_km: float
    avg_current_speed: float
    current_source: CurrentSource
    
    # Metadata
    model_name: str
    assumptions: List[str]
    warnings: List[str]
    simulation_time: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'start_location': {'lat': self.start_latitude, 'lon': self.start_longitude},
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'direction': self.direction.value,
            'duration_hours': self.duration_hours,
            'time_step_hours': self.time_step_hours,
            'end_location': {'lat': self.end_latitude, 'lon': self.end_longitude},
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'trajectory': [step.to_dict() for step in self.trajectory],
            'total_distance_km': self.total_distance_km,
            'net_displacement_km': self.net_displacement_km,
            'avg_current_speed': self.avg_current_speed,
            'current_source': self.current_source.value,
            'model_name': self.model_name,
            'assumptions': self.assumptions,
            'warnings': self.warnings,
            'simulation_time': self.simulation_time.isoformat() if self.simulation_time else None,
        }


@dataclass
class OilSpillDriftQuery:
    """Query parameters for drift simulation from an oil spill observation."""
    spill_latitude: float
    spill_longitude: float
    spill_time: datetime
    backtrack_hours: float = 24.0
    time_step_hours: float = 1.0
    current_source: CurrentSource = CurrentSource.PROTOTYPE_CONSTANT
    current_data: Optional[Any] = None  # For real current data (xarray, netcdf, etc.)


class BaseDriftModel(ABC):
    """
    Abstract base class for ocean drift models.
    
    All drift models (HYCOM, Copernicus, prototype, etc.) must implement
    the simulate method and optionally the get_current method.
    """
    
    def __init__(
        self,
        model_name: str,
        current_source: CurrentSource = CurrentSource.USER_PROVIDED,
        assumptions: Optional[List[str]] = None,
    ):
        self.model_name = model_name
        self.current_source = current_source
        self.assumptions = assumptions or []
        self.warnings = []
    
    @abstractmethod
    def get_current(
        self,
        latitude: float,
        longitude: float,
        timestamp: datetime,
        depth: float = 0.0,
    ) -> CurrentVector:
        """
        Get ocean current at a specific location and time.
        
        Args:
            latitude: Latitude in degrees
            longitude: Longitude in degrees
            timestamp: Time for current query
            depth: Depth in meters (0 = surface)
            
        Returns:
            CurrentVector with u/v components
        """
        pass
    
    @abstractmethod
    def simulate(
        self,
        query: OilSpillDriftQuery,
    ) -> DriftResult:
        """
        Run drift simulation for an oil spill query.
        
        Args:
            query: OilSpillDriftQuery with spill location, time, and parameters
            
        Returns:
            DriftResult with trajectory and statistics
        """
        pass
    
    def _rk4_step(
        self,
        lat: float,
        lon: float,
        time: datetime,
        dt_hours: float,
        direction: DriftDirection,
    ) -> Tuple[float, float, CurrentVector]:
        """
        Single RK4 integration step for particle tracking.
        
        Returns new (lat, lon) and the current used.
        """
        # Get current at current position
        current = self.get_current(lat, lon, time)
        
        # Convert current velocity (m/s) to degrees per hour
        # 1 degree latitude ≈ 111.32 km
        # 1 degree longitude ≈ 111.32 * cos(lat) km
        lat_km_per_deg = 111.32
        lon_km_per_deg = 111.32 * np.cos(np.radians(lat))
        
        # Velocity in degrees per hour
        u_deg_per_hour = (current.u_velocity * 3600) / (lon_km_per_deg * 1000)
        v_deg_per_hour = (current.v_velocity * 3600) / (lat_km_per_deg * 1000)
        
        # RK4 integration
        if direction == DriftDirection.FORWARD:
            sign = 1
        else:
            sign = -1
        
        # k1
        k1_lat = sign * v_deg_per_hour * dt_hours
        k1_lon = sign * u_deg_per_hour * dt_hours
        
        # k2
        lat2 = lat + k1_lat / 2
        lon2 = lon + k1_lon / 2
        time2 = time + timedelta(hours=dt_hours / 2 * sign)
        current2 = self.get_current(lat2, lon2, time2)
        u2 = (current2.u_velocity * 3600) / (111.32 * np.cos(np.radians(lat2)) * 1000)
        v2 = (current2.v_velocity * 3600) / (111.32 * 1000)
        k2_lat = sign * v2 * dt_hours
        k2_lon = sign * u2 * dt_hours
        
        # k3
        lat3 = lat + k2_lat / 2
        lon3 = lon + k2_lon / 2
        time3 = time + timedelta(hours=dt_hours / 2 * sign)
        current3 = self.get_current(lat3, lon3, time3)
        u3 = (current3.u_velocity * 3600) / (111.32 * np.cos(np.radians(lat3)) * 1000)
        v3 = (current3.v_velocity * 3600) / (111.32 * 1000)
        k3_lat = sign * v3 * dt_hours
        k3_lon = sign * u3 * dt_hours
        
        # k4
        lat4 = lat + k3_lat
        lon4 = lon + k3_lon
        time4 = time + timedelta(hours=dt_hours * sign)
        current4 = self.get_current(lat4, lon4, time4)
        u4 = (current4.u_velocity * 3600) / (111.32 * np.cos(np.radians(lat4)) * 1000)
        v4 = (current4.v_velocity * 3600) / (111.32 * 1000)
        k4_lat = sign * v4 * dt_hours
        k4_lon = sign * u4 * dt_hours
        
        # Weighted average
        dlat = (k1_lat + 2*k2_lat + 2*k3_lat + k4_lat) / 6
        dlon = (k1_lon + 2*k2_lon + 2*k3_lon + k4_lon) / 6
        
        new_lat = lat + dlat
        new_lon = lon + dlon
        
        return new_lat, new_lon, current
    
    def _validate_location(self, latitude: float, longitude: float) -> bool:
        """Validate latitude/longitude are within valid ranges."""
        return -90 <= latitude <= 90 and -180 <= longitude <= 180
    
    def _add_warning(self, msg: str):
        self.warnings.append(msg)


class DriftModelFactory:
    """Factory for creating drift model instances."""
    
    _models: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, model_class: type):
        """Register a drift model class."""
        cls._models[name] = model_class
    
    @classmethod
    def create(
        cls,
        name: str,
        **kwargs
    ) -> BaseDriftModel:
        """Create a drift model instance."""
        if name not in cls._models:
            available = list(cls._models.keys())
            raise ValueError(f"Unknown drift model: {name}. Available: {available}")
        return cls._models[name](**kwargs)
    
    @classmethod
    def list_models(cls) -> List[str]:
        """List available model names."""
        return list(cls._models.keys())


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points (km)."""
    from math import radians, sin, cos, sqrt, asin
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371.0 * c


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate bearing from point 1 to point 2 (degrees)."""
    from math import radians, degrees, sin, cos, atan2
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    y = sin(dlon) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    bearing = degrees(atan2(y, x))
    return (bearing + 360) % 360