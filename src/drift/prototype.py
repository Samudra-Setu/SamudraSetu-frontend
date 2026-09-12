"""
PROTOTYPE Drift Model for Samudra Setu.

This is a DEMONSTRATION model with simplified assumptions.
It is NOT a validated ocean prediction model.

WARNING: This model uses constant or uniform current assumptions.
Real ocean drift requires validated current data from sources like:
- HYCOM (Hybrid Coordinate Ocean Model)
- Copernicus Marine Environment Monitoring Service
- NOAA NCEP Global Ocean Forecast System
- OSCAR (Ocean Surface Current Analyses Real-time)

USE ONLY for development, testing, and demonstration purposes.
"""

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta, timezone
import numpy as np
import warnings

from src.drift.base import (
    BaseDriftModel,
    CurrentVector,
    DriftStep,
    DriftResult,
    OilSpillDriftQuery,
    DriftDirection,
    CurrentSource,
    DriftModelFactory,
    haversine_distance,
    calculate_bearing,
)


class PrototypeConstantCurrentModel(BaseDriftModel):
    """
    PROTOTYPE: Constant current drift model.
    
    Assumes a single, uniform current vector everywhere for all time.
    This is a gross oversimplification of real ocean dynamics.
    
    Assumptions:
    - Current is constant in space and time
    - No wind drift (leeway) included
    - No Stokes drift included
    - No diffusion/random walk
    - Flat Earth approximation for coordinate conversion
    - Surface current only (no vertical shear)
    
    For demonstration only. Do not use for real spill response.
    """
    
    def __init__(
        self,
        u_velocity: float = 0.1,      # Eastward current (m/s) ~ 8.6 km/day
        v_velocity: float = 0.05,     # Northward current (m/s) ~ 4.3 km/day
        model_name: str = "PrototypeConstantCurrent",
    ):
        """
        Initialize constant current model.
        
        Args:
            u_velocity: Eastward current component in m/s (positive = eastward)
            v_velocity: Northward current component in m/s (positive = northward)
            model_name: Name for this model instance
        """
        assumptions = [
            "Constant current everywhere (spatially uniform)",
            "Constant current forever (temporally uniform)",
            "No wind drift (leeway) included",
            "No Stokes drift included",
            "No turbulent diffusion / random walk",
            "Flat Earth coordinate approximation",
            "Surface current only (no vertical structure)",
            "No coastline/bathymetry interaction",
        ]
        
        warnings_list = [
            "PROTOTYPE MODEL - NOT VALIDATED FOR REAL SPILL RESPONSE",
            "Uses constant current assumption - real oceans have spatially/temporally varying currents",
            "No wind, Stokes drift, or diffusion included",
            "Results are illustrative only",
        ]
        
        super().__init__(
            model_name=model_name,
            current_source=CurrentSource.PROTOTYPE_CONSTANT,
            assumptions=assumptions,
        )
        
        self.u_velocity = u_velocity
        self.v_velocity = v_velocity
        self.warnings = warnings_list
        
        # Add specific warnings
        if abs(u_velocity) > 1.0 or abs(v_velocity) > 1.0:
            self.warnings.append(f"High current speed: u={u_velocity}, v={v_velocity} m/s (> 1 m/s unusual)")
    
    def get_current(
        self,
        latitude: float,
        longitude: float,
        timestamp: datetime,
        depth: float = 0.0,
    ) -> CurrentVector:
        """Return the constant current vector."""
        return CurrentVector(
            latitude=latitude,
            longitude=longitude,
            u_velocity=self.u_velocity,
            v_velocity=self.v_velocity,
            timestamp=timestamp,
            depth=depth,
            source=self.current_source,
        )
    
    def simulate(self, query: OilSpillDriftQuery) -> DriftResult:
        """
        Run drift simulation with constant current.
        
        Uses RK4 integration for accuracy.
        """
        import time
        sim_start = time.time()
        
        # Validate inputs
        if not self._validate_location(query.spill_latitude, query.spill_longitude):
            raise ValueError(f"Invalid location: {query.spill_latitude}, {query.spill_longitude}")
        
        if query.backtrack_hours <= 0:
            raise ValueError("backtrack_hours must be positive")
        
        if query.time_step_hours <= 0:
            raise ValueError("time_step_hours must be positive")
        
        # Determine direction
        direction = DriftDirection.BACKWARD  # Default for spill origin finding
        
        # Use query time step or default
        dt = query.time_step_hours
        
        # Number of steps
        n_steps = int(query.backtrack_hours / dt)
        if n_steps == 0:
            n_steps = 1
            dt = query.backtrack_hours
        
        # Initialize
        lat = query.spill_latitude
        lon = query.spill_longitude
        current_time = query.spill_time
        
        trajectory = []
        cumulative_distance = 0.0
        
        # Run simulation
        for step in range(n_steps + 1):
            # Get current
            current = self.get_current(lat, lon, current_time)
            
            if step == 0:
                displacement = 0.0
            else:
                # Calculate displacement from previous step
                prev_step = trajectory[-1]
                displacement = haversine_distance(prev_step.latitude, prev_step.longitude, lat, lon)
                cumulative_distance += displacement
            
            # Create step record
            drift_step = DriftStep(
                step=step,
                latitude=lat,
                longitude=lon,
                timestamp=current_time,
                current_u=current.u_velocity,
                current_v=current.v_velocity,
                current_speed=current.speed,
                current_direction=current.direction,
                displacement_km=displacement,
                cumulative_distance_km=cumulative_distance,
            )
            trajectory.append(drift_step)
            
            # Integrate to next step (unless last step)
            if step < n_steps:
                lat, lon, _ = self._rk4_step(lat, lon, current_time, dt, direction)
                current_time = current_time + timedelta(hours=-dt)
        
        # Final position
        end_lat = trajectory[-1].latitude
        end_lon = trajectory[-1].longitude
        end_time = trajectory[-1].timestamp
        
        # Net displacement (straight line)
        net_disp = haversine_distance(
            query.spill_latitude, query.spill_longitude,
            end_lat, end_lon
        )
        
        # Average current speed
        avg_speed = np.mean([s.current_speed for s in trajectory])
        
        sim_time = time.time() - sim_start
        
        return DriftResult(
            start_latitude=query.spill_latitude,
            start_longitude=query.spill_longitude,
            start_time=query.spill_time,
            direction=direction,
            duration_hours=query.backtrack_hours,
            time_step_hours=query.time_step_hours,
            trajectory=trajectory,
            end_latitude=end_lat,
            end_longitude=end_lon,
            end_time=end_time,
            total_distance_km=cumulative_distance,
            net_displacement_km=net_disp,
            avg_current_speed=avg_speed,
            current_source=self.current_source,
            model_name=self.model_name,
            assumptions=self.assumptions,
            warnings=self.warnings,
            simulation_time=datetime.now(timezone.utc),
        )


class PrototypeUniformCurrentModel(BaseDriftModel):
    """
    PROTOTYPE: Uniform current within a region.
    
    Current varies by latitude band but constant in time.
    Still a gross oversimplification - for demonstration only.
    """
    
    def __init__(
        self,
        # Current as function of latitude: u = f(lat), v = f(lat)
        current_func: Optional[callable] = None,
        model_name: str = "PrototypeUniformCurrent",
    ):
        assumptions = [
            "Current varies only with latitude (zonal bands)",
            "Constant current in time",
            "No wind drift (leeway) included",
            "No Stokes drift included",
            "No turbulent diffusion",
            "Flat Earth approximation",
            "No coastline/bathymetry interaction",
        ]
        
        warnings_list = [
            "PROTOTYPE MODEL - NOT VALIDATED FOR REAL SPILL RESPONSE",
            "Zonal current assumption - real oceans have eddies, fronts, etc.",
            "No wind, Stokes drift, or diffusion included",
            "Results are illustrative only",
        ]
        
        super().__init__(
            model_name=model_name,
            current_source=CurrentSource.PROTOTYPE_UNIFORM,
            assumptions=assumptions,
        )
        
        self.warnings = warnings_list
        
        # Default: simple westward current strengthening toward equator
        if current_func is None:
            self.current_func = self._default_current
        else:
            self.current_func = current_func
    
    def _default_current(self, lat: float) -> Tuple[float, float]:
        """Default: westward current, stronger near equator."""
        # u: eastward positive, v: northward positive
        # Westward = negative u
        # Stronger near equator: cos(lat) pattern
        u = -0.15 * np.cos(np.radians(lat))  # Up to 0.15 m/s westward
        v = 0.02 * np.sin(np.radians(2 * lat))  # Weak meridional
        return u, v
    
    def get_current(
        self,
        latitude: float,
        longitude: float,
        timestamp: datetime,
        depth: float = 0.0,
    ) -> CurrentVector:
        u, v = self.current_func(latitude)
        return CurrentVector(
            latitude=latitude,
            longitude=longitude,
            u_velocity=u,
            v_velocity=v,
            timestamp=timestamp,
            depth=depth,
            source=self.current_source,
        )
    
    def simulate(self, query: OilSpillDriftQuery) -> DriftResult:
        """Run simulation with latitude-dependent current."""
        import time
        sim_start = time.time()
        
        if not self._validate_location(query.spill_latitude, query.spill_longitude):
            raise ValueError(f"Invalid location: {query.spill_latitude}, {query.spill_longitude}")
        
        direction = DriftDirection.BACKWARD
        dt = query.time_step_hours
        n_steps = max(1, int(query.backtrack_hours / dt))
        
        lat = query.spill_latitude
        lon = query.spill_longitude
        current_time = query.spill_time
        
        trajectory = []
        cumulative_distance = 0.0
        
        for step in range(n_steps + 1):
            current = self.get_current(lat, lon, current_time)
            
            if step == 0:
                displacement = 0.0
            else:
                prev = trajectory[-1]
                displacement = haversine_distance(prev.latitude, prev.longitude, lat, lon)
                cumulative_distance += displacement
            
            trajectory.append(DriftStep(
                step=step,
                latitude=lat,
                longitude=lon,
                timestamp=current_time,
                current_u=current.u_velocity,
                current_v=current.v_velocity,
                current_speed=current.speed,
                current_direction=current.direction,
                displacement_km=displacement,
                cumulative_distance_km=cumulative_distance,
            ))
            
            if step < n_steps:
                lat, lon, _ = self._rk4_step(lat, lon, current_time, dt, direction)
                current_time = current_time + timedelta(hours=-dt)
        
        end_lat = trajectory[-1].latitude
        end_lon = trajectory[-1].longitude
        end_time = trajectory[-1].timestamp
        
        net_disp = haversine_distance(query.spill_latitude, query.spill_longitude, end_lat, end_lon)
        avg_speed = np.mean([s.current_speed for s in trajectory])
        
        return DriftResult(
            start_latitude=query.spill_latitude,
            start_longitude=query.spill_longitude,
            start_time=query.spill_time,
            direction=direction,
            duration_hours=query.backtrack_hours,
            time_step_hours=query.time_step_hours,
            trajectory=trajectory,
            end_latitude=end_lat,
            end_longitude=end_lon,
            end_time=end_time,
            total_distance_km=cumulative_distance,
            net_displacement_km=net_disp,
            avg_current_speed=avg_speed,
            current_source=self.current_source,
            model_name=self.model_name,
            assumptions=self.assumptions,
            warnings=self.warnings,
            simulation_time=datetime.now(timezone.utc),
        )


# Register prototype models
DriftModelFactory.register("constant", PrototypeConstantCurrentModel)
DriftModelFactory.register("uniform", PrototypeUniformCurrentModel)


def create_prototype_model(
    model_type: str = "constant",
    **kwargs
) -> BaseDriftModel:
    """
    Factory function for prototype drift models.
    
    Args:
        model_type: "constant" or "uniform"
        **kwargs: Model-specific parameters
        
    Returns:
        BaseDriftModel instance
    """
    return DriftModelFactory.create(model_type, **kwargs)


# Future: Real current data models
# class HYCOMDriftModel(BaseDriftModel): ...
# class CopernicusDriftModel(BaseDriftModel): ...
# class NCEPDriftModel(BaseDriftModel): ...