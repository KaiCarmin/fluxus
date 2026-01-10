from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from typing import Literal

class SimulationConfig(BaseSettings):
    # Physics Parameters
    gamma: float = Field(1.4, gt=1.0, description="Adiabatic index")
    
    # Grid Parameters
    nx: int = Field(..., gt=0, description="Number of cells in X")
    ny: int = Field(..., gt=0, description="Number of cells in Y")
    ng: int = Field(2, ge=2, description="Number of ghost cells")
    
    # Solver Strategy
    riemann_solver: Literal["HLL", "HLLC", "Exact"] = "HLLC"
    time_integrator: Literal["Godunov", "RK2"] = "Godunov"
    cfl: float = Field(0.8, gt=0, le=1.0, description="CFL safety factor")

    # Validator example
    @field_validator("riemann_solver")
    @classmethod
    def check_solver_support(cls, v):
        if v == "Exact":
            print("Warning: Exact solver is slow!")
        return v

    class Config:
        # Allows loading from environment variables (e.g., FLUXUS_GAMMA=1.67)
        env_prefix = "FLUXUS_"