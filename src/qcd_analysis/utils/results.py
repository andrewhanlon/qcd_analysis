from dataclasses import dataclass

from qcd_analysis.data_handling import data_handler

@dataclass
class SingleHadronResult:
    particle_name: str
    Psq: int
    energy: data_handler.Data
    fit_model: str
    tmin: int
    tmax: int
    Q: float

