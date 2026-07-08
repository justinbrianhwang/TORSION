"""Analysis-only metrics derived from existing TORSION traces."""

from torsion.analysis.propagation import (
    NormalizedStageError,
    RecoveryTime,
    StageError,
    StageScales,
    attenuation_ratios,
    censor_ttc,
    compute_normalized_stage_errors,
    compute_propagation_metrics,
    compute_stage_errors,
    compute_stage_scales,
    critical_interface_score,
    fault_amplification_ratio,
    interface_gains,
    propagation_depth,
    recovery_time,
    source_fault_amplification_ratio,
)
from torsion.analysis.transfer_function import (
    INTERFACE_ORDER,
    characterize_linearity,
    interface_raw_gains,
)

__all__ = [
    "NormalizedStageError",
    "RecoveryTime",
    "StageError",
    "StageScales",
    "attenuation_ratios",
    "censor_ttc",
    "compute_normalized_stage_errors",
    "compute_propagation_metrics",
    "compute_stage_errors",
    "compute_stage_scales",
    "critical_interface_score",
    "fault_amplification_ratio",
    "interface_gains",
    "INTERFACE_ORDER",
    "characterize_linearity",
    "interface_raw_gains",
    "propagation_depth",
    "recovery_time",
    "source_fault_amplification_ratio",
]
