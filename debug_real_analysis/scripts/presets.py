"""Named diagnostic presets; actuator_delay is the current reference, not hardware calibration."""

from dataclasses import replace
from harness import Noise

REFERENCE_PRESET = "actuator_delay"
PRESETS = {
    "baseline": Noise(),
    REFERENCE_PRESET: Noise(rotor_rate=0.8, omega_delay=2, thrust_cap=0.70),
    # Earlier gain-changing candidates are retained for historical comparisons.
    "paired": Noise(
        omega_tau=0.05,
        pose_tau=0.01,
        thrust_cap=0.76,
        damping=0.7,
        command_substeps=2,
        drag_scale=1.4,
    ),
    "input_match": Noise(
        omega_tau=0.1,
        pose_tau=0.02,
        thrust_cap=0.72,
        damping=0.7,
        command_substeps=3,
        drag_scale=1.4,
    ),
    "minimal": Noise(omega_tau=0.1, thrust_cap=0.72, damping=0.7),
}
REFERENCE_ABLATIONS = {
    REFERENCE_PRESET: PRESETS[REFERENCE_PRESET],
    "without_motor_slowing": replace(PRESETS[REFERENCE_PRESET], rotor_rate=1.0),
    "without_input_delay": replace(PRESETS[REFERENCE_PRESET], omega_delay=0),
    "without_capacity_reduction": replace(PRESETS[REFERENCE_PRESET], thrust_cap=0.8),
}
