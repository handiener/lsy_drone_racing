"""Competition loader entry point for the Level 2 PPO actor."""

from lsy_drone_racing.control.rl_single_l2.controller import Level2RLController


class SubmissionController(Level2RLController):
    """Expose one locally defined controller class to the official loader."""
