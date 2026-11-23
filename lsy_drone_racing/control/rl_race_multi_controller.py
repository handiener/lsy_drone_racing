"""Official loader entry point for rl_race_multi recurrent exports."""

from lsy_drone_racing.control.rl_race_multi.controller import RaceController


class SubmissionController(RaceController):
    """One rank-local LSTM policy instance per physical drone."""
