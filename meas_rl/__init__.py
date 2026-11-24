"""Adaptive measurement scheduling and stopping with reinforcement learning."""

from gymnasium.envs.registration import register

from .env import AdaptiveMeasureEnv

register(id="AdaptiveMeasure-v0", entry_point="meas_rl.env:AdaptiveMeasureEnv")
register(
    id="AdaptiveMeasureDiscrete-v0",
    entry_point="meas_rl.env:AdaptiveMeasureEnv",
    kwargs={"action_mode": "discrete"},
)

register(
    id="AdaptiveMeasureDwellStop-v0",
    entry_point="meas_rl.env:AdaptiveMeasureEnv",
    kwargs={"action_mode": "dwell_stop"},
)

__all__ = ["AdaptiveMeasureEnv"]
