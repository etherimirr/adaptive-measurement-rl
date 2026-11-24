"""Hand-written measurement policies to compare the learned agent against.

All of them drive the same environment through the same action interface, so
every method is scored by exactly the same code path.

* ``SweepPolicy``: what an instrument does by default. Step across the band
  on an even grid at a fixed dwell, then report.
* ``RandomPolicy``: random frequencies at a fixed dwell, a fixed number of
  times. A floor, not a contender.
* ``InfoGainStopPolicy``: the Bayesian experimental-design baseline. Measure
  wherever the expected information gain is highest, and stop as soon as the
  posterior is tight enough on both parameters. This is the same shape of rule
  as Bayesian optimal stopping for repeated sampling (Huang, Ma and Zhou,
  arXiv:2602.05395): keep paying for samples until the posterior says the
  answer is settled, then stop.
"""

from __future__ import annotations

import numpy as np

from . import physics
from .belief import best_frequency
from .env import AdaptiveMeasureEnv


def _encode_freq(freq: float) -> float:
    return 2.0 * (freq - physics.F_MIN) / (physics.F_MAX - physics.F_MIN) - 1.0


def _encode_dwell(dwell: float) -> float:
    lo, hi = np.log(physics.DWELL_MIN), np.log(physics.DWELL_MAX)
    return 2.0 * (np.log(dwell) - lo) / (hi - lo) - 1.0


STOP = 1.0
GO = -1.0


class SweepPolicy:
    """Even sweep across the band at a fixed dwell, then stop."""

    def __init__(self, n_points: int, dwell: float) -> None:
        self.freqs = np.linspace(physics.F_MIN, physics.F_MAX, n_points)
        self.dwell = dwell
        self.name = f"sweep(n={n_points},dwell={dwell:g})"

    def reset(self) -> None:
        self.i = 0

    def act(self, env: AdaptiveMeasureEnv) -> np.ndarray:
        if self.i >= self.freqs.size:
            return np.array([0.0, 0.0, STOP])
        f = self.freqs[self.i]
        self.i += 1
        return np.array([_encode_freq(f), _encode_dwell(self.dwell), GO])


class RandomPolicy:
    """Uniformly random frequencies at a fixed dwell, a fixed number of times."""

    def __init__(self, n_points: int, dwell: float, seed: int = 0) -> None:
        self.n_points = n_points
        self.dwell = dwell
        self.rng = np.random.default_rng(seed)
        self.name = f"random(n={n_points},dwell={dwell:g})"

    def reset(self) -> None:
        self.i = 0

    def act(self, env: AdaptiveMeasureEnv) -> np.ndarray:
        if self.i >= self.n_points:
            return np.array([0.0, 0.0, STOP])
        self.i += 1
        f = self.rng.uniform(physics.F_MIN, physics.F_MAX)
        return np.array([_encode_freq(f), _encode_dwell(self.dwell), GO])


class InfoGainStopPolicy:
    """Greedy expected-information-gain placement with a posterior stopping rule.

    Stops once the posterior standard deviation of ``f0`` is below
    ``f0_std_stop`` GHz and that of ``log Q`` is below ``logq_std_stop``.
    Since ``log Q`` spread is roughly relative error in Q, a ``logq_std_stop``
    of 0.04 asks for Q known to about 4 percent at one standard deviation.
    """

    def __init__(
        self,
        dwell: float,
        f0_std_stop: float = 0.0003,
        logq_std_stop: float = 0.04,
    ) -> None:
        self.dwell = dwell
        self.f0_std_stop = f0_std_stop
        self.logq_std_stop = logq_std_stop
        self.name = (
            f"infogain(dwell={dwell:g},f0_sd<{f0_std_stop * 1e3:g}MHz,"
            f"logQ_sd<{logq_std_stop:g})"
        )

    def reset(self) -> None:
        pass

    def act(self, env: AdaptiveMeasureEnv) -> np.ndarray:
        belief = env.belief
        _, f0_std, _, logq_std = belief.mean_std()
        if env.n_steps > 0 and f0_std < self.f0_std_stop and logq_std < self.logq_std_stop:
            return np.array([0.0, 0.0, STOP])
        f = best_frequency(belief, self.dwell)
        return np.array([_encode_freq(f), _encode_dwell(self.dwell), GO])
