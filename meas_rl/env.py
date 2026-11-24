"""Gymnasium environment: schedule measurements, then decide when to stop.

Every episode hands the agent a fresh device it cannot see. On each step it
picks a frequency, picks how long to dwell there, and says whether it is done.
Dwell time buys accuracy and costs wall clock. Stopping early is cheap and
risks reporting the wrong device; stopping late is safe and wastes instrument
time. The whole task is that trade.

Scoring is deliberately all-or-nothing at the end: the reported centre
frequency has to land within `F0_TOL` and the reported Q within `Q_REL_TOL`,
or the episode counts as a miss. A calibration either passes spec or it does
not, and a reward that paid out partial credit for a near miss would train a
policy that stops too early.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import physics
from .belief import Belief, best_frequency

# What counts as having characterised the device.
F0_TOL = 0.001  # GHz, i.e. 1 MHz
Q_REL_TOL = 0.10  # 10 % relative error on Q

# Episode limits.
MAX_STEPS = 40
TIME_BUDGET = 60.0  # ms

# Discrete action grid, used when ``action_mode="discrete"``.
N_FREQ_BINS = 161  # 1.25 MHz apart across the band
N_DWELL_LEVELS = 7  # log-spaced from DWELL_MIN to DWELL_MAX

# Histogram resolution of the belief marginals, used when ``obs_mode="marginal"``.
N_F0_HIST = 40
N_LOGQ_HIST = 8

# Reward weights.
COST_WEIGHT = 0.05  # reward units charged per ms of instrument time
INFO_WEIGHT = 0.20  # weight on the potential-based information bonus
SUCCESS_BONUS = 10.0
MISS_PENALTY = 10.0


class AdaptiveMeasureEnv(gym.Env):
    """Pick where to measure, how long, and when to stop.

    Observation (9 numbers, all rescaled to roughly [-1, 1]):
        0  posterior mean of f0, mapped across the band
        1  posterior spread of f0, log-scaled
        2  posterior mean of log Q, mapped across its range
        3  posterior spread of log Q, log-scaled
        4  posterior entropy as a fraction of the prior entropy
        5  fraction of the time budget already spent
        6  fraction of the step limit already used
        7  frequency of the last measurement, mapped across the band
        8  the last reading

    With ``obs_mode="marginal"`` the observation also carries the belief
    itself: the posterior marginal of f0 binned into 40 slices of the band,
    then the marginal of log Q in 8 slices (57 numbers in all). During the
    search phase the posterior is multimodal, with holes wherever a near-zero
    reading has ruled the peak out, and a mean and a spread cannot say where
    the agent has not looked yet. The marginal can.

    Action, ``action_mode="continuous"`` (3 numbers in [-1, 1]):
        0  next frequency, mapped linearly across the band
        1  dwell time, mapped log-linearly over the allowed range
        2  stop signal; the episode ends when it is positive

    Action, ``action_mode="discrete"`` (MultiDiscrete [161, 7, 2]):
        0  frequency bin, 1.25 MHz apart across the band
        1  dwell level, log-spaced from 0.1 ms to 10 ms
        2  1 to stop, 0 to keep measuring

    Action, ``action_mode="dwell_stop"`` (MultiDiscrete [7, 2]):
        0  dwell level, log-spaced from 0.1 ms to 10 ms
        1  1 to stop, 0 to keep measuring

        Where to measure is not the agent's call in this mode: the frequency
        is the one with the largest expected information gain under the
        current belief at the chosen dwell (`belief.best_frequency`). The
        agent learns the cost side of the problem, how long to integrate and
        when the answer is good enough, on top of a placement rule that is
        already near-optimal.

        The continuous form is what SAC needs. With PPO's Gaussian policy it
        has a failure mode: once a mean drifts past the [-1, 1] bound, every
        sample is clipped to the same action, the environment stops
        responding to that output, and it never comes back. Categorical heads
        cannot saturate that way, and they turn "stop" into an explicit
        probability.

    Reward:
        every measurement is charged ``COST_WEIGHT`` per millisecond of
        instrument time. On stopping the agent collects ``SUCCESS_BONUS`` if
        both reported parameters are inside tolerance and pays
        ``MISS_PENALTY`` otherwise.

        On top of that sits potential-based shaping (Ng, Harada and Russell,
        1999) with potential ``Phi(s) = -INFO_WEIGHT * entropy(s)`` and
        ``Phi = 0`` at the absorbing terminal state. Each transition adds
        ``gamma * Phi(s') - Phi(s)``, so a reading that sharpens the belief is
        paid for immediately, and over a whole episode the bonus sums to the
        same constant (``INFO_WEIGHT * prior entropy`` when ``gamma = 1``)
        whatever the agent does. It speeds up credit assignment without
        changing which policy is optimal. Pass the agent's discount as
        ``shaping_gamma`` to keep that guarantee exact.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cost_weight: float = COST_WEIGHT,
        info_weight: float = INFO_WEIGHT,
        max_steps: int = MAX_STEPS,
        time_budget: float = TIME_BUDGET,
        shaping_gamma: float = 1.0,
        action_mode: str = "continuous",
        obs_mode: str = "summary",
    ) -> None:
        super().__init__()
        self.cost_weight = cost_weight
        self.info_weight = info_weight
        self.max_steps = max_steps
        self.time_budget = time_budget
        self.shaping_gamma = shaping_gamma

        if obs_mode not in ("summary", "marginal"):
            raise ValueError(f"unknown obs_mode {obs_mode!r}")
        self.obs_mode = obs_mode
        n_obs = 9 + (N_F0_HIST + N_LOGQ_HIST if obs_mode == "marginal" else 0)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )
        if action_mode not in ("continuous", "discrete", "dwell_stop"):
            raise ValueError(f"unknown action_mode {action_mode!r}")
        self.action_mode = action_mode
        if action_mode == "continuous":
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        elif action_mode == "discrete":
            self.action_space = spaces.MultiDiscrete([N_FREQ_BINS, N_DWELL_LEVELS, 2])
        else:
            self.action_space = spaces.MultiDiscrete([N_DWELL_LEVELS, 2])
        self.freq_bins = np.linspace(physics.F_MIN, physics.F_MAX, N_FREQ_BINS)
        self.dwell_levels = np.geomspace(physics.DWELL_MIN, physics.DWELL_MAX, N_DWELL_LEVELS)

        self._log_dwell_min = float(np.log(physics.DWELL_MIN))
        self._log_dwell_max = float(np.log(physics.DWELL_MAX))

        self.belief = Belief()
        self._prior_entropy = self.belief.entropy()
        self.f0_true = 0.0
        self.q_true = 0.0

    # ------------------------------------------------------------- gym API

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.belief = Belief()
        self.f0_true, self.q_true = physics.sample_device(self.np_random)
        self.time_used = 0.0
        self.n_steps = 0
        self.last_freq = 0.5 * (physics.F_MIN + physics.F_MAX)
        self.last_reading = 0.0
        self.last_dwell = 0.0
        self._entropy = self.belief.entropy()
        return self._observation(), {}

    def step(self, action: np.ndarray):
        freq, dwell, wants_stop = self.decode(action)

        if wants_stop:
            return self._finish(forced=False)

        reading = physics.measure(
            freq, dwell, self.f0_true, self.q_true, self.np_random
        )
        self.belief.update(freq, dwell, reading)
        cost = physics.measurement_cost(dwell)
        self.time_used += cost
        self.n_steps += 1
        self.last_freq = freq
        self.last_dwell = dwell
        self.last_reading = reading

        new_entropy = self.belief.entropy()
        # gamma * Phi(s') - Phi(s) with Phi = -info_weight * entropy.
        reward = self.info_weight * (self._entropy - self.shaping_gamma * new_entropy)
        reward -= self.cost_weight * cost
        self._entropy = new_entropy

        out_of_budget = (
            self.time_used >= self.time_budget or self.n_steps >= self.max_steps
        )
        if out_of_budget:
            obs, stop_reward, terminated, truncated, info = self._finish(forced=True)
            return obs, reward + stop_reward, terminated, truncated, info

        return self._observation(), reward, False, False, {}

    # -------------------------------------------------------------- helpers

    def _finish(self, forced: bool):
        """End the episode: report the posterior estimate and score it."""
        f0_hat, q_hat = self.belief.estimate()
        f0_err = abs(f0_hat - self.f0_true)
        q_rel_err = abs(q_hat - self.q_true) / self.q_true
        success = bool(f0_err <= F0_TOL and q_rel_err <= Q_REL_TOL)
        reward = SUCCESS_BONUS if success else -MISS_PENALTY
        # Shaping into the absorbing state, where Phi = 0: 0 - Phi(s).
        reward += self.info_weight * self._entropy
        info = {
            "success": success,
            "f0_err": f0_err,
            "q_rel_err": q_rel_err,
            "time_used": self.time_used,
            "n_measurements": self.n_steps,
            "forced_stop": forced,
            "f0_true": self.f0_true,
            "q_true": self.q_true,
            "f0_hat": f0_hat,
            "q_hat": q_hat,
        }
        # Both kinds of stop are real terminations. Running out of budget is
        # not a time-limit truncation: the report has been made and scored,
        # and nothing follows. Flagging it as truncated makes Stable-Baselines3
        # bootstrap ``gamma * V(s_final)`` on top of the terminal reward, which
        # pays the success bonus twice and teaches the agent to ride the
        # budget instead of stopping. See tests/test_env.py.
        return self._observation(), reward, True, False, info

    def decode(self, action) -> tuple[float, float, bool]:
        """Turn an action into (frequency in GHz, dwell in ms, stop?)."""
        if self.action_mode == "dwell_stop":
            a = np.asarray(action).astype(int).reshape(2)
            dwell = float(self.dwell_levels[a[0]])
            if a[1] == 1:
                return 0.0, dwell, True
            return best_frequency(self.belief, dwell), dwell, False
        if self.action_mode == "discrete":
            a = np.asarray(action).astype(int).reshape(3)
            return float(self.freq_bins[a[0]]), float(self.dwell_levels[a[1]]), bool(a[2] == 1)
        a = np.asarray(action, dtype=float).reshape(3)
        return self._decode_freq(a[0]), self._decode_dwell(a[1]), bool(a[2] > 0.0)

    def _decode_freq(self, raw: float) -> float:
        unit = 0.5 * (float(np.clip(raw, -1.0, 1.0)) + 1.0)
        return physics.F_MIN + unit * (physics.F_MAX - physics.F_MIN)

    def _decode_dwell(self, raw: float) -> float:
        unit = 0.5 * (float(np.clip(raw, -1.0, 1.0)) + 1.0)
        return float(
            np.exp(self._log_dwell_min + unit * (self._log_dwell_max - self._log_dwell_min))
        )

    def _observation(self) -> np.ndarray:
        f0_mean, f0_std, logq_mean, logq_std = self.belief.mean_std()
        band = physics.F_MAX - physics.F_MIN
        logq_span = np.log(physics.Q_MAX) - np.log(physics.Q_MIN)
        obs = np.array(
            [
                (f0_mean - physics.F_MIN) / band * 2.0 - 1.0,
                np.log10(max(f0_std, 1e-6)),
                (logq_mean - np.log(physics.Q_MIN)) / logq_span * 2.0 - 1.0,
                np.log10(max(logq_std, 1e-6)),
                self._entropy / self._prior_entropy,
                self.time_used / self.time_budget,
                self.n_steps / self.max_steps,
                (self.last_freq - physics.F_MIN) / band * 2.0 - 1.0,
                self.last_reading,
            ],
            dtype=np.float32,
        )
        if self.obs_mode == "marginal":
            post = self.belief.posterior.reshape(self.belief.f0_axis.size, self.belief.logq_axis.size)
            f0_marg = _rebin(post.sum(axis=1), N_F0_HIST)
            logq_marg = _rebin(post.sum(axis=0), N_LOGQ_HIST)
            obs = np.concatenate([obs, f0_marg, logq_marg]).astype(np.float32)
        return obs


def _rebin(p: np.ndarray, n: int) -> np.ndarray:
    """Sum a 1-D probability vector into ``n`` contiguous bins."""
    edges = np.linspace(0, p.size, n + 1).astype(int)
    return np.add.reduceat(p, edges[:-1])


def make_env(**kwargs) -> AdaptiveMeasureEnv:
    return AdaptiveMeasureEnv(**kwargs)
