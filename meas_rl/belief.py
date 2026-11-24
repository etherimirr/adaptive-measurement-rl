"""Exact grid posterior over the two unknown device parameters.

The agent never sees the device. What it sees is this belief: a probability
distribution over ``(f0, Q)`` that starts flat and sharpens with every
reading. Each measurement multiplies the belief by a Gaussian likelihood and
renormalises, which is plain Bayes on a grid rather than any approximation.

Two things make the grid worth keeping instead of a particle filter. It is
exact up to the grid spacing, so a wrong answer is a modelling bug and never a
resampling artifact. And it hands the policy a sufficient statistic for free:
posterior spread and entropy are what "how much do I still not know" means
here, and they are what both the learned policy and the stopping-rule
baselines read.
"""

from __future__ import annotations

import numpy as np

from . import physics

# Frequencies considered when choosing where to measure next: the range a
# resonance can occupy plus a 10 MHz margin each side. A local, finer set
# around the current estimate is added at selection time.
GLOBAL_CANDIDATES = np.linspace(physics.F0_MIN - 0.01, physics.F0_MAX + 0.01, 101)

# Grid resolution. 0.5 MHz in centre frequency is half the 1 MHz tolerance the
# task is scored against, so grid discretisation is not the binding error.
N_F0 = 201
N_LOGQ = 61


class Belief:
    """Posterior over ``(f0, Q)`` on a fixed grid, updated by measurement."""

    def __init__(self) -> None:
        self.f0_axis = np.linspace(physics.F0_MIN, physics.F0_MAX, N_F0)
        self.logq_axis = np.linspace(
            np.log(physics.Q_MIN), np.log(physics.Q_MAX), N_LOGQ
        )
        # Flattened grid of candidate devices, one per cell.
        f0_mesh, logq_mesh = np.meshgrid(self.f0_axis, self.logq_axis, indexing="ij")
        self.f0_grid = f0_mesh.ravel()
        self.q_grid = np.exp(logq_mesh.ravel())
        self.logq_grid = logq_mesh.ravel()
        self.n_cells = self.f0_grid.size
        self._two_q_over_f0 = 2.0 * self.q_grid / self.f0_grid
        # Uniform prior, matching how devices are actually drawn: f0 uniform,
        # Q log-uniform.
        self.log_post = np.full(self.n_cells, -np.log(self.n_cells))
        self._post: np.ndarray | None = None

    # ---------------------------------------------------------------- update

    def model_grid(self, freq: float) -> np.ndarray:
        """Noiseless response at ``freq`` for every candidate device."""
        detune = self._two_q_over_f0 * (freq - self.f0_grid)
        return 1.0 / np.sqrt(1.0 + detune**2)

    def update(self, freq: float, dwell: float, reading: float) -> None:
        """Fold one measurement into the posterior."""
        sigma = physics.noise_std(dwell)
        residual = (reading - self.model_grid(freq)) / sigma
        self.log_post = self.log_post - 0.5 * residual**2
        self.log_post -= _logsumexp(self.log_post)
        self._post = None

    # ------------------------------------------------------------- summaries

    @property
    def posterior(self) -> np.ndarray:
        if self._post is None:
            self._post = np.exp(self.log_post)
            self._post /= self._post.sum()
        return self._post

    def mean_std(self) -> tuple[float, float, float, float]:
        """Posterior mean and standard deviation of ``f0`` and of ``log Q``."""
        p = self.posterior
        f0_mean = float(p @ self.f0_grid)
        f0_var = float(p @ (self.f0_grid - f0_mean) ** 2)
        logq_mean = float(p @ self.logq_grid)
        logq_var = float(p @ (self.logq_grid - logq_mean) ** 2)
        return f0_mean, np.sqrt(max(f0_var, 0.0)), logq_mean, np.sqrt(max(logq_var, 0.0))

    def entropy(self) -> float:
        """Shannon entropy of the posterior in nats."""
        p = self.posterior
        nz = p > 0.0
        return float(-(p[nz] * np.log(p[nz])).sum())

    def estimate(self) -> tuple[float, float]:
        """The answer the agent reports when it stops: (f0_hat, Q_hat).

        Posterior mean for ``f0``, and the geometric mean for ``Q`` since the
        prior and the grid are uniform in ``log Q``.
        """
        f0_mean, _, logq_mean, _ = self.mean_std()
        return f0_mean, float(np.exp(logq_mean))

    # ------------------------------------------------- measurement selection

    def predictive_std(self, freqs: np.ndarray) -> np.ndarray:
        """Posterior standard deviation of the predicted response at ``freqs``.

        Where candidate devices disagree most about what the instrument would
        read is where a reading is most informative. This is the quantity the
        information-gain baseline maximises.
        """
        p = self.posterior
        freqs = np.asarray(freqs, dtype=float)
        # (n_freqs, n_cells): predicted response of every candidate device at
        # every candidate frequency, in one vectorised pass.
        detune = self._two_q_over_f0[None, :] * (freqs[:, None] - self.f0_grid[None, :])
        model = 1.0 / np.sqrt(1.0 + detune**2)
        mean = model @ p
        var = (model**2) @ p - mean**2
        return np.sqrt(np.clip(var, 0.0, None))

    def expected_info_gain(self, freqs: np.ndarray, dwell: float) -> np.ndarray:
        """Expected information gain of measuring each of ``freqs``, in nats.

        Uses the Gaussian approximation
        ``EIG = 0.5 log(1 + Var_posterior[model] / sigma^2)``, which is exact
        when the response is linear in the unknowns and is the standard BALD
        proxy otherwise.
        """
        sigma = physics.noise_std(dwell)
        pred = self.predictive_std(freqs)
        return 0.5 * np.log1p((pred / sigma) ** 2)


def best_frequency(belief: "Belief", dwell: float) -> float:
    """Where to measure next: the candidate with the largest expected
    information gain at this dwell. Candidates are the global set plus a finer
    set around the current estimate, so a narrow high-Q line is not missed
    between coarse candidates."""
    f0_mean, f0_std, _, _ = belief.mean_std()
    local = f0_mean + np.linspace(-4.0, 4.0, 21) * max(f0_std, 2e-4)
    cands = np.clip(np.concatenate([GLOBAL_CANDIDATES, local]), physics.F_MIN, physics.F_MAX)
    gains = belief.expected_info_gain(cands, dwell)
    return float(cands[int(np.argmax(gains))])


def _logsumexp(x: np.ndarray) -> float:
    peak = float(x.max())
    return peak + float(np.log(np.exp(x - peak).sum()))
