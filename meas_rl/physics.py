"""Forward model of the device under test and of a single noisy measurement.

The device is a single-pole resonator. Sweeping frequency across it gives the
familiar bell-shaped transmission curve: flat and near zero far from
resonance, rising to a peak at the resonant frequency ``f0``. How sharp that
peak is depends on the quality factor ``Q``: the -3 dB linewidth is ``f0 / Q``,
so a high-Q device hides in a narrow slice of the band.

A measurement is a single frequency point held for a dwell time. Holding
longer averages away more noise, with the standard square-root law, so dwell
time is the knob that trades measurement time against noise.
"""

from __future__ import annotations

import numpy as np

# Band the instrument can reach, in GHz.
F_MIN = 2.30
F_MAX = 2.50

# Device parameter ranges, sampled fresh for every episode.
F0_MIN = 2.35
F0_MAX = 2.45
Q_MIN = 50.0
Q_MAX = 500.0

# Noise standard deviation at a dwell time of 1.0 (see `noise_std`).
SIGMA_REF = 0.05

# Dwell times the agent may ask for, in milliseconds.
DWELL_MIN = 0.1
DWELL_MAX = 10.0

# Fixed cost of moving the source to a new frequency and settling, in ms.
SETTLING_COST = 0.5


def response(freq: np.ndarray | float, f0: float, q: float) -> np.ndarray | float:
    """Noiseless transmission magnitude |S21| of the resonator.

    Lorentzian line shape, normalised so the peak is 1.0 at ``freq == f0``:

        |S21(f)| = 1 / sqrt(1 + (2 Q (f - f0) / f0)^2)

    Amplitude is held at 1.0 so the unknown state is exactly two dimensional
    and the belief can be kept as an exact grid posterior. Letting amplitude
    float as a third unknown is the obvious extension.
    """
    detune = 2.0 * q * (np.asarray(freq, dtype=float) - f0) / f0
    return 1.0 / np.sqrt(1.0 + detune**2)


def noise_std(dwell: float | np.ndarray) -> float | np.ndarray:
    """Noise standard deviation after averaging for ``dwell`` milliseconds.

    Averaging independent samples shrinks the standard deviation as
    1/sqrt(n), and dwell time is proportional to n, so sigma ~ 1/sqrt(dwell).
    """
    return SIGMA_REF / np.sqrt(np.asarray(dwell, dtype=float))


def measure(
    freq: float,
    dwell: float,
    f0: float,
    q: float,
    rng: np.random.Generator,
) -> float:
    """Take one noisy reading of the device at ``freq`` held for ``dwell`` ms."""
    clean = response(freq, f0, q)
    return float(clean + rng.normal(0.0, noise_std(dwell)))


def measurement_cost(dwell: float) -> float:
    """Wall-clock cost in ms of one measurement: settling plus dwell."""
    return SETTLING_COST + float(dwell)


def sample_device(rng: np.random.Generator) -> tuple[float, float]:
    """Draw a device: f0 uniform in the band, Q log-uniform over its range."""
    f0 = float(rng.uniform(F0_MIN, F0_MAX))
    q = float(np.exp(rng.uniform(np.log(Q_MIN), np.log(Q_MAX))))
    return f0, q


def linewidth(f0: float, q: float) -> float:
    """The -3 dB linewidth f0 / Q, in GHz."""
    return f0 / q
