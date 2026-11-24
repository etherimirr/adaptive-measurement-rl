import numpy as np
import pytest

from meas_rl import physics
from meas_rl.belief import Belief


def test_peak_is_one_at_resonance():
    assert physics.response(2.4, 2.4, 200.0) == pytest.approx(1.0, abs=1e-12)


def test_half_power_points_sit_half_a_linewidth_away():
    f0, q = 2.4, 200.0
    half = physics.linewidth(f0, q) / 2
    for f in (f0 - half, f0 + half):
        assert physics.response(f, f0, q) == pytest.approx(1 / np.sqrt(2), abs=1e-12)


def test_noise_follows_square_root_law():
    rng = np.random.default_rng(0)
    for dwell in (0.25, 1.0, 4.0):
        draws = [physics.measure(2.3, dwell, 2.4, 300.0, rng) for _ in range(20000)]
        clean = physics.response(2.3, 2.4, 300.0)
        assert np.std(np.array(draws) - clean) == pytest.approx(
            physics.SIGMA_REF / np.sqrt(dwell), rel=0.03
        )


def test_prior_is_uniform_and_normalised():
    b = Belief()
    assert b.posterior.sum() == pytest.approx(1.0)
    assert b.entropy() == pytest.approx(np.log(b.n_cells))


def test_posterior_concentrates_on_the_true_device():
    rng = np.random.default_rng(1)
    f0, q = 2.4123, 180.0
    b = Belief()
    for f in np.linspace(physics.F_MIN, physics.F_MAX, 400):
        b.update(f, 2.0, physics.measure(f, 2.0, f0, q, rng))
    f0_hat, q_hat = b.estimate()
    spacing = b.f0_axis[1] - b.f0_axis[0]
    assert abs(f0_hat - f0) < spacing
    assert abs(q_hat - q) / q < 0.05
    assert b.posterior.sum() == pytest.approx(1.0)


def test_readings_reduce_entropy_on_average():
    rng = np.random.default_rng(2)
    drops = []
    for _ in range(20):
        f0, q = physics.sample_device(rng)
        b = Belief()
        h0 = b.entropy()
        for f in np.linspace(physics.F_MIN, physics.F_MAX, 30):
            b.update(f, 1.0, physics.measure(f, 1.0, f0, q, rng))
        drops.append(h0 - b.entropy())
    assert np.mean(drops) > 1.0


def test_info_gain_is_highest_where_candidates_disagree():
    b = Belief()
    freqs = np.array([physics.F_MIN, 2.40, physics.F_MAX])
    eig = b.expected_info_gain(freqs, 1.0)
    # Under the prior every device's peak sits in 2.35-2.45 GHz, so the band
    # centre is where candidate devices disagree and the band edges are not.
    assert eig[1] > eig[0] and eig[1] > eig[2]
