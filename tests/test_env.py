import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import meas_rl  # noqa: F401  (registers AdaptiveMeasure-v0)
from meas_rl import env as env_mod
from meas_rl.baselines import InfoGainStopPolicy, SweepPolicy
from meas_rl.env import AdaptiveMeasureEnv


def test_passes_gymnasium_checker():
    check_env(AdaptiveMeasureEnv(), skip_render_check=True)


def test_registered_id_builds():
    import gymnasium as gym

    e = gym.make("AdaptiveMeasure-v0")
    obs, _ = e.reset(seed=0)
    assert obs.shape == (9,)


def _run(env, actions, seed):
    obs, _ = env.reset(seed=seed)
    out = [obs]
    for a in actions:
        obs, r, term, trunc, info = env.step(a)
        out.append((obs, r, term, trunc))
        if term or trunc:
            break
    return out


def test_same_seed_same_actions_is_bitwise_reproducible():
    rng = np.random.default_rng(0)
    actions = rng.uniform(-1, 1, size=(40, 3))
    actions[:, 2] = -1.0
    a = _run(AdaptiveMeasureEnv(), actions, seed=7)
    b = _run(AdaptiveMeasureEnv(), actions, seed=7)
    assert len(a) == len(b)
    for x, y in zip(a[1:], b[1:]):
        assert np.array_equal(x[0], y[0]) and x[1] == y[1]


def test_shaping_sums_to_constant_regardless_of_policy():
    """With cost switched off and gamma = 1, every episode's shaping adds up
    to ``info_weight * prior_entropy``. Only the terminal +/- bonus differs."""
    for policy in (SweepPolicy(10, 1.0), SweepPolicy(30, 0.3)):
        env = AdaptiveMeasureEnv(cost_weight=0.0, info_weight=1.0, shaping_gamma=1.0)
        env.reset(seed=3)
        h0 = env.belief.entropy()
        policy.reset()
        total = 0.0
        while True:
            _, r, term, trunc, info = env.step(policy.act(env))
            total += r
            if term or trunc:
                break
        terminal = env_mod.SUCCESS_BONUS if info["success"] else -env_mod.MISS_PENALTY
        assert total - terminal == pytest.approx(h0, rel=1e-9)


def test_running_out_of_budget_is_a_termination_not_a_truncation():
    """Regression test for a real bug. Budget exhaustion used to be flagged
    ``truncated``, so Stable-Baselines3 bootstrapped ``gamma * V(s_final)`` on
    top of the terminal bonus. Every agent learned to ride the budget at
    maximum dwell and never stop on its own."""
    env = AdaptiveMeasureEnv()
    env.reset(seed=0)
    slow = np.array([0.0, 1.0, -1.0])  # maximum dwell, never stop
    while True:
        _, _, term, trunc, info = env.step(slow)
        if term or trunc:
            break
    assert term and not trunc and info["forced_stop"]
    assert info["time_used"] >= env_mod.TIME_BUDGET


def test_sb3_does_not_bootstrap_a_budget_stop():
    from stable_baselines3.common.env_util import make_vec_env

    venv = make_vec_env(AdaptiveMeasureEnv, n_envs=1, seed=0)
    venv.reset()
    slow = np.array([[0.0, 1.0, -1.0]])
    for _ in range(100):
        _, _, dones, infos = venv.step(slow)
        if dones[0]:
            break
    assert dones[0] and infos[0]["forced_stop"]
    assert not infos[0].get("TimeLimit.truncated", False)


def test_info_gain_baseline_still_solves_the_task():
    """Regression guard: the strongest hand-written baseline must keep working."""
    env = AdaptiveMeasureEnv()
    policy = InfoGainStopPolicy(dwell=3.0)
    wins = 0
    for k in range(20):
        env.reset(seed=50_000 + k)
        policy.reset()
        while True:
            _, _, term, trunc, info = env.step(policy.act(env))
            if term or trunc:
                break
        wins += info["success"]
    assert wins >= 18


def test_discrete_mode_passes_checker_and_decodes():
    env = AdaptiveMeasureEnv(action_mode="discrete")
    check_env(env, skip_render_check=True)
    env.reset(seed=0)
    freq, dwell, stop = env.decode(np.array([0, 0, 0]))
    assert freq == pytest.approx(env_mod.physics.F_MIN)
    assert dwell == pytest.approx(env_mod.physics.DWELL_MIN)
    assert stop is False
    freq, dwell, stop = env.decode(np.array([env_mod.N_FREQ_BINS - 1, env_mod.N_DWELL_LEVELS - 1, 1]))
    assert freq == pytest.approx(env_mod.physics.F_MAX)
    assert dwell == pytest.approx(env_mod.physics.DWELL_MAX)
    assert stop is True


def test_stopping_immediately_ends_the_episode_as_a_termination():
    env = AdaptiveMeasureEnv(action_mode="discrete")
    env.reset(seed=1)
    _, _, term, trunc, info = env.step(np.array([0, 0, 1]))
    assert term and not trunc and not info["forced_stop"]
    assert info["n_measurements"] == 0


def test_dwell_stop_mode_measures_where_information_gain_is_highest():
    from meas_rl.belief import best_frequency

    env = AdaptiveMeasureEnv(action_mode="dwell_stop")
    check_env(env, skip_render_check=True)
    env.reset(seed=4)
    expected = best_frequency(env.belief, env.dwell_levels[3])
    env.step(np.array([3, 0]))
    assert env.last_freq == pytest.approx(expected)
    assert env.last_dwell == pytest.approx(env.dwell_levels[3])
    _, _, term, _, info = env.step(np.array([0, 1]))
    assert term and info["n_measurements"] == 1


def test_marginal_observation_carries_normalised_belief():
    env = AdaptiveMeasureEnv(obs_mode="marginal")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (9 + env_mod.N_F0_HIST + env_mod.N_LOGQ_HIST,)
    assert obs[9 : 9 + env_mod.N_F0_HIST].sum() == pytest.approx(1.0, abs=1e-5)
    assert obs[9 + env_mod.N_F0_HIST :].sum() == pytest.approx(1.0, abs=1e-5)
