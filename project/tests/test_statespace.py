"""Isolation checks for the state-space realisation -- eq (30)-(34), (45), (57).

Target: the (A, B, C, D) built from per-channel FOPTD descriptions must
reproduce a direct simulation of eq 33 exactly, be stable (rho(A) < 1 for
a < 1), and feed the affine map of eq 57 consistently.
"""
import numpy as np
import pytest

from src.features.lag import foptd
from src.ident.stability import spectral_radius
from src.predict import multistep
from src.predict.statespace import ChannelSpec, build_state_space

Ts = 600.0


def test_realisation_matches_direct_foptd_simulation():
    """Exact check: from a zero state the realisation reproduces eq 33."""
    tau, delay = 7200.0, 5
    a = float(np.exp(-Ts / tau))
    ss = build_state_space([ChannelSpec("u", delay, a, 2.5)])
    rng = np.random.default_rng(0)
    u = rng.normal(size=120)
    ref = 2.5 * foptd(u, tau, delay * Ts, Ts)
    got = ss.simulate(np.zeros(ss.n), u.reshape(-1, 1))
    assert np.allclose(got, ref, atol=1e-12)


def test_initial_state_converges_to_the_true_state():
    """Warm start from history: the continuation must match a full replay."""
    tau, delay = 7200.0, 5
    a = float(np.exp(-Ts / tau))
    ss = build_state_space([ChannelSpec("u", delay, a, 2.5)])
    rng = np.random.default_rng(2)
    n_pre, H = 200, 30
    u_all = rng.normal(size=n_pre + H)
    full = ss.simulate(np.zeros(ss.n), u_all.reshape(-1, 1))
    x0 = ss.initial_state({"u": u_all[:n_pre]})
    warm = ss.simulate(x0, u_all[n_pre:].reshape(-1, 1))
    assert np.allclose(warm, full[n_pre:], atol=1e-9)


def test_pure_delay_channel_is_exact():
    ss = build_state_space([ChannelSpec("u", 3, 0.0, 1.0)])
    u = np.arange(10.0)
    y = ss.simulate(np.zeros(ss.n), u.reshape(-1, 1))
    # a = 0 -> z_{k+1} = u_{k-3}, i.e. the output at step k is u_{k-4}
    assert np.allclose(y[4:], u[:len(y) - 4])


def test_no_delay_channel_is_a_plain_first_order_lag():
    a = 0.8
    ss = build_state_space([ChannelSpec("u", 0, a, 1.0)])
    u = np.ones(50)
    y = ss.simulate(np.zeros(ss.n), u.reshape(-1, 1))
    assert y[0] == pytest.approx(0.0)          # strictly causal (ZOH numerator)
    assert y[1] == pytest.approx(1 - a)
    assert y[-1] == pytest.approx(1.0, abs=1e-4)


def test_realisation_is_stable_and_sized_correctly():
    chans = [ChannelSpec("a", 4, 0.9, 1.0), ChannelSpec("b", 0, 0.5, -2.0)]
    ss = build_state_space(chans)
    assert ss.n == (4 + 1 + 1) + (0 + 1 + 1) and ss.nu == 2
    assert spectral_radius(ss.A) < 1.0


def test_multistep_maps_agree_with_the_realisation():
    chans = [ChannelSpec("a", 3, 0.85, 1.5), ChannelSpec("b", 1, 0.6, -0.7)]
    ss = build_state_space(chans)
    H = 12
    maps = multistep.build_maps(ss.A, ss.B, ss.C, D=ss.D, H=H)
    rng = np.random.default_rng(1)
    x0 = rng.normal(size=ss.n)
    U = rng.normal(size=(H, ss.nu))
    assert np.allclose(maps.predict(x0, U.ravel(), np.zeros(0)),
                       ss.simulate(x0, U), atol=1e-10)


def test_steady_state_gain_equals_the_channel_gain():
    ss = build_state_space([ChannelSpec("u", 2, 0.9, 3.0)])
    y = ss.simulate(np.zeros(ss.n), np.ones((300, 1)))
    assert y[-1] == pytest.approx(3.0, abs=1e-6)


def test_build_state_space_guards():
    with pytest.raises(ValueError):
        build_state_space([ChannelSpec("u", -1, 0.5, 1.0)])
    with pytest.raises(ValueError):
        build_state_space([ChannelSpec("u", 1, 1.0, 1.0)])


def test_initial_state_backfills_short_history():
    ss = build_state_space([ChannelSpec("u", 5, 0.8, 1.0)])
    x = ss.initial_state({"u": np.array([7.0])})
    assert np.allclose(x, 7.0)
    assert np.isfinite(ss.initial_state({"u": np.array([np.nan])})).all()
