"""Tests for CategoricalInitial and StationaryTransitions."""
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from dynamax.hmm.components.initial import CategoricalInitial
from dynamax.hmm.components.transitions import StationaryTransitions


# ---- CategoricalInitial tests (5) ----

def test_initial_probs_on_simplex():
    """Initial probs sum to 1 and are all non-negative."""
    K = 4
    probs = jnp.ones(K) / K
    comp = CategoricalInitial(probs)
    p = comp.probs
    assert jnp.allclose(p.sum(), 1.0, atol=1e-6)
    assert jnp.all(p >= 0)


def test_initial_log_prior_finite():
    """log_prior returns a finite scalar."""
    K = 3
    probs = jnp.array([0.5, 0.3, 0.2])
    comp = CategoricalInitial(probs, concentration=2.0)
    lp = comp.log_prior()
    assert jnp.isfinite(lp)
    assert lp.ndim == 0


def test_initial_m_step_matches_dirichlet_mode():
    """M-step output matches Dirichlet(concentration + counts).mode()."""
    K = 3
    probs = jnp.ones(K) / K
    concentration = 2.0
    comp = CategoricalInitial(probs, concentration=concentration)

    batch_stats = jnp.array([[10.0, 5.0, 3.0], [8.0, 6.0, 4.0]])
    new_comp, _ = comp.m_step(batch_stats, None)

    expected_counts = batch_stats.sum(axis=0)
    expected_probs = tfd.Dirichlet(concentration * jnp.ones(K) + expected_counts).mode()
    assert jnp.allclose(new_comp.probs, expected_probs, atol=1e-5)


def test_initial_ldj_finite():
    """log_det_jacobian is finite."""
    K = 4
    probs = jnp.ones(K) / K
    comp = CategoricalInitial(probs)
    ldj = comp.log_det_jacobian()
    assert jnp.isfinite(ldj)


def test_initial_frozen_m_step_noop():
    """Frozen parameter does not change during m_step."""
    K = 3
    probs = jnp.array([0.6, 0.3, 0.1])
    comp = CategoricalInitial(probs, trainable=False)
    batch_stats = jnp.array([[10.0, 5.0, 3.0]])
    new_comp, _ = comp.m_step(batch_stats, None)
    assert jnp.allclose(new_comp.probs, comp.probs)


# ---- StationaryTransitions tests (7) ----

def test_trans_rows_sum_to_one():
    """Each row of the transition matrix sums to 1."""
    K = 3
    tm = jnp.eye(K) * 0.8 + jnp.ones((K, K)) * 0.2 / K
    tm = tm / tm.sum(axis=1, keepdims=True)
    comp = StationaryTransitions(tm)
    result = comp.transition_matrix()
    row_sums = result.sum(axis=1)
    assert jnp.allclose(row_sums, jnp.ones(K), atol=1e-6)
    assert jnp.all(result >= 0)


def test_trans_log_prior_finite():
    """log_prior returns a finite scalar."""
    K = 3
    tm = jnp.ones((K, K)) / K
    comp = StationaryTransitions(tm, concentration=2.0)
    lp = comp.log_prior()
    assert jnp.isfinite(lp)
    assert lp.ndim == 0


def test_trans_m_step_matches_dirichlet_mode():
    """M-step output matches per-row Dirichlet MAP."""
    K = 3
    tm = jnp.ones((K, K)) / K
    concentration = 1.5
    comp = StationaryTransitions(tm, concentration=concentration)

    batch_stats = jnp.stack([
        jnp.array([[10, 2, 1], [3, 8, 2], [1, 3, 9]]),
        jnp.array([[8, 3, 2], [2, 7, 3], [2, 2, 8]])
    ], axis=0).astype(float)

    new_comp, _ = comp.m_step(batch_stats, None)

    expected_counts = batch_stats.sum(axis=0)
    conc_matrix = concentration * jnp.ones((K, K))
    expected_tm = tfd.Dirichlet(conc_matrix + expected_counts).mode()
    assert jnp.allclose(new_comp.transition_matrix(), expected_tm, atol=1e-5)


def test_trans_stickiness_increases_diagonal():
    """Stickiness parameter increases diagonal concentration."""
    K = 3
    tm = jnp.ones((K, K)) / K
    comp_no_stick = StationaryTransitions(tm, concentration=1.1, stickiness=0.0)
    comp_sticky = StationaryTransitions(tm, concentration=1.1, stickiness=5.0)
    diff = comp_sticky._concentration - comp_no_stick._concentration
    assert jnp.allclose(diff, 5.0 * jnp.eye(K))


def test_trans_ldj_finite():
    """log_det_jacobian is finite."""
    K = 4
    tm = jnp.ones((K, K)) / K
    comp = StationaryTransitions(tm)
    ldj = comp.log_det_jacobian()
    assert jnp.isfinite(ldj)


def test_trans_frozen_m_step_noop():
    """Frozen parameter does not change during m_step."""
    K = 3
    tm = jnp.ones((K, K)) / K
    comp = StationaryTransitions(tm, trainable=False)
    batch_stats = jnp.ones((2, K, K)) * 5.0
    new_comp, _ = comp.m_step(batch_stats, None)
    assert jnp.allclose(new_comp.transition_matrix(), comp.transition_matrix())


def test_trans_compute_transition_matrices_stationary():
    """compute_transition_matrices returns (K, K) for stationary case."""
    K = 3
    tm = jnp.ones((K, K)) / K
    comp = StationaryTransitions(tm)
    result = comp.compute_transition_matrices()
    assert result.shape == (K, K)
    assert jnp.allclose(result, comp.transition_matrix())
