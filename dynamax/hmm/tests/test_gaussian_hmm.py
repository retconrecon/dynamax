"""Tests for HMM base class and GaussianHMM."""
import jax
import jax.numpy as jnp
import pytest

from dynamax.hmm.models.base import HMM, GaussianHMM
from dynamax.hmm.components.initial import CategoricalInitial
from dynamax.hmm.components.transitions import StationaryTransitions
from dynamax.hmm.components.emissions import GaussianEmissions


def test_gaussian_hmm_create():
    """GaussianHMM.create produces valid model."""
    model = GaussianHMM.create(3, 4, key=jax.random.PRNGKey(0))
    assert model.num_states == 3
    assert model.emission_dim == 4
    assert isinstance(model.initial, CategoricalInitial)
    assert isinstance(model.transitions, StationaryTransitions)
    assert isinstance(model.emissions, GaussianEmissions)


def test_gaussian_hmm_log_prior_finite():
    """Log prior is finite."""
    model = GaussianHMM.create(3, 4, key=jax.random.PRNGKey(0))
    lp = model.log_prior()
    assert jnp.isfinite(lp)


def test_gaussian_hmm_log_det_jacobian_finite():
    """Log-det-Jacobian is finite."""
    model = GaussianHMM.create(3, 4, key=jax.random.PRNGKey(0))
    ldj = model.log_det_jacobian()
    assert jnp.isfinite(ldj)


def test_gaussian_hmm_marginal_log_prob():
    """Marginal log prob is finite for random data."""
    model = GaussianHMM.create(3, 4, key=jax.random.PRNGKey(0))
    y = jax.random.normal(jax.random.PRNGKey(1), (50, 4))
    ll = model.marginal_log_prob(y)
    assert jnp.isfinite(ll)
    assert ll < 0  # log prob should be negative


def test_gaussian_hmm_e_step():
    """E-step returns correct shapes."""
    K, D, T = 3, 4, 50
    model = GaussianHMM.create(K, D, key=jax.random.PRNGKey(0))
    y = jax.random.normal(jax.random.PRNGKey(1), (T, D))
    (init_stats, trans_stats, emit_stats), ll = model.e_step(y)
    assert init_stats.shape == (K,)
    assert trans_stats.shape == (K, K)
    assert emit_stats.sum_w.shape == (K,)
    assert emit_stats.sum_x.shape == (K, D)
    assert emit_stats.sum_xxT.shape == (K, D, D)
    assert jnp.isfinite(ll)


def test_gaussian_hmm_m_step():
    """M-step produces valid updated model."""
    K, D, T = 3, 4, 50
    model = GaussianHMM.create(K, D, key=jax.random.PRNGKey(0))
    y = jax.random.normal(jax.random.PRNGKey(1), (T, D))

    # Run e_step then m_step with batch dim
    stats, ll = model.e_step(y)
    # Add batch dim to match what vmap produces
    import jax.tree_util as jtu
    batch_stats = jtu.tree_map(lambda x: x[None], stats)

    new_model, _ = model.m_step(batch_stats)
    assert new_model.num_states == K
    assert new_model.emission_dim == D
    assert jnp.isfinite(new_model.log_prior())


def test_gaussian_hmm_sample():
    """Sample produces correct shapes."""
    K, D, T = 3, 4, 100
    model = GaussianHMM.create(K, D, key=jax.random.PRNGKey(0))
    states, emissions = model.sample(jax.random.PRNGKey(1), T)
    assert states.shape == (T,)
    assert emissions.shape == (T, D)
    # States should be valid indices
    assert jnp.all(states >= 0)
    assert jnp.all(states < K)


def test_gaussian_hmm_fit_em_runs():
    """fit_em runs without error and returns correct shapes."""
    K, D, T = 2, 3, 50
    model = GaussianHMM.create(K, D, key=jax.random.PRNGKey(0))
    y = jax.random.normal(jax.random.PRNGKey(1), (T, D))
    fitted, log_probs = model.fit_em(y, num_iters=3, verbose=False)
    assert log_probs.shape == (3,)
    assert fitted.num_states == K
    assert fitted.emission_dim == D
    assert jnp.all(jnp.isfinite(log_probs))


def test_gaussian_hmm_em_monotonicity():
    """EM log probs are non-decreasing (up to numerical noise)."""
    K, D, T = 2, 3, 100
    model = GaussianHMM.create(K, D, key=jax.random.PRNGKey(0))

    # Generate data from the model itself for well-specified fit
    states, y = model.sample(jax.random.PRNGKey(1), T)

    fitted, log_probs = model.fit_em(y, num_iters=20, verbose=False)

    # Check monotonicity with small tolerance for numerical noise
    diffs = jnp.diff(log_probs)
    assert jnp.all(diffs >= -1e-2), f"EM not monotonic: min diff = {diffs.min()}"


def test_gaussian_hmm_sample_fit_recovery():
    """EM recovers approximate parameters from sampled data."""
    K, D = 2, 2
    key = jax.random.PRNGKey(42)

    # Create a model with well-separated means
    k1, k2 = jax.random.split(key)
    means = jnp.array([[-3.0, -3.0], [3.0, 3.0]])
    covs = jnp.tile(0.5 * jnp.eye(D), (K, 1, 1))
    emissions = GaussianEmissions(means, covs)
    initial = CategoricalInitial(jnp.array([0.5, 0.5]))
    transitions = StationaryTransitions(
        jnp.array([[0.95, 0.05], [0.05, 0.95]])
    )
    true_model = GaussianHMM(initial, transitions, emissions)

    # Sample long sequence
    states, y = true_model.sample(k1, 500)

    # Fit from random initialization
    init_model = GaussianHMM.create(K, D, key=k2)
    fitted, log_probs = init_model.fit_em(y, num_iters=50, verbose=False)

    # Check that fitted means are close to true means (up to permutation)
    fitted_means = fitted.emissions.means
    # Sort by first coordinate to handle permutation
    true_sorted = true_model.emissions.means[jnp.argsort(true_model.emissions.means[:, 0])]
    fitted_sorted = fitted.emissions.means[jnp.argsort(fitted.emissions.means[:, 0])]
    assert jnp.allclose(true_sorted, fitted_sorted, atol=0.5), \
        f"Means not recovered:\ntrue={true_sorted}\nfitted={fitted_sorted}"


def test_gaussian_hmm_fit_em_batch():
    """fit_em handles batched emissions (N, T, D)."""
    K, D, T, N = 2, 3, 50, 4
    model = GaussianHMM.create(K, D, key=jax.random.PRNGKey(0))
    y = jax.random.normal(jax.random.PRNGKey(1), (N, T, D))
    fitted, log_probs = model.fit_em(y, num_iters=3, verbose=False)
    assert log_probs.shape == (3,)
    assert jnp.all(jnp.isfinite(log_probs))
