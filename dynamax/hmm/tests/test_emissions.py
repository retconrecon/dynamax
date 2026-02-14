"""Tests for GaussianEmissions."""
import jax
import jax.numpy as jnp

from dynamax.hmm.components.emissions import GaussianEmissions, GaussianSufficientStats


def test_gaussian_emissions_create():
    """Create produces valid emissions component."""
    em = GaussianEmissions.create(3, 4, key=jax.random.PRNGKey(0))
    assert em.means.shape == (3, 4)
    assert em.covariances.shape == (3, 4, 4)
    # Each covariance should be SPD
    for k in range(3):
        eigvals = jnp.linalg.eigvalsh(em.covariances[k])
        assert jnp.all(eigvals > 0)


def test_gaussian_emissions_log_prob_shape():
    """log_prob returns [K] for a single observation."""
    em = GaussianEmissions.create(3, 4, key=jax.random.PRNGKey(0))
    y = jnp.ones(4)
    log_probs = em.log_prob(y)
    assert log_probs.shape == (3,)
    assert jnp.all(jnp.isfinite(log_probs))


def test_gaussian_emissions_log_likelihoods_shape():
    """compute_log_likelihoods returns [T, K]."""
    em = GaussianEmissions.create(3, 4, key=jax.random.PRNGKey(0))
    T = 50
    y = jax.random.normal(jax.random.PRNGKey(1), (T, 4))
    log_liks = em.compute_log_likelihoods(y)
    assert log_liks.shape == (T, 3)
    assert jnp.all(jnp.isfinite(log_liks))


def test_gaussian_emissions_log_likelihoods_with_mask():
    """Masked timesteps have zero log-likelihood."""
    em = GaussianEmissions.create(3, 4, key=jax.random.PRNGKey(0))
    T = 50
    y = jax.random.normal(jax.random.PRNGKey(1), (T, 4))
    mask = jnp.ones(T).at[40:].set(0.0)
    log_liks = em.compute_log_likelihoods(y, mask=mask)
    assert jnp.allclose(log_liks[40:], 0.0)


def test_gaussian_emissions_log_prior_finite():
    """Log prior is finite."""
    em = GaussianEmissions.create(3, 4, key=jax.random.PRNGKey(0))
    lp = em.log_prior()
    assert jnp.isfinite(lp)


def test_gaussian_emissions_ldj_finite():
    """Log-det-Jacobian is finite."""
    em = GaussianEmissions.create(3, 4, key=jax.random.PRNGKey(0))
    ldj = em.log_det_jacobian()
    assert jnp.isfinite(ldj)


def test_gaussian_emissions_collect_suff_stats():
    """Sufficient stats have correct shapes."""
    K, D, T = 3, 4, 50
    em = GaussianEmissions.create(K, D, key=jax.random.PRNGKey(0))
    y = jax.random.normal(jax.random.PRNGKey(1), (T, D))

    # Fake posterior with random smoothed_probs (must sum to 1 per timestep)
    key = jax.random.PRNGKey(2)
    raw = jax.random.dirichlet(key, jnp.ones(K), shape=(T,))

    # Create a mock posterior (collect_suff_stats only uses smoothed_probs)
    import types
    posterior = types.SimpleNamespace(smoothed_probs=raw)

    stats = em.collect_suff_stats(posterior, y)
    assert stats.sum_w.shape == (K,)
    assert stats.sum_x.shape == (K, D)
    assert stats.sum_xxT.shape == (K, D, D)


def test_gaussian_emissions_m_step():
    """M-step produces valid updated emissions."""
    K, D = 3, 4
    em = GaussianEmissions.create(K, D, key=jax.random.PRNGKey(0))

    # Fake sufficient stats (as if from 2 sequences)
    sum_w = jnp.array([[10.0, 8.0, 12.0], [9.0, 11.0, 10.0]])  # (batch=2, K)
    sum_x = jax.random.normal(jax.random.PRNGKey(1), (2, K, D))
    sum_xxT = jnp.tile(jnp.eye(D), (2, K, 1, 1)) * 10.0  # (batch=2, K, D, D)

    stats = GaussianSufficientStats(sum_w, sum_x, sum_xxT)
    observations = jax.random.normal(jax.random.PRNGKey(2), (2, 50, D))

    new_em, _ = em.m_step(stats, observations)
    assert new_em.means.shape == (K, D)
    assert new_em.covariances.shape == (K, D, D)
    # Covariances should be SPD
    for k in range(K):
        eigvals = jnp.linalg.eigvalsh(new_em.covariances[k])
        assert jnp.all(eigvals > 0), f"State {k} covariance not SPD"


def test_gaussian_emissions_distribution():
    """distribution() returns valid TFP distribution."""
    em = GaussianEmissions.create(3, 4, key=jax.random.PRNGKey(0))
    dist = em.distribution(0)
    sample = dist.sample(seed=jax.random.PRNGKey(0))
    assert sample.shape == (4,)
    lp = dist.log_prob(sample)
    assert jnp.isfinite(lp)


def test_covariance_bijector_batching():
    """StableCholeskyOuterProduct handles (K, D, D) batch correctly."""
    from dynamax.hmm.bijectors import StableCholeskyOuterProduct
    K, D = 3, 4
    covs = jnp.tile(jnp.eye(D), (K, 1, 1))
    bij = StableCholeskyOuterProduct()
    # Inverse should handle batch
    unc = jax.vmap(bij.inverse)(covs)
    assert unc.shape == (K, D, D)
    # Forward round-trip
    covs_rt = jax.vmap(bij.forward)(unc)
    assert jnp.allclose(covs, covs_rt, atol=1e-6)
