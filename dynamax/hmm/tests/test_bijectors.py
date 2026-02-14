"""Tests for StableCholeskyOuterProduct bijector."""
import jax
import jax.numpy as jnp

from dynamax.hmm.bijectors import StableCholeskyOuterProduct


def test_forward_produces_spd():
    """Output is symmetric positive definite."""
    bij = StableCholeskyOuterProduct()
    key = jax.random.PRNGKey(0)
    x = jnp.tril(jax.random.normal(key, (4, 4)))
    sigma = bij.forward(x)
    # Symmetric
    assert jnp.allclose(sigma, sigma.T, atol=1e-10)
    # Positive definite
    eigvals = jnp.linalg.eigvalsh(sigma)
    assert jnp.all(eigvals > 0)


def test_no_overflow_extreme():
    """Clipping prevents overflow for large unconstrained values."""
    bij = StableCholeskyOuterProduct()
    x = jnp.array([[50.0, 0.0], [1.0, 50.0]])
    sigma = bij.forward(x)
    assert jnp.all(jnp.isfinite(sigma))
    assert jnp.all(jnp.linalg.eigvalsh(sigma) > 0)


def test_round_trip_well_conditioned():
    """forward(inverse(Sigma)) ~ Sigma for well-conditioned matrices."""
    bij = StableCholeskyOuterProduct()
    D = 4
    key = jax.random.PRNGKey(0)
    L = jnp.eye(D) + 0.1 * jnp.tril(jax.random.normal(key, (D, D)))
    sigma = L @ L.T
    x = bij.inverse(sigma)
    sigma_rt = bij.forward(x)
    assert jnp.allclose(sigma, sigma_rt, atol=1e-6)


def test_round_trip_ill_conditioned():
    """Characterize round-trip error for kappa ~ 1e8."""
    bij = StableCholeskyOuterProduct()
    D = 4
    diag = jnp.array([1.0, 1e-2, 1e-4, 1e-8])
    key = jax.random.PRNGKey(42)
    Q, _ = jnp.linalg.qr(jax.random.normal(key, (D, D)))
    sigma = Q @ jnp.diag(diag) @ Q.T
    x = bij.inverse(sigma)
    sigma_rt = bij.forward(x)
    rel_error = jnp.linalg.norm(sigma - sigma_rt) / jnp.linalg.norm(sigma)
    assert jnp.isfinite(rel_error)
    assert rel_error < 1e-1  # Loose bound for ill-conditioned


def test_ldj_finite_difference():
    """Analytic log-det-Jacobian matches finite differences.

    This is the most important test in the entire bijector suite.
    If this fails, HMC will sample from the wrong posterior.

    Computes the Jacobian from free parameters (lower-triangular entries of x)
    to the unique entries of Sigma (lower-triangular entries), then takes
    slogdet of the square Jacobian for the numerical reference.
    """
    bij = StableCholeskyOuterProduct()
    D = 3
    key = jax.random.PRNGKey(0)
    x = jnp.tril(jax.random.normal(key, (D, D)))

    # Analytic
    analytic_ldj = bij.forward_log_det_jacobian(x, event_ndims=2)

    # Numerical: Jacobian from free params to unique output entries
    tril_indices = jnp.tril_indices(D)
    x_free = x[tril_indices]

    def forward_unique(x_free):
        x_mat = jnp.zeros((D, D)).at[tril_indices].set(x_free)
        Sigma = bij.forward(x_mat)
        return Sigma[tril_indices]

    J = jax.jacobian(forward_unique)(x_free)
    _, numerical_ldj = jnp.linalg.slogdet(J)

    rel_error = jnp.abs(analytic_ldj - numerical_ldj) / jnp.abs(numerical_ldj)
    assert rel_error < 1e-3, (
        f"LDJ mismatch: analytic={analytic_ldj}, numerical={numerical_ldj}, "
        f"rel_error={rel_error}"
    )


def test_ldj_finite():
    """LDJ is finite for typical inputs."""
    bij = StableCholeskyOuterProduct()
    key = jax.random.PRNGKey(1)
    x = jnp.tril(jax.random.normal(key, (5, 5)))
    ldj = bij.forward_log_det_jacobian(x, event_ndims=2)
    assert jnp.isfinite(ldj)


def test_inverse_identity():
    """Inverse of identity: diagonal should be log(1) = 0, off-diag 0."""
    bij = StableCholeskyOuterProduct()
    D = 3
    x = bij.inverse(jnp.eye(D))
    assert jnp.allclose(x, jnp.zeros((D, D)), atol=1e-6)


def test_with_constrained_parameter():
    """StableCholeskyOuterProduct works inside ConstrainedParameter."""
    from dynamax.hmm.parameters import ConstrainedParameter

    D = 3
    key = jax.random.PRNGKey(0)
    L = jnp.eye(D) + 0.1 * jnp.tril(jax.random.normal(key, (D, D)))
    sigma = L @ L.T

    cp = ConstrainedParameter.from_constrained(
        sigma, StableCholeskyOuterProduct(), trainable=True
    )

    # Constrained value matches original
    assert jnp.allclose(cp.value, sigma, atol=1e-5)
    # Unconstrained is lower triangular shaped
    assert cp.unconstrained.shape == (D, D)
    # LDJ is finite
    assert jnp.isfinite(cp.log_det_jacobian())
