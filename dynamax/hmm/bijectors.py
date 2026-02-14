"""Custom bijectors for dynamax v2.

StableCholeskyOuterProduct replaces v1's RealToPSDBijector with overflow protection
and a 2D unconstrained representation (lower triangular matrix).
"""
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.bijectors as tfb


class StableCholeskyOuterProduct(tfb.Bijector):
    """Maps unconstrained lower triangular matrix to SPD matrix.

    Forward: x (lower tri with real diagonal) -> L @ L.T
        where diag(L) = exp(clip(diag(x), -20, 20))
        and off-diagonal of L = off-diagonal of x

    Inverse: Sigma (SPD) -> lower tri with log diagonal
        L = cholesky(Sigma)
        return tril(L, k=-1) + diag(log(diag(L)))

    The clipping prevents exp overflow while maintaining
    differentiability within [-20, 20]. exp(-20) ~ 2e-9,
    exp(20) ~ 4.8e8 -- covers any practical covariance range.

    Note on unconstrained shape:
        V1 used RealToPSDBijector with flat vector shape (D*(D+1)/2,).
        V2 uses (D, D) lower triangular directly. Constrained shape is (D, D) for both.
    """
    _CLIP_MIN = -20.0
    _CLIP_MAX = 20.0

    def __init__(self, validate_args=False, name='stable_cholesky_outer_product'):
        super().__init__(
            forward_min_event_ndims=2,
            inverse_min_event_ndims=2,
            validate_args=validate_args,
            name=name,
        )

    def _forward(self, x):
        n = x.shape[-1]
        diag_raw = jnp.diagonal(x, axis1=-2, axis2=-1)
        diag_clipped = jnp.clip(diag_raw, self._CLIP_MIN, self._CLIP_MAX)
        diag_positive = jnp.exp(diag_clipped)
        # Build L: off-diagonal from x, positive diagonal from exp
        # Use batch-compatible diagonal construction: (..., D, 1) * (D, D) -> (..., D, D)
        L = jnp.tril(x, k=-1) + diag_positive[..., None] * jnp.eye(n)
        return L @ jnp.swapaxes(L, -2, -1)

    def _inverse(self, y):
        n = y.shape[-1]
        L = jnp.linalg.cholesky(y)
        diag_positive = jnp.diagonal(L, axis1=-2, axis2=-1)
        diag_log = jnp.log(jnp.maximum(diag_positive, 1e-10))
        return jnp.tril(L, k=-1) + diag_log[..., None] * jnp.eye(n)

    def _forward_log_det_jacobian(self, x):
        """Log-det-Jacobian of the forward map x -> L L^T.

        Two contributions per diagonal element i (0-indexed):
          1. exp transform: contributes +1 * x_ii
          2. L @ L.T outer product: contributes +(D - i) * x_ii
          3. Constant: D * log(2) from the Cholesky outer product Jacobian
        Combined: sum_i (D - i + 1) * x_ii + D * log(2)

        Returns scalar for single (D, D) event, or (...) for batched (..., D, D) input.
        """
        n = x.shape[-1]
        diag_raw = jnp.diagonal(x, axis1=-2, axis2=-1)
        diag_clipped = jnp.clip(diag_raw, self._CLIP_MIN, self._CLIP_MAX)
        multipliers = n - jnp.arange(n) + 1
        return jnp.sum(multipliers * diag_clipped, axis=-1) + n * jnp.log(2.0)


def simplex_bijector():
    """Bijector constraining to the probability simplex."""
    return tfb.SoftmaxCentered()


def positive_bijector():
    """Bijector constraining to positive reals."""
    return tfb.Softplus()


def unit_interval_bijector():
    """Bijector constraining to [0, 1]."""
    return tfb.Sigmoid()


def spd_bijector():
    """Bijector constraining to symmetric positive definite matrices."""
    return StableCholeskyOuterProduct()
