"""ConstrainedParameter: the core parameter abstraction for dynamax v2.

Replaces v1's (param_value, ParameterProperties) pair with an Equinox module
that stores unconstrained values internally with co-located bijectors.
"""
import jax
import jax.numpy as jnp
from jaxtyping import Array
import equinox as eqx
import tensorflow_probability.substrates.jax.bijectors as tfb


class ConstrainedParameter(eqx.Module):
    """Replaces (param_value, ParameterProperties) pair.

    Stores unconstrained representation internally.
    Bijector transforms to constrained on property access.
    trainable is static to preserve v1 recompilation semantics.

    Attributes:
        _unconstrained: The unconstrained parameter value (JAX array, pytree leaf)
        bijector: TFP bijector mapping unconstrained -> constrained (static, compile-time constant)
        trainable: Whether updated during fitting (static, compile-time constant)
    """
    _unconstrained: Array
    bijector: tfb.Bijector = eqx.field(static=True)
    trainable: bool = eqx.field(static=True, default=True)

    @property
    def value(self):
        """Constrained parameter value. Primary access point for downstream code."""
        result = self.bijector.forward(self._unconstrained)
        if not self.trainable:
            result = jax.lax.stop_gradient(result)
        return result

    @property
    def constrained(self):
        """Alias for value."""
        return self.value

    @property
    def unconstrained(self):
        """Raw unconstrained parameter value."""
        return self._unconstrained

    def log_det_jacobian(self):
        """Log-determinant of the forward Jacobian df/dx.

        Returns a scalar JAX array (never a Python float).
        Frozen parameters contribute zero.
        """
        if not self.trainable:
            return jnp.array(0.0)
        return jnp.sum(
            self.bijector.forward_log_det_jacobian(
                self._unconstrained,
                event_ndims=self._unconstrained.ndim,
            )
        )

    @classmethod
    def from_constrained(cls, value, bijector, trainable=True):
        """Create from a constrained value by inverting through the bijector."""
        unconstrained = bijector.inverse(value)
        return cls(_unconstrained=unconstrained, bijector=bijector, trainable=trainable)

    @classmethod
    def unconstrained_param(cls, value, trainable=True):
        """Create with Identity bijector (unconstrained == constrained)."""
        return cls(_unconstrained=value, bijector=tfb.Identity(), trainable=trainable)

    def freeze(self):
        """Return a new ConstrainedParameter with trainable=False."""
        return ConstrainedParameter(
            _unconstrained=self._unconstrained,
            bijector=self.bijector,
            trainable=False,
        )

    def unfreeze(self):
        """Return a new ConstrainedParameter with trainable=True."""
        return ConstrainedParameter(
            _unconstrained=self._unconstrained,
            bijector=self.bijector,
            trainable=True,
        )
