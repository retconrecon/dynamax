"""Concrete implementation of HMM initial state distributions."""
import jax.numpy as jnp
from jaxtyping import Float, Array
import equinox as eqx
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb

from dynamax.hmm.components.base import HMMInitialState
from dynamax.hmm.parameters import ConstrainedParameter


class CategoricalInitial(HMMInitialState):
    """Standard HMM initial state with Dirichlet prior.

    Stores probabilities via SoftmaxCentered bijector (K-1 unconstrained -> K constrained).
    M-step uses Dirichlet MAP: mode of Dirichlet(concentration + expected_counts).
    """
    _probs: ConstrainedParameter
    _concentration: Float[Array, " K"]
    _num_states: int = eqx.field(static=True)

    @classmethod
    def create(cls, num_states, *, concentration=1.1):
        """Factory method with uniform initialization.

        Args:
            num_states: number of discrete states K
            concentration: Dirichlet concentration parameter

        Returns:
            CategoricalInitial with uniform initial probabilities
        """
        probs = jnp.ones(num_states) / num_states
        return cls(probs, concentration=concentration)

    def __init__(self, probs, concentration=1.1, trainable=True):
        """
        Args:
            probs: initial state probabilities, shape (K,), must sum to 1
            concentration: Dirichlet concentration parameter (scalar or array of shape K)
            trainable: whether probs are updated during fitting
        """
        K = probs.shape[0]
        self._num_states = K
        self._concentration = concentration * jnp.ones(K)
        self._probs = ConstrainedParameter.from_constrained(
            probs, tfb.SoftmaxCentered(), trainable=trainable
        )

    @property
    def num_states(self):
        return self._num_states

    @property
    def probs(self):
        return self._probs.value

    def log_prior(self):
        return tfd.Dirichlet(self._concentration).log_prob(self.probs)

    def log_det_jacobian(self):
        return self._probs.log_det_jacobian()

    def m_step(self, batch_stats, m_step_state):
        """M-step: MAP estimate of initial probs under Dirichlet prior.

        Args:
            batch_stats: expected initial state counts, shape (batch, K)
            m_step_state: optimizer state (unused, passed through)

        Returns:
            (new_self, m_step_state)
        """
        if not self._probs.trainable:
            return self, m_step_state
        if self._num_states == 1:
            new_probs = jnp.array([1.0])
        else:
            expected_counts = batch_stats.sum(axis=0)
            new_probs = tfd.Dirichlet(self._concentration + expected_counts).mode()
        new_probs_param = ConstrainedParameter.from_constrained(
            new_probs, self._probs.bijector, trainable=self._probs.trainable
        )
        new_self = eqx.tree_at(lambda s: s._probs, self, new_probs_param)
        return new_self, m_step_state
