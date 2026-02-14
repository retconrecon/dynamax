"""Concrete implementation of HMM transition distributions."""
import jax.numpy as jnp
from jaxtyping import Float, Array
import equinox as eqx
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb

from dynamax.hmm.components.base import HMMTransitions
from dynamax.hmm.parameters import ConstrainedParameter


class StationaryTransitions(HMMTransitions):
    r"""Standard stationary HMM transitions with per-row Dirichlet prior.

    Prior on each row of transition matrix A:
        A_k ~ Dir(beta * 1_K + kappa * e_k)
    where beta is the concentration and kappa is the stickiness.

    Stores transition matrix via SoftmaxCentered bijector
    (K x (K-1) unconstrained -> K x K constrained).
    """
    _transition_matrix: ConstrainedParameter
    _concentration: Float[Array, "K K"]
    _num_states: int = eqx.field(static=True)

    @classmethod
    def create(cls, num_states, *, concentration=1.1, stickiness=0.0):
        """Factory method with uniform initialization.

        Args:
            num_states: number of discrete states K
            concentration: Dirichlet concentration parameter
            stickiness: extra concentration on self-transitions

        Returns:
            StationaryTransitions with uniform transition matrix
        """
        transition_matrix = jnp.ones((num_states, num_states)) / num_states
        return cls(transition_matrix, concentration=concentration, stickiness=stickiness)

    def __init__(self, transition_matrix, concentration=1.1, stickiness=0.0, trainable=True):
        """
        Args:
            transition_matrix: K x K transition matrix, rows sum to 1
            concentration: Dirichlet concentration (scalar or K x K array)
            stickiness: extra concentration on diagonal (encourages self-transitions)
            trainable: whether the transition matrix is updated during fitting
        """
        K = transition_matrix.shape[0]
        self._num_states = K
        self._concentration = concentration * jnp.ones((K, K)) + stickiness * jnp.eye(K)
        self._transition_matrix = ConstrainedParameter.from_constrained(
            transition_matrix, tfb.SoftmaxCentered(), trainable=trainable
        )

    @property
    def num_states(self):
        return self._num_states

    def transition_matrix(self, inputs=None):
        return self._transition_matrix.value

    def log_prior(self):
        return tfd.Dirichlet(self._concentration).log_prob(self._transition_matrix.value).sum()

    def log_det_jacobian(self):
        return self._transition_matrix.log_det_jacobian()

    def m_step(self, batch_stats, m_step_state):
        """M-step: MAP estimate of transition matrix under per-row Dirichlet prior.

        Args:
            batch_stats: expected transition counts, shape (batch, K, K)
            m_step_state: optimizer state (unused, passed through)

        Returns:
            (new_self, m_step_state)
        """
        if not self._transition_matrix.trainable:
            return self, m_step_state
        if self._num_states == 1:
            new_tm = jnp.array([[1.0]])
        else:
            expected_counts = batch_stats.sum(axis=0)
            new_tm = tfd.Dirichlet(self._concentration + expected_counts).mode()
        new_tm_param = ConstrainedParameter.from_constrained(
            new_tm, self._transition_matrix.bijector, trainable=self._transition_matrix.trainable
        )
        new_self = eqx.tree_at(lambda s: s._transition_matrix, self, new_tm_param)
        return new_self, m_step_state
