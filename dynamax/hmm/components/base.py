"""Abstract base classes for HMM components.

Defines the contract that all concrete initial state, transition,
and emission components must implement. The HMM model class composes
one of each.
"""
from abc import abstractmethod
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
import equinox as eqx


class HMMInitialState(eqx.Module):
    """Abstract base class for HMM initial state distributions.

    Concrete subclasses must implement:
        - num_states (property)
        - probs (property): constrained initial state probabilities on the simplex
        - log_prior(): log prior probability of the parameters
        - m_step(): returns (new_self, new_m_step_state)
    """

    @property
    @abstractmethod
    def num_states(self) -> int:
        """Number of discrete states K."""
        raise NotImplementedError

    @property
    @abstractmethod
    def probs(self) -> Float[Array, " K"]:
        """Initial state probabilities (constrained, on simplex)."""
        raise NotImplementedError

    @property
    def log_probs(self) -> Float[Array, " K"]:
        """Log initial state probabilities (with epsilon for numerical safety)."""
        return jnp.log(self.probs + 1e-10)

    @abstractmethod
    def log_prior(self) -> float:
        """Log prior probability of the initial state parameters."""
        raise NotImplementedError

    @abstractmethod
    def m_step(self, batch_stats, m_step_state):
        """Perform the M-step update.

        Args:
            batch_stats: sufficient statistics aggregated over batch of sequences
            m_step_state: optimizer state or None

        Returns:
            (new_self, new_m_step_state): updated component and optimizer state
        """
        raise NotImplementedError

    def log_det_jacobian(self):
        """Log-det-Jacobian of all constrained parameters. Override in subclasses."""
        return jnp.array(0.0)

    def collect_suff_stats(self, posterior, inputs=None):
        """Collect sufficient statistics from the posterior.

        Default: smoothed probability of initial state, shape [K].

        Args:
            posterior: HMMPosterior from the inference engine
            inputs: optional inputs (unused in default)

        Returns:
            sufficient statistics for the M-step
        """
        return posterior.smoothed_probs[0]


class HMMTransitions(eqx.Module):
    """Abstract base class for HMM transition distributions.

    Concrete subclasses must implement:
        - num_states (property)
        - transition_matrix(): returns the K x K transition matrix
        - log_prior(): log prior probability of the parameters
        - m_step(): returns (new_self, new_m_step_state)
    """

    @property
    @abstractmethod
    def num_states(self) -> int:
        """Number of discrete states K."""
        raise NotImplementedError

    @abstractmethod
    def transition_matrix(self, inputs=None) -> Float[Array, "K K"]:
        """Transition matrix A where A_jk = P(z_t=k | z_{t-1}=j).

        Args:
            inputs: optional inputs at the current time step

        Returns:
            K x K transition matrix (rows sum to 1)
        """
        raise NotImplementedError

    @abstractmethod
    def log_prior(self) -> float:
        """Log prior probability of the transition parameters."""
        raise NotImplementedError

    @abstractmethod
    def m_step(self, batch_stats, m_step_state):
        """Perform the M-step update.

        Args:
            batch_stats: sufficient statistics aggregated over batch of sequences
            m_step_state: optimizer state or None

        Returns:
            (new_self, new_m_step_state): updated component and optimizer state
        """
        raise NotImplementedError

    def compute_transition_matrices(self, inputs=None):
        """Compute transition matrices, possibly time-varying.

        Args:
            inputs: None for stationary, or Array[T-1, ...] for time-varying

        Returns:
            Array[K, K] if stationary, or Array[T-1, K, K] if time-varying
        """
        if inputs is None:
            return self.transition_matrix()
        else:
            return jax.vmap(self.transition_matrix)(inputs)

    def log_det_jacobian(self):
        """Log-det-Jacobian of all constrained parameters. Override in subclasses."""
        return jnp.array(0.0)

    def collect_suff_stats(self, posterior, inputs=None):
        """Collect sufficient statistics from the posterior.

        Default: returns posterior.trans_probs directly.
        For stationary transitions, this is already shape [K, K] (summed by inference).
        For time-varying, this is shape [T-1, K, K].

        Args:
            posterior: HMMPosterior from the inference engine
            inputs: optional inputs (unused in default)

        Returns:
            sufficient statistics for the M-step
        """
        return posterior.trans_probs


class HMMEmissions(eqx.Module):
    """Abstract base class for HMM emission distributions.

    Concrete subclasses must implement:
        - num_states (property)
        - emission_dim (property)
        - distribution(state, inputs): returns a TFP distribution for state k
        - log_prior(): log prior probability of the parameters
        - collect_suff_stats(posterior, observations, inputs): emission-specific stats
        - m_step(batch_stats, observations, inputs, m_step_state): returns (new_self, new_state)
    """

    @property
    @abstractmethod
    def num_states(self) -> int:
        """Number of discrete states K."""
        raise NotImplementedError

    @property
    @abstractmethod
    def emission_dim(self) -> int:
        """Dimensionality of the emission vector."""
        raise NotImplementedError

    @abstractmethod
    def distribution(self, state, inputs=None):
        """Return the emission distribution for a given state.

        Args:
            state: discrete state index k
            inputs: optional inputs at the current time step

        Returns:
            a TFP distribution p(y_t | z_t=k)
        """
        raise NotImplementedError

    @abstractmethod
    def log_prior(self) -> float:
        """Log prior probability of the emission parameters."""
        raise NotImplementedError

    @abstractmethod
    def collect_suff_stats(self, posterior, observations, inputs=None):
        """Collect sufficient statistics for the M-step.

        Args:
            posterior: HMMPosterior from the inference engine
            observations: observed emissions, shape [T, D]
            inputs: optional inputs

        Returns:
            sufficient statistics for the M-step
        """
        raise NotImplementedError

    @abstractmethod
    def m_step(self, batch_stats, observations=None, inputs=None, m_step_state=None):
        """Perform the M-step update.

        Args:
            batch_stats: sufficient statistics aggregated over batch of sequences
            observations: observed emissions (may be needed for gradient-based updates)
            inputs: optional inputs
            m_step_state: optimizer state or None

        Returns:
            (new_self, new_m_step_state): updated component and optimizer state
        """
        raise NotImplementedError

    def log_prob(self, observation, inputs=None) -> Float[Array, " K"]:
        """Log probability of observation under each state.

        Args:
            observation: single emission y_t, shape [D]
            inputs: optional inputs at time t

        Returns:
            log p(y_t | z_t=k) for k=0,...,K-1, shape [K]
        """
        return jax.vmap(
            lambda k: self.distribution(k, inputs).log_prob(observation)
        )(jnp.arange(self.num_states))

    def compute_log_likelihoods(self, observations, inputs=None, mask=None):
        """Compute log-likelihoods for all timesteps and states.

        Args:
            observations: emissions, shape [T, D]
            inputs: optional inputs, shape [T, ...] or None
            mask: optional binary mask, shape [T], zeros out padded timesteps

        Returns:
            log p(y_t | z_t=k) for all t,k, shape [T, K]
        """
        if inputs is None:
            log_liks = jax.vmap(self.log_prob)(observations)
        else:
            log_liks = jax.vmap(self.log_prob)(observations, inputs)
        if mask is not None:
            log_liks = log_liks * mask[:, None]
        return log_liks

    def log_det_jacobian(self):
        """Log-det-Jacobian of all constrained parameters. Override in subclasses."""
        return jnp.array(0.0)
