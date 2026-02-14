"""Concrete implementation of Gaussian HMM emission distributions."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
import equinox as eqx
import tensorflow_probability.substrates.jax.distributions as tfd

from dynamax.hmm.components.base import HMMEmissions
from dynamax.hmm.parameters import ConstrainedParameter
from dynamax.hmm.bijectors import StableCholeskyOuterProduct
from dynamax.utils.distributions import NormalInverseWishart, niw_posterior_update


class GaussianSufficientStats(NamedTuple):
    """Sufficient statistics for Gaussian emissions.

    Attributes:
        sum_w: sum of responsibilities per state, shape [K]
        sum_x: weighted sum of observations, shape [K, D]
        sum_xxT: weighted sum of outer products, shape [K, D, D]
    """
    sum_w: Float[Array, " K"]
    sum_x: Float[Array, "K D"]
    sum_xxT: Float[Array, "K D D"]


class GaussianEmissions(HMMEmissions):
    """Gaussian emissions: y_t | z_t=k ~ N(mu_k, Sigma_k).

    Parameters:
        _means: ConstrainedParameter with Identity bijector, shape [K, D]
        _covs: ConstrainedParameter with StableCholeskyOuterProduct bijector, shape [K, D, D]

    Prior:
        Normal-Inverse-Wishart on (mu_k, Sigma_k) for each state k
    """
    _means: ConstrainedParameter
    _covs: ConstrainedParameter
    _prior_mean: Float[Array, " D"]
    _prior_mean_concentration: float
    _prior_df: float
    _prior_scale: Float[Array, "D D"]

    def __init__(self, means, covs,
                 prior_mean=None, prior_mean_concentration=0.1,
                 prior_df=None, prior_scale=None):
        """
        Args:
            means: emission means, shape (K, D)
            covs: emission covariances, shape (K, D, D), must be SPD
            prior_mean: NIW prior mean, shape (D,). Default: zeros
            prior_mean_concentration: NIW prior concentration. Default: 0.1
            prior_df: NIW prior degrees of freedom. Default: D + 2.0
            prior_scale: NIW prior scale matrix, shape (D, D). Default: identity
        """
        D = means.shape[-1]
        self._means = ConstrainedParameter.unconstrained_param(means)
        self._covs = ConstrainedParameter.from_constrained(
            covs, StableCholeskyOuterProduct()
        )
        self._prior_mean = prior_mean if prior_mean is not None else jnp.zeros(D)
        self._prior_mean_concentration = float(prior_mean_concentration)
        self._prior_df = float(prior_df) if prior_df is not None else float(D + 2.0)
        self._prior_scale = prior_scale if prior_scale is not None else jnp.eye(D)

    @classmethod
    def create(cls, num_states, emission_dim, *, key):
        """Factory method with default initialization.

        Args:
            num_states: number of discrete states K
            emission_dim: dimensionality of emissions D
            key: JAX random key

        Returns:
            GaussianEmissions with random means and identity covariances
        """
        means = jax.random.normal(key, (num_states, emission_dim))
        covs = jnp.tile(jnp.eye(emission_dim), (num_states, 1, 1))
        return cls(means, covs)

    @property
    def num_states(self):
        return self._means.value.shape[0]

    @property
    def emission_dim(self):
        return self._means.value.shape[1]

    @property
    def means(self):
        return self._means.value

    @property
    def covariances(self):
        return self._covs.value

    def distribution(self, state, inputs=None):
        return tfd.MultivariateNormalFullCovariance(
            loc=self._means.value[state],
            covariance_matrix=self._covs.value[state],
        )

    def log_prior(self):
        prior = NormalInverseWishart(
            loc=self._prior_mean,
            mean_concentration=self._prior_mean_concentration,
            df=self._prior_df,
            scale=self._prior_scale,
        )
        return jax.vmap(
            lambda mu, sigma: prior.log_prob((sigma, mu))
        )(self._means.value, self._covs.value).sum()

    def log_det_jacobian(self):
        return self._means.log_det_jacobian() + self._covs.log_det_jacobian()

    def collect_suff_stats(self, posterior, observations, inputs=None):
        gamma = posterior.smoothed_probs  # [T, K]
        sum_w = jnp.einsum('tk->k', gamma)
        sum_x = jnp.einsum('tk,td->kd', gamma, observations)
        sum_xxT = jnp.einsum('tk,td,te->kde', gamma, observations, observations)
        return GaussianSufficientStats(sum_w, sum_x, sum_xxT)

    def m_step(self, batch_stats, observations=None, inputs=None, m_step_state=None):
        """M-step: MAP estimate of emission parameters under NIW prior.

        Args:
            batch_stats: GaussianSufficientStats with batch leading dim
            observations: unused (NIW M-step is closed-form from suff stats)
            inputs: unused
            m_step_state: optimizer state (unused, passed through)

        Returns:
            (new_self, m_step_state)
        """
        if not self._means.trainable or not self._covs.trainable:
            return self, m_step_state

        # Aggregate sufficient stats over batch
        sum_w = batch_stats.sum_w.sum(axis=0)      # [K]
        sum_x = batch_stats.sum_x.sum(axis=0)      # [K, D]
        sum_xxT = batch_stats.sum_xxT.sum(axis=0)  # [K, D, D]

        prior = NormalInverseWishart(
            loc=self._prior_mean,
            mean_concentration=self._prior_mean_concentration,
            df=self._prior_df,
            scale=self._prior_scale,
        )

        def update_state(sum_w_k, sum_x_k, sum_xxT_k):
            posterior = niw_posterior_update(prior, (sum_x_k, sum_xxT_k, sum_w_k))
            cov_mode, mean_mode = posterior.mode()
            return mean_mode, cov_mode

        new_means, new_covs = jax.vmap(update_state)(sum_w, sum_x, sum_xxT)

        # Ridge regularization for numerical stability
        new_covs = new_covs + 1e-6 * jnp.eye(self.emission_dim)

        new_means_param = ConstrainedParameter.unconstrained_param(
            new_means, trainable=self._means.trainable
        )
        new_covs_param = ConstrainedParameter.from_constrained(
            new_covs, self._covs.bijector, trainable=self._covs.trainable
        )
        new_self = eqx.tree_at(
            lambda s: (s._means, s._covs), self, (new_means_param, new_covs_param)
        )
        return new_self, m_step_state
