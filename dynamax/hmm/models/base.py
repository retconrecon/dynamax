"""HMM model class that composes components and runs inference + EM."""
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax, vmap
from functools import partial
import equinox as eqx
import tensorflow_probability.substrates.jax.distributions as tfd

from dynamax.hmm.components.base import HMMInitialState, HMMTransitions, HMMEmissions
from dynamax.hmm.components.initial import CategoricalInitial
from dynamax.hmm.components.transitions import StationaryTransitions
from dynamax.hmm.components.emissions import GaussianEmissions
from dynamax.utils.utils import ensure_array_has_batch_dim

# Import inference functions, handling Python 3.9 compatibility.
# dynamax.hidden_markov_model.__init__.py imports gaussian_hmm.py which uses
# X | Y type syntax (Python 3.10+). We bypass it by loading inference.py directly.
try:
    from dynamax.hidden_markov_model.inference import (
        hmm_two_filter_smoother, hmm_filter, HMMPosterior
    )
except TypeError:
    import importlib.util as _ilu
    import os as _os
    _p = _os.path.join(
        _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
        'hidden_markov_model', 'inference.py')
    _s = _ilu.spec_from_file_location('dynamax._hmm_inference', _p)
    _m = _ilu.module_from_spec(_s)
    _s.loader.exec_module(_m)
    hmm_two_filter_smoother = _m.hmm_two_filter_smoother
    hmm_filter = _m.hmm_filter
    HMMPosterior = _m.HMMPosterior


class HMM(eqx.Module):
    """Base HMM model composing initial, transition, and emission components.

    This is an Equinox module: the model IS the parameters. M-steps return
    new model instances (immutability). All parameters are stored as
    ConstrainedParameters inside the components.

    Attributes:
        initial: HMMInitialState component
        transitions: HMMTransitions component
        emissions: HMMEmissions component
    """
    initial: HMMInitialState
    transitions: HMMTransitions
    emissions: HMMEmissions

    def __init__(self, initial, transitions, emissions):
        self.initial = initial
        self.transitions = transitions
        self.emissions = emissions

    @property
    def num_states(self):
        return self.initial.num_states

    @property
    def emission_dim(self):
        return self.emissions.emission_dim

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    def log_prior(self):
        """Sum of component log priors."""
        return (self.initial.log_prior()
                + self.transitions.log_prior()
                + self.emissions.log_prior())

    def log_det_jacobian(self):
        """Sum of component log-det-Jacobians."""
        return (self.initial.log_det_jacobian()
                + self.transitions.log_det_jacobian()
                + self.emissions.log_det_jacobian())

    def _inference_args(self, emissions, inputs=None):
        """Compute the triple (initial_probs, transition_matrix, log_likelihoods)."""
        initial_probs = self.initial.probs
        transition_matrix = self.transitions.compute_transition_matrices(inputs)
        log_likelihoods = self.emissions.compute_log_likelihoods(emissions, inputs)
        return initial_probs, transition_matrix, log_likelihoods

    def marginal_log_prob(self, emissions, inputs=None):
        """Compute marginal log probability p(y_{1:T} | theta)."""
        post = hmm_filter(*self._inference_args(emissions, inputs))
        return post.marginal_loglik

    def filter(self, emissions, inputs=None):
        """Compute filtered posteriors."""
        return hmm_filter(*self._inference_args(emissions, inputs))

    def smoother(self, emissions, inputs=None):
        """Compute smoothed posteriors using two-filter smoother."""
        return hmm_two_filter_smoother(*self._inference_args(emissions, inputs))

    def e_step(self, emissions, inputs=None):
        """E-step: run two-filter smoother and collect sufficient statistics.

        Args:
            emissions: observations, shape [T, D]
            inputs: optional inputs (None for stationary models)

        Returns:
            (suff_stats, marginal_loglik) where suff_stats is a tuple of
            (initial_stats, transition_stats, emission_stats)
        """
        posterior = hmm_two_filter_smoother(*self._inference_args(emissions, inputs))
        initial_stats = self.initial.collect_suff_stats(posterior, inputs)
        transition_stats = self.transitions.collect_suff_stats(posterior, inputs)
        emission_stats = self.emissions.collect_suff_stats(posterior, emissions, inputs)
        return (initial_stats, transition_stats, emission_stats), posterior.marginal_loglik

    def m_step(self, batch_stats, m_step_state=None):
        """M-step: update all components from batched sufficient statistics.

        Args:
            batch_stats: tuple of (initial_stats, transition_stats, emission_stats)
                with batch leading dimension
            m_step_state: unused (closed-form M-steps), passed through

        Returns:
            (new_model, m_step_state)
        """
        batch_initial_stats, batch_transition_stats, batch_emission_stats = batch_stats
        new_initial, _ = self.initial.m_step(batch_initial_stats, None)
        new_transitions, _ = self.transitions.m_step(batch_transition_stats, None)
        new_emissions, _ = self.emissions.m_step(batch_emission_stats)
        new_model = eqx.tree_at(
            lambda m: (m.initial, m.transitions, m.emissions),
            self, (new_initial, new_transitions, new_emissions)
        )
        return new_model, m_step_state

    def fit_em(self, emissions, inputs=None, num_iters=50, verbose=True):
        """Fit model parameters via Expectation-Maximization.

        Replicates v1's exact control flow: ensure batch dim, JIT-compiled
        em_step with vmapped E-step, M-step, log prob collection.

        Args:
            emissions: observations, shape [T, D] or [N, T, D] (batched)
            inputs: optional inputs, same batch structure as emissions
            num_iters: number of EM iterations
            verbose: whether to show progress bar

        Returns:
            (fitted_model, log_probs) where log_probs is [num_iters]
        """
        batch_emissions = ensure_array_has_batch_dim(emissions, self.emission_shape)
        batch_inputs = ensure_array_has_batch_dim(inputs, None) if inputs is not None else None

        @eqx.filter_jit
        def em_step(model, m_step_state):
            e_fn = lambda e, inp: model.e_step(e, inp)
            batch_stats, lls = vmap(e_fn)(batch_emissions, batch_inputs)
            lp = model.log_prior() + lls.sum()
            new_model, m_step_state = model.m_step(batch_stats, m_step_state)
            return new_model, m_step_state, lp

        model = self
        m_step_state = None
        log_probs = []

        itr = range(num_iters)
        if verbose:
            try:
                from fastprogress.fastprogress import progress_bar
                itr = progress_bar(itr)
            except ImportError:
                pass

        for _ in itr:
            model, m_step_state, lp = em_step(model, m_step_state)
            log_probs.append(lp)

        return model, jnp.array(log_probs)

    def sample(self, key, num_timesteps, inputs=None):
        """Sample states and emissions from the model.

        Args:
            key: JAX random key
            num_timesteps: number of time steps T
            inputs: optional inputs (None for stationary models)

        Returns:
            (states, emissions) where states is [T] and emissions is [T, D]
        """
        def _step(prev_state, key):
            key1, key2 = jr.split(key)
            trans_probs = self.transitions.transition_matrix()[prev_state]
            state = tfd.Categorical(probs=trans_probs).sample(seed=key2)
            emission = self.emissions.distribution(state).sample(seed=key1)
            return state, (state, emission)

        key1, key2, key = jr.split(key, 3)
        initial_state = tfd.Categorical(probs=self.initial.probs).sample(seed=key1)
        initial_emission = self.emissions.distribution(initial_state).sample(seed=key2)

        next_keys = jr.split(key, num_timesteps - 1)
        _, (next_states, next_emissions) = lax.scan(_step, initial_state, next_keys)

        states = jnp.concatenate([initial_state[None], next_states])
        emissions = jnp.concatenate([initial_emission[None], next_emissions])
        return states, emissions


class GaussianHMM(HMM):
    """Gaussian HMM: y_t | z_t=k ~ N(mu_k, Sigma_k).

    Composes CategoricalInitial + StationaryTransitions + GaussianEmissions.
    """

    @classmethod
    def create(cls, num_states, emission_dim, *, key):
        """Factory method with default initialization.

        Args:
            num_states: number of discrete states K
            emission_dim: dimensionality of emissions D
            key: JAX random key

        Returns:
            GaussianHMM with uniform initial/transitions and random emissions
        """
        initial = CategoricalInitial.create(num_states)
        transitions = StationaryTransitions.create(num_states)
        emissions = GaussianEmissions.create(num_states, emission_dim, key=key)
        return cls(initial, transitions, emissions)
