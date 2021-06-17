import numpy as np
from numba import njit
from numba import prange

import lib.fun_hmm as fun

class binomial_HMM():
    """
    Multidimensional Hidden Markov Model with D independent binomial emission,
    and N hidden states.

    This class wraps the functions defined in lib/fun.py that use numba.
    """
    def __init__(self, N, D, x = None):
        self.N = N
        self.D = D
        
        if x is not None:
            assert x.shape[1] == D, 'Transpose the data'

        # Initialize the parameters of the HMM to some random values
        state_priors = np.random.rand(N)
        state_priors /= state_priors.sum(axis = 0)
        self.log_state_priors = np.log(state_priors)

        transition_matrix = np.diag(np.ones(N)) + np.random.rand(N,N)
        transition_matrix /= transition_matrix.sum(axis = 1)[:, None]
        self.log_transition_matrix = np.log(transition_matrix)
        
        if x is not None:
            emission_matrix = np.array([np.random.rand(x.shape[1])*0.1,
                                        x.sum(axis = 0)/x.shape[1]])
            emission_matrix[emission_matrix == 0] = 1e-3
        else:
            emission_matrix = np.random.rand(N, D)
        self.log_emission_matrix = np.log(emission_matrix)

        self.best_likelihood = -np.inf

    def update(self, x, print_results = True):
        """
        -----------------------------------------------------------------
        Arguments:  - self
                    - x, 2D numpy array of shape T x D
        -----------------------------------------------------------------
        -----------------------------------------------------------------
        This function wraps up all the functions defined in lib/fun.py
        and performs the M-step, updating the parameters of the model
        given the observation sequence.
        -----------------------------------------------------------------
        """

        assert x.shape[1] == self.D
        assert np.all(np.unique(x) == np.array([0,1])), 'Variables must be in {0,1}'

        log_emission_probabilities = fun.emission_model(self.log_emission_matrix, x)
        log_alpha, log_beta = fun.forward_backward(log_emission_probabilities, self.log_transition_matrix,
                                                   self.log_state_priors)

        log_gamma = fun.evaluate_log_gamma(log_alpha, log_beta)
        log_xi = fun.evaluate_log_xi(log_alpha, log_beta, self.log_transition_matrix,
                                     log_emission_probabilities)

        self.log_state_priors, self.log_transition_matrix, self.log_emission_matrix = fun.M_step(log_gamma, log_xi,
                                                                                                 self.log_transition_matrix, x)

        likelihood = fun.logsumexp(log_alpha[-1], axis = 0)
        if print_results:
            print('\t Current log-likelihood: ' + "{:.2f}".format(likelihood))

        if likelihood > self.best_likelihood:
            self.best_likelihood = likelihood
            if print_results:
                print('Best log-likelihood!')

            self.states_probability = np.exp(log_gamma)

            self.best_log_state_priors = self.log_state_priors
            self.best_log_transition_matrix = self.log_transition_matrix
            self.best_log_emission_matrix = self.log_emission_matrix

    def sort_states(self):
        idx_sort = np.argsort(np.exp(self.best_log_emission_matrix).mean(axis = 1))

        self.best_log_emission_matrix = self.best_log_emission_matrix[idx_sort]
        self.best_log_state_priors = self.best_log_state_priors[idx_sort]
        new_tmat = np.zeros((self.N, self.N))

        for new_idx, old_idx in enumerate(idx_sort):
            new_tmat[new_idx] = self.best_log_transition_matrix[old_idx][idx_sort]

        self.best_log_transition_matrix = new_tmat
        self.states_probability = self.states_probability[:,idx_sort]
        


    def load_parameters(self, log_state_priors, log_transition_matrix, log_emission_matrix, x):
        assert log_transition_matrix.shape == (self.N, self.N)
        assert log_emission_matrix.shape == (self.N, self.D)
        assert log_state_priors.size == self.N

        self.best_log_state_priors = log_state_priors
        self.best_log_transition_matrix = log_transition_matrix
        self.best_log_emission_matrix = log_emission_matrix

        log_emission_probabilities = fun.emission_model(log_emission_matrix, x)
        log_alpha, log_beta = fun.forward_backward(log_emission_probabilities, log_transition_matrix, log_state_priors)
        log_gamma = fun.evaluate_log_gamma(log_alpha, log_beta)
        likelihood = fun.logsumexp(log_alpha[-1], axis = 0)

        print('Log-likelihood: ' + "{:.2f}".format(likelihood))
        self.best_likelihood = likelihood
        self.states_probability = np.exp(log_gamma)

        self.sort_states()
