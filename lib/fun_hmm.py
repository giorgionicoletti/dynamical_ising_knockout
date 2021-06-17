import numpy as np
from numba import njit
from numba import prange

@njit
def logsumexp(log_mat, axis = 0):
    """
    -----------------------------------------------------------------
    Arguments:  - log_mat, 2D numpy array
                - axis, int
    -----------------------------------------------------------------
    -----------------------------------------------------------------
    This function implements a numerically stable version of the
    logsumexp function along a given axis.

    It takes as input the log of a matrix and returns the log of the
    sum along an axis of the exponential of the matrix.
    -----------------------------------------------------------------
    """
    nconst = np.max(log_mat)
    out = np.sum(np.exp(log_mat - nconst), axis = axis)
    return nconst + np.log(out)


@njit
def log_domain_matmul(log_A, log_B):
    """
    -----------------------------------------------------------------
    Arguments:  - log_A, 2D numpy array of shape m x n
                - log_B, 2D numpy array of shape n x p
    -----------------------------------------------------------------
    -----------------------------------------------------------------
    This function implements the log version of the multiplication
    between two matrices C_{ij} = \sum_k A_{ik} B_{kj}, which gives

    \log C_{ij} = \log \sum_{k} \exp[log A_{ik} + log B_{k,j}]

    so we can work with the log of the matrices directly.
    -----------------------------------------------------------------
    """

    m = log_A.shape[0]
    n = log_A.shape[1]
    p = log_B.shape[1]

    elementwise_sum = np.empty((m, p, n), dtype = np.float64)
    log_A_hD = np.empty((m, n, 1), dtype = np.float64)
    log_B_hD = np.empty((1, n, p), dtype = np.float64)

    log_A_hD = np.expand_dims(log_A, axis = 2)
    log_B_hD = np.expand_dims(log_B, axis = 0)
    elementwise_sum = log_A_hD + log_B_hD

    return logsumexp(elementwise_sum, axis = 1)


@njit(parallel = True)
def emission_model(log_emission_matrix, x):
    """
    -----------------------------------------------------------------
    Arguments:  - log_emission_matrix, 2D numpy array of shape N x D
                - x, 1D numpy array of shape T
    -----------------------------------------------------------------
    -----------------------------------------------------------------
    This function evaluates the log of the probability of observing
    the sequence of observed states x given that the system is in the
    i-th hidden state at time t. x is observed for T timesteps, x[t]
    is a D-dimensional array and the system can be in N hidden states.

    The result is a T x N matrix where each column gives the logarithm
    of the probability of observing x[t] for each of the hidden states
    the system can be in.
    -----------------------------------------------------------------
    """
    emission_matrix = np.exp(log_emission_matrix)
    T = x.shape[0]
    N = emission_matrix.shape[0]
    emission_probabilities = np.empty((T, N), dtype = np.float64)

    for t in prange(T):
        to_marginalize = emission_matrix*x[t] + (1 - emission_matrix)*(1 - x[t])
        emission_probabilities[t] = np.sum(np.log(to_marginalize), axis = 1)

    return emission_probabilities



@njit
def forward_backward(log_emission_probabilities, log_transition_matrix, log_state_priors):
    """
    -----------------------------------------------------------------
    Arguments:  - log_emission_probabilities, 2D numpy array of shape
                  T x N
                - log_transition_matrix, 2D numpy array of shape
                  N x N
                - log_state_priors, 1D numpy array of shape N
    -----------------------------------------------------------------
    -----------------------------------------------------------------
    This function evaluates in a single pass the log of the forward
    and the backward probabilities of the Hidden Markov Model.
    -----------------------------------------------------------------
    """

    T, N = log_emission_probabilities.shape
    log_alpha = np.empty((T, N), dtype = np.float64)
    log_beta = np.empty((T, N), dtype = np.float64)

    log_beta[T-1] = np.zeros(N, dtype = np.float64)
    log_alpha[0] = log_emission_probabilities[0] + log_state_priors

    for t in range(1, T):
        a_matrix_mult = log_domain_matmul(log_alpha[t-1].reshape(1, N), log_transition_matrix)
        log_alpha[t] = log_emission_probabilities[t] + a_matrix_mult

        b_elementwise_mult = log_emission_probabilities[T-t] + log_beta[T-t]
        b_matrix_mult = log_domain_matmul(log_transition_matrix, b_elementwise_mult.reshape(N,1))
        log_beta[T-t-1] = b_matrix_mult.reshape(-1)

    return log_alpha, log_beta


@njit
def evaluate_log_gamma(log_alpha, log_beta):
    """
    -----------------------------------------------------------------
    Arguments:  - log_alpha, 2D numpy array of shape T x N
                - log_beta, 2D numpy array of shape T x N
    -----------------------------------------------------------------
    -----------------------------------------------------------------
    This function evaluates the probability that the Hidden Markov
    Model is in one of the possible hidden states at each time, given
    its parameters (that are encoded in the foward and backward
    probabilities).
    -----------------------------------------------------------------
    """
    g = np.empty(log_alpha.shape, dtype = np.float64)
    g = log_alpha + log_beta
    return g - np.expand_dims(logsumexp(g, axis = 1), axis = 1)


@njit
def evaluate_log_xi_old(log_gamma, log_beta, log_transition_matrix, log_emission_probabilities):
    """
    -----------------------------------------------------------------
    Arguments:  - log_gamma, 2D numpy array of shape T x N
                - log_transition_matrix, 2D numpy array of shape
                  N x N
                - log_emission_probabilities, 2D numpy array of shape
                  T x N
                - log_beta, 2D numpy array of shape T x N
    -----------------------------------------------------------------
    -----------------------------------------------------------------
    This function evaluates the probability that the Hidden Markov
    Model is in a subsequent pair of the possible hidden states at
    each times (t-1, t), given its parameters.
    -----------------------------------------------------------------
    """
    T = log_gamma.shape[0]
    N = log_gamma.shape[1]
    log_xi = np.empty((T-1, N, N), dtype = np.float64)

    log_xi = np.expand_dims(log_gamma[:-1], axis = 2) + log_transition_matrix \
             + np.expand_dims(log_emission_probabilities[1:], axis = 1) \
             + np.expand_dims(log_beta[1:], axis = 1) \
             - np.expand_dims(log_beta[:-1], axis = 2)
        
    return log_xi


@njit(parallel = True)
def evaluate_log_xi_parallel(log_alpha, log_beta, log_transition_matrix, log_emission_probabilities):
    """
    -----------------------------------------------------------------
    Arguments:  - log_alpha, 2D numpy array of shape T x N
                - log_transition_matrix, 2D numpy array of shape
                  N x N
                - log_emission_probabilities, 2D numpy array of shape
                  T x N
                - log_beta, 2D numpy array of shape T x N
    -----------------------------------------------------------------
    -----------------------------------------------------------------
    This function evaluates the probability that the Hidden Markov
    Model is in a subsequent pair of the possible hidden states at
    each times (t-1, t), given its parameters.
    -----------------------------------------------------------------
    """

    T = log_alpha.shape[0] - 1
    N = log_alpha.shape[1]
    log_xi_norm = np.empty((T, N, N), dtype = np.float64)
    current_log_xi = np.empty((N, N), dtype = np.float64)
    
    log_tmat = log_transition_matrix.T
    
    for idx in prange(T):
        t = idx + 1

        current_log_xi  = (log_alpha[t-1] + log_tmat).T + log_emission_probabilities[t] + log_beta[t]
        
        nconst = np.max(current_log_xi)
        norm = np.log(np.exp(current_log_xi - nconst).sum()) + nconst
        
        log_xi_norm[idx] = current_log_xi - norm
    
    return log_xi_norm


def evaluate_log_xi(log_alpha, log_beta, log_transition_matrix, log_emission_probabilities):
    """
    -----------------------------------------------------------------
    Arguments:  - log_alpha, 2D numpy array of shape T x N
                - log_transition_matrix, 2D numpy array of shape
                  N x N
                - log_emission_probabilities, 2D numpy array of shape
                  T x N
                - log_beta, 2D numpy array of shape T x N
    -----------------------------------------------------------------
    -----------------------------------------------------------------
    This function evaluates the probability that the Hidden Markov
    Model is in a subsequent pair of the possible hidden states at
    each times (t-1, t), given its parameters.
    -----------------------------------------------------------------
    """

    log_xi = np.expand_dims(log_alpha[:-1], axis = 2) + log_transition_matrix \
             + np.expand_dims(log_emission_probabilities[1:], axis = 1) \
             + np.expand_dims(log_beta[1:], axis = 1) \
    
    nconst = log_xi.max(axis = (1,2))
    norm  = np.log(np.sum(np.exp(log_xi - nconst[..., None, None]), axis = (1,2))) + nconst
    
    return log_xi - norm[..., None, None]


@njit
def M_step(log_gamma, log_xi, log_transition_matrix, x):
    """
    -----------------------------------------------------------------
    Arguments:  - log_gamma, 2D numpy array of shape T x N
                - log_xi, 3D numpy array of shape (T - 1) x N x N
                - log_transition_matrix, 2D numpy array of shape
                  N x N
                - x, 1D numpy array of shape T
    -----------------------------------------------------------------
    -----------------------------------------------------------------
    This function performs the maximization step for a D-dimensional
    binomial Hidden Markov Model. The M-step evaluates a new set of
    parameters to minimize the overall likelihood of the model for
    the observed data.
    -----------------------------------------------------------------
    """

    T = log_gamma.shape[0]
    N = log_gamma.shape[1]
    D = x.shape[1]

    up_log_state_prior = np.empty(N, dtype = np.float64)
    up_log_transition_matrix = np.empty((N, N), dtype = np.float64)
    up_log_emission_matrix = np.empty((N, D), dtype = np.float64)

    up_log_state_prior = log_gamma[0]
    up_log_transition_matrix = logsumexp(log_xi, axis = 0) \
                               - np.expand_dims(logsumexp(log_gamma[:-1], axis = 0), axis = 1)

    up_log_emission_matrix = log_domain_matmul(log_gamma.T, np.log(x)) \
                             - np.expand_dims(logsumexp(log_gamma, axis = 0), axis = 1)

    return up_log_state_prior, up_log_transition_matrix, up_log_emission_matrix
