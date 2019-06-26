"""
The `int_part` module computes the number of `restricted integer partitions` :math:`q(m, n)` of the integer :math:`m`
into at most :math:`n` parts.

Specifically, it counts the number of different degree counts with the sum of degrees
being exactly :math:`m` and that have at most :math:`n` nonzero counts. Since the quantity can only computed via a
recurrence, we pre-compute the values to fill up a memoization table when the number of edges and nodes is not too
large; otherwise, we use accurate asymptotic expressions to efficiently compute the values for large arguments.
[peixoto-nonparametric-2017]_

References
----------
.. [peixoto-nonparametric-2017] Tiago P. Peixoto, "Nonparametric
   Bayesian inference of the microcanonical stochastic block model",
   Phys. Rev. E 95 012317 (2017), :doi:`10.1103/PhysRevE.95.012317`,
   :arxiv:`1610.02703`

"""
import numpy as np
from numba import jit  # TODO: why adding signatures does not make it faster??

from scipy.special import gammaln, spence, loggamma


# for computing the number of restricted partitions of the integer m into at most n pairs
# @jit(float32(uint32, uint32, float32[:, :]), cache=True)
@jit(cache=True)
def log_q(n, k, __q_cache):
    """log_q

    Parameters
    ----------
    n : ``int``

    k : ``int``

    __q_cache : :class:`numpy.ndarray`

    Returns
    -------

    """
    n = int(n)
    k = int(k)
    if n <= 0 or k < 1:
        return 0
    if k > n:
        k = n
    if n < __q_cache.shape[0]:
        return __q_cache[n][k]
    return log_q_approx(n, k)


# @jit(uint32(uint32, float32), cache=True)
@jit(cache=True)
def get_v(u, epsilon=1e-8):
    """get_v

    Parameters
    ----------
    u : ``int``

    epsilon : ``float``

    Returns
    -------

    """
    v = u
    delta = 1
    while delta > epsilon:
        n_v = u * np.sqrt(spence(np.exp(-v)))
        delta = abs(n_v - v)
        v = n_v
    return v


# @jit(float32(uint32, uint32), cache=True)
@jit(cache=True)
def log_q_approx_small(n, k):
    """log_q_approx_small

    Parameters
    ----------
    n : ``int``
    k : ``int``

    Returns
    -------

    """
    return lbinom(n - 1, k - 1) - loggamma(k + 1)


# @jit(float32(uint32, uint32), cache=True)
@jit(cache=True)
def log_q_approx(n, k):
    """log_q_approx

    Parameters
    ----------
    n : ``int``
    k : ``int``

    Returns
    -------

    """
    if k < pow(n, 1 / 4.):
        return log_q_approx_small(n, k)
    u = k / np.sqrt(n)
    v = get_v(u)
    lf = np.log(v) - np.log1p(- np.exp(-v) * (1 + u * u / 2)) / 2 - np.log(2) * 3 / 2. - np.log(u) - np.log(np.pi)
    g = 2 * v / u - u * np.log1p(-np.exp(-v))
    return lf - np.log(n) + np.sqrt(n) * g


# @jit(float32(uint32, float32[:, :]), cache=True)
@jit(cache=True)
def init_q_cache(n_max, __q_cache=np.array([], ndmin=2)):
    """Initiate the look-up table for :math:`q(m, n)`.

    Parameters
    ----------
    n_max : ``int``

    __q_cache : :class:`numpy.ndarray` (required, default: ``np.array([], ndmin=2)``)

    Returns
    -------

    """
    old_n = __q_cache.shape[0]
    if old_n >= n_max:
        return
    __q_cache = np.resize(__q_cache, [n_max + 1, n_max + 1])
    __q_cache.fill(-np.inf)
    for n in range(1, n_max + 1):
        __q_cache[n][1] = 0
        for k in range(2, n + 1):
            __q_cache[n][k] = log_sum(__q_cache[n][k], __q_cache[n][k - 1])
            if n > k:
                __q_cache[n][k] = log_sum(__q_cache[n][k], __q_cache[n - k][k])
    return __q_cache


# @jit(float32(float32, float32), cache=True)
@jit(cache=True)
def log_sum(a, b):
    """log_sum

    Parameters
    ----------
    a : ``int``
    b : ``int``

    Returns
    -------

    """
    return np.maximum(a, b) + np.log1p(np.exp(-np.abs(a - b)))


def lbinom(n, k):
    """Return log of binom(n, k)."""
    if type(n) in [float, int, np.int64, np.float64]:
        n = np.array([n])
        k = np.array([k])
    return (gammaln(np.array([float(x) for x in n + 1])) -
            gammaln(np.array([float(x) for x in n - k + 1])) -
            gammaln(np.array([float(x) for x in k + 1])))
