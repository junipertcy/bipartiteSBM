import numpy as np
from numba import jit  # TODO: why adding signatures does not make it faster??

from scipy.special import gammaln, spence, loggamma


# for computing the number of restricted partitions of the integer m into at most n pairs
# @jit(float32(uint32, uint32, float32[:, :]), cache=True)
@jit(cache=True)
def log_q(n, k, __q_cache):
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
    return lbinom(n - 1, k - 1) - loggamma(k + 1)


# @jit(float32(uint32, uint32), cache=True)
@jit(cache=True)
def log_q_approx(n, k):
    if k < pow(n, 1/4.):
        return log_q_approx_small(n, k)
    u = k / np.sqrt(n)
    v = get_v(u)
    lf = np.log(v) - np.log1p(- np.exp(-v) * (1 + u * u/2)) / 2 - np.log(2) * 3 / 2. - np.log(u) - np.log(np.pi)
    g = 2 * v / u - u * np.log1p(-np.exp(-v))
    return lf - np.log(n) + np.sqrt(n) * g


# @jit(float32(uint32, float32[:, :]), cache=True)
@jit(cache=True)
def init_q_cache(n_max, __q_cache):
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
    return np.maximum(a, b) + np.log1p(np.exp(-np.abs(a - b)))


def lbinom(n, k):
    """Return log of binom(n, k)."""
    if type(n) in [float, int, np.int64]:
        n = np.array([n])
        k = np.array([k])
    return (gammaln(np.array([float(x) for x in n + 1])) -
            gammaln(np.array([float(x) for x in n - k + 1])) -
            gammaln(np.array([float(x) for x in k + 1])))
