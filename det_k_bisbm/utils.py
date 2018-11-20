""" utilities """
from .int_part import *
from scipy.sparse import lil_matrix


def db_factorial_ln(val):
    m = int(val)
    if m & 0x1 == 1:  # m is odd
        return gammaln(m + 1) - gammaln((m - 1) / 2 + 1) - ((m - 1) / 2) * np.log(2)
    else:
        return gammaln(m / 2 + 1) + (m / 2) * np.log(2)


def partition_entropy(ka=None, kb=None, k=None, na=None, nb=None, n=None, nr=None, allow_empty=False):
    """
    Compute the partition entropy, P(b), for the current partition. It has several variations depending on the priors
    used. In the crudest way (`nr == None`), we formulate P(b) = P(b | B) * P(B). Or, by a two-level Bayesian hierarchy,
    we can do P(b) = P(b | n) * P(n | B) * P(B).

    Parameters
    ----------
    ka: `int`
    kb: `int`
    k: `int`
    na: `int`
    nb: `int`
    n: `int`
    nr: `array-like`
    allow_empty: `bool`

    Returns
    -------
    ent: `float`

    """
    if type(n) is int:
        n = np.array([n])
        k = np.array([k])
    elif type(na) is int and type(nb) is int:
        na = np.array([na])
        ka = np.array([ka])
        nb = np.array([nb])
        kb = np.array([kb])

    ent = 0.
    if nr is None:
        ent = n * np.log(k) + np.log1p(-(1 - 1./k) ** n)  # TODO: check this term
    else:
        if ka is None and kb is None and k is not None:
            if allow_empty:
                ent = lbinom(k + n - 1, n)
            else:
                ent = lbinom(n - 1, k - 1)
            ent += (gammaln(n + 1) - gammaln(nr + 1).sum()) + np.log(n)  # TODO: check the last term (should be alright)
        elif ka is not None and kb is not None and k is None:
            if allow_empty:
                # TODO
                raise NotImplementedError
            else:
                ent = lbinom(na - 1, ka - 1) + lbinom(nb - 1, kb - 1)
            ent += (gammaln(na + 1) + gammaln(nb + 1) - gammaln(nr + 1).sum()) + np.log(na) + np.log(nb)
        else:
            raise AttributeError
    return ent


def adjacency_entropy(edgelist, mb, exact=True, multigraph=True):
    """
    Calculate the entropy (a.k.a. negative log-likelihood) associated with the current block partition. It does not
    include the model entropy.

    Parameters
    ----------
    edgelist: `array-like`

    mb: `array-like`

    exact: `bool`

    multigraph: `bool`

    Returns
    -------
    ent: `float`
        the entropy.
    """
    ent = 0.
    m_e_rs = np.zeros((max(mb) + 1, max(mb) + 1))
    m_ij = lil_matrix((len(mb), len(mb)), dtype=np.int8)
    for i in edgelist:
        # Please do check the index convention of the edgelist
        source_group = int(mb[int(i[0])])
        target_group = int(mb[int(i[1])])
        m_e_rs[source_group][target_group] += 1
        m_e_rs[target_group][source_group] += 1
        m_ij[int(i[0]), int(i[1])] += 1  # we only update the upper triangular part of the adj-matrix
    italic_i = 0.
    m_e_r = np.sum(m_e_rs, axis=1)
    sum_m_ii = 0.
    sum_m_ij = 0.
    sum_e_rs = 0.
    sum_e_rr = 0.
    sum_e_r = 0.
    if exact:
        if multigraph:
            ind_i, ind_j = m_ij.nonzero()
            for ind in zip(ind_i, ind_j):
                val = m_ij[ind[0], ind[1]]
                if val > 1:
                    if ind[0] == ind[1]:
                        sum_m_ii += db_factorial_ln(val)
                    else:
                        sum_m_ij += gammaln(val + 1)
        for _m_e_r in m_e_r:
            sum_e_r += gammaln(_m_e_r + 1)

    for ind, e_val in enumerate(np.nditer(m_e_rs)):
        ind_i = int(np.floor(ind / (m_e_rs.shape[0])))
        ind_j = ind % (m_e_rs.shape[0])
        if exact:
            if ind_j > ind_i:
                sum_e_rs += gammaln(e_val + 1)
            elif ind_j == ind_i:
                sum_e_rr += db_factorial_ln(e_val)
        else:
            if e_val != 0.0:
                italic_i += e_val * np.log(
                    e_val / m_e_r[ind_i] / m_e_r[ind_j]
                )

    ent += -italic_i / 2
    n_k = get_n_k_from_edgelist(edgelist, mb)

    ent_deg = 0
    for deg, k in enumerate(n_k):
        if deg != 0 and k != 0:
            ent_deg -= k * gammaln(deg + 1)

    ent += ent_deg
    if exact:
        return ent_deg - sum_e_rs - sum_e_rr + sum_e_r + sum_m_ii + sum_m_ij
    else:
        num_edges = len(edgelist)
        ent += -num_edges
        return ent


def model_entropy(e, ka=None, kb=None, k=None, na=None, nb=None, n=None, nr=None, allow_empty=False):
    if ka is None and kb is None and k is not None:
        x = (k * (k + 1)) / 2
        if nr is False:
            l = lbinom(x + e - 1, e)
        else:
            l = lbinom(x + e - 1, e) + partition_entropy(k=k, n=n, nr=nr, allow_empty=allow_empty)
    elif ka is not None and kb is not None and k is None:
        x = ka * kb
        if nr is False:
            l = lbinom(x + e - 1, e)
        else:
            l = lbinom(x + e - 1, e) + partition_entropy(ka=ka, kb=kb, na=na, nb=nb, nr=nr, allow_empty=allow_empty)
            # print("P(e) and P(b|n_r) are {} and {}".format(lbinom(x + e - 1, e), partition_entropy(ka=ka, kb=kb, na=na, nb=nb, nr=nr, allow_empty=allow_empty)))
    else:
        raise AttributeError
    return l


def gen_equal_partition(n, total):
    """

    Parameters
    ----------
    n: `int`
    total: `int`

    Returns
    -------
    n_blocks: `list[int]`

    """
    all_nodes = np.arange(total)
    n_blocks = list(map(len, np.array_split(all_nodes, n)))

    return n_blocks


def gen_equal_bipartite_partition(na, nb, ka, kb):
    """

    Parameters
    ----------
    na
    nb
    ka
    kb

    Returns
    -------
    n: `list[int]`

    """
    n_blocks = map(int, gen_equal_partition(ka, na) + gen_equal_partition(kb, nb))
    n = []
    for idx, i in enumerate(n_blocks):
        n += [idx] * i
    return n


def get_n_r_from_mb(mb):
    """
    Get n_r vector (number of nodes in each group) from the membership vector.

    Parameters
    ----------
    mb

    Returns
    -------
    n_r

    """
    assert type(mb) is list, "ERROR: the type of the input parameter should be a list"
    n_r = np.zeros(np.max(mb) + 1)
    for block_id in mb:
        n_r[block_id] += 1
    n_r = np.array([int(x) for x in n_r])
    return n_r


def get_n_k_from_edgelist(edgelist, mb):
    """
    Get n_k, or the number n_k of nodes of degree k.

    Parameters
    ----------
    edgelist: array-like
    mb: array-like

    Returns
    -------
    n_k: array-like

    """
    k = np.zeros(len(mb) + 1)
    for edge in edgelist:
        k[edge[0]] += 1
        k[edge[1]] += 1

    max_ = np.max(k).__int__()
    n_k = np.zeros(max_ + 1)
    for k_ in k:
        n_k[k_.__int__()] += 1
    return n_k


def get_eta_rk_from_edgelist_and_mb(edgelist, mb):
    """
    Get eta_rk, or the number eta_rk of nodes of degree k that belong to group r.

    Parameters
    ----------
    edgelist

    Returns
    -------

    """
    assert np.min(mb).__int__() == 0, "The index of a membership label must start from 0."

    mb_max_ = np.max(mb).__int__()

    k = np.zeros([mb_max_ + 1, len(mb)])
    for edge in edgelist:
        k[mb[edge[0]]][edge[0]] += 1
        k[mb[edge[1]]][edge[1]] += 1
    max_deg_in_each_mb_ = np.max(k, axis=0)
    max_ = int(np.max(max_deg_in_each_mb_))
    eta_rk = np.zeros([mb_max_ + 1, max_ + 1])
    for mb_ in range(mb_max_ + 1):
        for node_idx, k_ in enumerate(k[mb_]):
            if mb[node_idx] == mb_:
                eta_rk[mb_][k_.__int__()] += 1

    return eta_rk.astype(int)


def compute_degree_entropy(edgelist, mb, __q_cache=np.array([], ndmin=2), degree_dl_kind="distributed", q_cache_max_e_r=10000):
    """

    Parameters
    ----------
    edgelist
    mb
    __q_cache
    degree_dl_kind: `str`

        1. ``degree_dl_kind == "uniform"``

        This corresponds to a non-informative prior, where the node
        degrees are sampled from an uniform distribution.

        2. ``degree_dl_kind == "distributed"``

        This option should be preferred in most cases.


    Returns
    -------

    """
    import time
    ent = 0
    n_r = get_n_r_from_mb(mb)

    e_r = np.zeros(max(mb) + 1)
    for i in edgelist:
        # Please do check the index convention of the edgelist
        source_group = int(mb[int(i[0])])
        target_group = int(mb[int(i[1])])
        e_r[source_group] += 1
        e_r[target_group] += 1

    if degree_dl_kind == "uniform":
        ent += np.sum(lbinom(n_r + e_r - 1, e_r))
    elif degree_dl_kind == "distributed":
        if len(__q_cache) == 1:
            t0 = time.time()
            __q_cache = init_q_cache(q_cache_max_e_r, __q_cache)  # the pre-computed lookup table affects precision!
            t1 = time.time()
            print("[Good news!] log q(m, n) look-up table computed for m <= 1e4; taking {} sec".format(t1 - t0))
        for ind, n_r_ in enumerate(n_r):
            # print("e_r[ind]: {}; n_r_: {}; val: {}".format(e_r[ind], n_r_, log_q(e_r[ind], n_r_, __q_cache)))
            ent += log_q(e_r[ind], n_r_, __q_cache)
        eta_rk = get_eta_rk_from_edgelist_and_mb(edgelist, mb)

        for mb_, eta_rk_ in enumerate(eta_rk):
            for eta in eta_rk_:
                # print("loggamma(eta + 1): ", loggamma(eta + 1))
                ent -= +loggamma(eta + 1)  #TODO
            ent -= -loggamma(n_r[mb_] + 1)
    elif degree_dl_kind == "entropy":
        raise NotImplementedError
    print("degree_dl: {}".format(ent))
    return ent


def compute_profile_likelihood(edgelist, mb, ka=None, kb=None, k=None):
    assert type(edgelist) is list, "[ERROR] the type of the input parameter (edgelist) should be a list"
    assert type(mb) is list, "[ERROR] the type of the input parameter (mb) should be a list"
    # First, let's compute the m_e_rs from the edgelist and mb
    m_e_rs = np.zeros((max(mb) + 1, max(mb) + 1))
    for i in edgelist:
        # Please do check the index convention of the edgelist
        source_group = int(mb[int(i[0])])
        target_group = int(mb[int(i[1])])
        if ka is not None or kb is not None:
            if source_group == target_group:
                raise ImportError("This is not a bipartite network!")
        m_e_rs[source_group][target_group] += 1
        m_e_rs[target_group][source_group] += 1

    # then, we compute the profile likelihood from the m_e_rs
    italic_i = 0.
    m_e_r = np.sum(m_e_rs, axis=1)
    num_edges = m_e_r.sum() / 2.
    for ind, e_val in enumerate(np.nditer(m_e_rs)):
        ind_i = int(np.floor(ind / (m_e_rs.shape[0])))
        ind_j = ind % (m_e_rs.shape[0])
        if e_val != 0.0:
            italic_i += e_val / 2. / num_edges * np.log(
                e_val / m_e_r[ind_i] / m_e_r[ind_j] * 2 * num_edges
            )
    if ka is not None or kb is not None:
        assert m_e_rs.shape[0] == ka + kb, "[ERROR] m_e_rs dimension (={}) is not equal to ka (={}) + kb (={})!".format(
            m_e_rs.shape[0], ka, kb
        )
    elif k is not None:
        assert m_e_rs.shape[0] == k, "[ERROR] m_e_rs dimension (={}) is not equal to k (={})!".format(
            m_e_rs.shape[0], k
        )
    return italic_i

