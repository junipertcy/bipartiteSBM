""" utilities """
from .int_part import *
from scipy.sparse import lil_matrix
from itertools import product
import random
from numba.types import Tuple
from collections import OrderedDict
from loky import get_reusable_executor


def db_factorial_ln(val):
    m = int(val)
    if m & 0x1 == 1:  # m is odd
        return gammaln(m + 1) - gammaln((m - 1) / 2 + 1) - ((m - 1) / 2) * np.log(2)
    else:
        return gammaln(m / 2 + 1) + (m / 2) * np.log(2)


# #################
# ENTROPY FUNCTIONS
# #################


@jit()
def partition_entropy(ka=None, kb=None, k=None, na=None, nb=None, n=None, nr=None, allow_empty=False):
    """
    Compute the partition entropy, P(b), for the current partition. It has several variations depending on the priors
    used. In the crudest way (`compute_profile_likelihood_from_e_rsnr == None`), we formulate P(b) = P(b | B) * P(B).
    Or, by a two-level Bayesian hierarchy,
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

    if nr is None:
        ent = n * np.log(k) + np.log1p(-(1 - 1. / k) ** n)  # TODO: check this term
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


@jit()
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
    e_rs = np.zeros((max(mb) + 1, max(mb) + 1))
    m_ij = lil_matrix((len(mb), len(mb)), dtype=np.int32)
    for i in edgelist:
        # Please do check the index convention of the edgelist
        source_group = int(mb[int(i[0])])
        target_group = int(mb[int(i[1])])
        e_rs[source_group][target_group] += 1
        e_rs[target_group][source_group] += 1
        m_ij[int(i[0]), int(i[1])] += 1  # we only update the upper triangular part of the adj-matrix
    italic_i = 0.
    e_r = np.sum(e_rs, axis=1)
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
        for _e_r in e_r:
            sum_e_r += gammaln(_e_r + 1)

    for ind, e_val in enumerate(np.nditer(e_rs)):
        ind_i = int(np.floor(ind / (e_rs.shape[0])))
        ind_j = ind % (e_rs.shape[0])
        if exact:
            if ind_j > ind_i:
                sum_e_rs += gammaln(e_val + 1)
            elif ind_j == ind_i:
                sum_e_rr += db_factorial_ln(e_val)
        else:
            if e_val != 0.0:
                italic_i += e_val * np.log(
                    e_val / e_r[ind_i] / e_r[ind_j]
                )

    ent += -italic_i / 2
    n_k = assemble_n_k_from_edgelist(edgelist, mb)

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


@jit()
def model_entropy(e, ka=None, kb=None, na=None, nb=None, nr=None, allow_empty=False, is_bipartite=True):
    if not is_bipartite:
        k = ka + kb
        x = (k * (k + 1)) / 2
    else:
        x = ka * kb

    if nr is False:
        dl = lbinom(x + e - 1, e)
    else:
        dl = lbinom(x + e - 1, e) + partition_entropy(ka=ka, kb=kb, na=na, nb=nb, nr=nr, allow_empty=allow_empty)
        # TO use the general prior for the partition entropy, replace the function with:
        # n = na + nb
        # partition_entropy(k=k, n=n, nr=nr, allow_empty=allow_empty)
    return dl


@jit()
def degree_entropy(edgelist, mb, __q_cache=np.array([], ndmin=2), degree_dl_kind="distributed",
                   q_cache_max_e_r=int(1e4)):
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
    ent = 0
    n_r = assemble_n_r_from_mb(mb)

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
            __q_cache = init_q_cache(q_cache_max_e_r, __q_cache)  # the pre-computed lookup table affects precision!
        for ind, n_r_ in enumerate(n_r):
            ent += log_q(e_r[ind], n_r_, __q_cache)
        eta_rk = assemble_eta_rk_from_edgelist_and_mb(edgelist, mb)

        for mb_, eta_rk_ in enumerate(eta_rk):
            for eta in eta_rk_:
                ent -= +loggamma(eta + 1)
            ent -= -loggamma(n_r[mb_] + 1)
    elif degree_dl_kind == "entropy":
        raise NotImplementedError
    # print("degree_dl: {}".format(ent))
    return ent


@jit()
def virtual_move_ds(ori_e_rs, mlist, ka):
    if (max(mlist) >= ka > min(mlist)) or (min(mlist) == 0 and ka == 1) or (
            min(mlist) == ka and ori_e_rs.shape[0] == 1 + ka):
        return np.inf

    _ds = 0
    size = ori_e_rs.shape[0] - 1
    if max(mlist) < ka:  # we are merging groups of type-a
        for idx in range(size - ka):
            _1 = ori_e_rs[mlist[0]][ka + idx + 1]
            _2 = ori_e_rs[mlist[1]][ka + idx + 1]
            _ds -= gammaln(_1 + _2 + 1)
            _ds += gammaln(_1 + 1)
            _ds += gammaln(_2 + 1)
    else:
        for idx in range(ka):
            _1 = ori_e_rs[mlist[0]][idx]
            _2 = ori_e_rs[mlist[1]][idx]
            _ds -= gammaln(_1 + _2 + 1)
            _ds += gammaln(_1 + 1)
            _ds += gammaln(_2 + 1)

    ori_e_r = np.sum(ori_e_rs, axis=1)
    _1 = ori_e_r[mlist[0]]
    _2 = ori_e_r[mlist[1]]
    _ds += gammaln(_1 + _2 + 1)
    _ds -= gammaln(_1 + 1) + gammaln(_2 + 1)

    return _ds


# ####################
# GENERATION FUNCTIONS
# ####################


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


def gen_unequal_partition(n, total, avg_deg, alpha):
    """

    Parameters
    ----------
    n: `int`
        The number of communities.
    total: `int`
        The number of nodes.
    avg_deg: `float`
        The average degree of the network.
    alpha: `float`
        The parameter of the Dirichlet distribution (the smaller the unevener the distribution is).
        We usually use alpha=1 to generate a case and
        filter it out if the lowest size of a community is less than (2 * total / avg_deg) ** 0.5 (resolution limit).

    Returns
    -------
    a: `list`
        The sizes of communities to sum to `total`.
    _: `float`
        The ratio of largest-sized community to the lowest-sized one.

    """
    cutoff = (2 * total / avg_deg) ** 0.5
    d = np.random.dirichlet([alpha] * n, 1)[0]

    while not np.all(d * total > cutoff):
        d = np.random.dirichlet([alpha] * n, 1)[0]

    a = list(map(int, d * total))
    remain_a = int(total - sum(a))
    for idx, _ in enumerate(range(remain_a)):
        a[idx] += 1

    return a, max(a) / min(a)


def gen_e_rs(b, n_edges, p):
    """

    Parameters
    ----------
    b: `int`
        The number of communities within each type. (suppose Ka = Kb)
    n_edges: `int`
        The number of edges planted in the system.
    p: `float`
        The edge propensity between groups; i.e., the ratio c_out / c_in

    Returns
    -------

    """
    c = n_edges / (b + (b ** 2 - b) * p)
    c_in = c
    c_out = c * p
    e_rs = np.zeros([b * 2, b * 2], dtype=int)

    perm = product(range(0, b), range(b, b * 2))
    idx_in = np.linspace(0, b ** 2 - 1, b, dtype=int)
    for idx, p in enumerate(perm):
        i = p[0]
        j = p[1]
        if idx in idx_in:
            e_rs[i][j] = c_in
        else:
            e_rs[i][j] = c_out
        e_rs[j][i] = e_rs[i][j]
    return e_rs


@jit()
def gen_e_rs_unequal(ka, kb, n_edges):
    """

    Parameters
    ----------
    ka: `int`
        The number of communities within type-a.
    kb: `int`
        The number of communities within type-b.
    n_edges: `int`
        The number of edges planted in the system.

    Returns
    -------

    """
    c = list(map(int, np.random.dirichlet([1] * ka * kb, 1)[0] * n_edges))
    remain_c = n_edges - sum(c)
    for idx, _ in enumerate(range(remain_c)):
        c[idx] += 1
    e_rs = np.zeros([ka + kb, ka + kb], dtype=int)

    perm = product(range(0, ka), range(ka, ka + kb))
    for idx, p in enumerate(perm):
        i = p[0]
        j = p[1]
        e_rs[i][j] = c[idx]
        e_rs[j][i] = e_rs[i][j]
    return e_rs


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


# ##################
# ASSEMBLE FUNCTIONS
# ##################


def assemble_n_r_from_mb(mb):
    """
    Get n_r vector (number of nodes in each group) from the membership vector.

    Parameters
    ----------
    mb

    Returns
    -------
    n_r

    """
    n_r = np.zeros(np.max(mb) + 1)
    for block_id in mb:
        n_r[block_id] += 1
    n_r = np.array([int(x) for x in n_r])
    return n_r


def assemble_n_k_from_edgelist(edgelist, mb):
    """
    Get n_k, or the number n_k of nodes of degree k.

    Parameters
    ----------
    edgelist: `array-like`
    mb: `array-like`

    Returns
    -------
    n_k: `array-like`

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


def assemble_e_rs_from_mb(edgelist, mb):
    assert type(edgelist) is list, \
        "[ERROR] the type of the first input should be a list; however, it is {}".format(str(type(edgelist)))
    # construct e_rs matrix
    e_rs = np.zeros((max(mb) + 1, max(mb) + 1))
    for i in edgelist:
        # Please do check the index convention of the edgelist
        source_group = int(mb[int(i[0])])
        target_group = int(mb[int(i[1])])
        if source_group == target_group:
            raise ImportError("[ERROR] This is not a bipartite network! The mb is {}".format(mb))
        e_rs[source_group][target_group] += 1
        e_rs[target_group][source_group] += 1

    e_r = np.sum(e_rs, axis=1)
    return e_rs, e_r


def assemble_eta_rk_from_edgelist_and_mb(edgelist, mb):
    """
    Get eta_rk, or the number eta_rk of nodes of degree k that belong to group r.

    Parameters
    ----------
    edgelist: `array-like`
    mb: `array-like`

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


@jit()
def compute_profile_likelihood_from_e_rs(e_rs):
    assert type(e_rs) is np.ndarray, "[ERROR] input parameter (m_e_rs) should be of type numpy.ndarray"
    italic_i = 0.
    e_r = np.sum(e_rs, axis=1)
    num_edges = e_r.sum() / 2.
    for ind, e_val in enumerate(np.nditer(e_rs)):
        ind_i = int(np.floor(ind / (e_rs.shape[0])))
        ind_j = ind % (e_rs.shape[0])
        if e_val != 0.0:
            italic_i += e_val / 2. / num_edges * np.log(
                e_val / e_r[ind_i] / e_r[ind_j] * 2 * num_edges
            )
    return italic_i


def get_desc_len_from_data(na, nb, n_edges, ka, kb, edgelist, mb, diff=False, nr=None, allow_empty=False,
                           degree_dl_kind="distributed", q_cache=np.array([], ndmin=2), is_bipartite=True):
    """
    Description length difference to a randomized instance

    Parameters
    ----------
    na: `int`
        Number of nodes in type-a.
    nb: `int`
        Number of nodes in type-b.
    n_edges: `int`
        Number of edges.
    ka: `int`
        Number of communities in type-a.
    kb: `int`
        Number of communities in type-b.
    edgelist: `list`
        Edgelist in Python list structure.
    mb: `list`
        Community membership of each node in Python list structure.
    diff: `bool`
        When `diff == True`,
        the returned description value will be the difference to that of a random bipartite network. Otherwise, it will
        return the entropy (a.k.a. negative log-likelihood) associated with the current block partition.
    allow_empty: `bool`
    nr: `array-like`
    degree_dl_kind: `str` (optional, default: `"distributed"`)
        1. `degree_dl_kind == "uniform"`
        2. `degree_dl_kind == "distributed"` (default)
        3. `degree_dl_kind == "entropy"`
    is_bipartite: `bool` (default: `"True"`)

    Returns
    -------
    desc_len_b: `float`
        Difference of the description length to the bipartite ER network, per edge.

    """
    edgelist = list(map(lambda e: [int(e[0]), int(e[1])], edgelist))
    desc_len = 0.
    # finally, we compute the description length
    if diff:  # todo: add more options to it; now, only uniform prior for P(e) is included.
        italic_i = compute_profile_likelihood(edgelist, mb, ka=ka, kb=kb)
        desc_len += (na * np.log(ka) + nb * np.log(kb) - n_edges * (italic_i - np.log(2))) / n_edges
        x = float(ka * kb) / n_edges
        desc_len += (1 + x) * np.log(1 + x) - x * np.log(x)
        desc_len -= (1 + 1 / n_edges) * np.log(1 + 1 / n_edges) - (1 / n_edges) * np.log(1 / n_edges)
    else:
        desc_len += adjacency_entropy(edgelist, mb)
        desc_len += model_entropy(n_edges, ka=ka, kb=kb, na=na, nb=nb, nr=nr, allow_empty=allow_empty,
                                  is_bipartite=is_bipartite)
        desc_len += degree_entropy(edgelist, mb, __q_cache=q_cache, degree_dl_kind=degree_dl_kind)
    return desc_len.__float__()


def get_desc_len_from_data_uni(n, n_edges, k, edgelist, mb):
    """
    Description length difference to a randomized instance, via PRL 110, 148701 (2013).

    Parameters
    ----------
    n: `int`
        Number of nodes.
    n_edges: `int`
        Number of edges.
    k: `int`
        Number of communities.
    edgelist: `list`
        A list of edges.
    mb: `list`
        A list of node community membership.

    Returns
    -------
    desc_len_b: `float`
        Difference of the description length to the ER network, per edge.

    """
    italic_i = compute_profile_likelihood(edgelist, mb, k=k)

    # finally, we compute the description length
    desc_len_b = (n * np.log(k) - n_edges * italic_i) / n_edges
    x = float(k * (k + 1)) / 2. / n_edges
    desc_len_b += (1 + x) * np.log(1 + x) - x * np.log(x)
    desc_len_b -= (1 + 1 / n_edges) * np.log(1 + 1 / n_edges) - (1 / n_edges) * np.log(1 / n_edges)
    return desc_len_b


@jit(uint32[:](uint32[:], uint32[:]), cache=True, fastmath=True)
def accept_mb_merge(mb, mlist):
    _mb = np.zeros(len(mb), dtype=np.uint32)
    mlist.sort()
    for _node_id, _g in enumerate(mb):
        if _g == mlist[1]:
            _mb[_node_id] = mlist[0]
        elif _g < mlist[1]:
            _mb[_node_id] = _g
        else:
            _mb[_node_id] = _g - 1
    assert max(_mb) + 1 == max(mb), \
        "[ERROR] inconsistency between the membership indexes and the number of blocks."
    return _mb


@jit(Tuple((uint32, uint32, uint32[:, :], uint32[:]))(uint32, uint32, uint32[:, :]), cache=True, fastmath=True)
def merge_matrix(ka, kb, m_e_rs):
    """
    Merge random two rows of the affinity matrix (dim = K) to gain a reduced matrix (dim = K - 1)
    Parameters
    ----------
    ka : int
        number of type-a communities in the affinity matrix
    kb : int
        number of type-b communities in the affinity matrix
    m_e_rs : numpy array
        the affinity matrix
    Returns
    -------
    new_ka : int
        the new number of type-a communities in the affinity matrix
    new_kb : int
        the new number of type-b communities in the affinity matrix
    c : numpy array
        the new affinity matrix
    merge_list : list(int, int)
        the two row-indexes of the original affinity matrix that were merged
    """
    assert type(m_e_rs) is np.ndarray, "[ERROR] input parameter (m_e_rs) should be of type numpy.ndarray"
    assert np.all(m_e_rs.transpose() == m_e_rs), "[ERROR] input m_e_rs matrix is not symmetric!"

    from_row = random.choices([0, ka], weights=[ka, kb], k=1)[0]
    a = m_e_rs[0:ka, ka:ka + kb]

    merge_list = list([0, 0])  # which two mb label should be merged together?
    mb_map = OrderedDict()
    new_ka = 0
    new_kb = 0

    if ka == 1:  # do not merge type-a rows (this happens when <i_0> is set too high)
        from_row = ka
    elif kb == 1:
        from_row = 0

    b = np.zeros([ka, kb])
    if from_row == 0:
        perm = np.arange(a.shape[0])
        np.random.shuffle(perm)
        for _ind in np.arange(a.shape[0]):
            mb_map[_ind] = perm[_ind]
        a = a[perm]

        new_ka = ka - 1
        new_kb = kb
        b = np.zeros([new_ka, new_kb])
        for ind_i, _a in enumerate(a):
            for ind_j, __a in enumerate(_a):
                if ind_i < from_row:
                    b[ind_i][ind_j] = __a
                elif ind_i > from_row + 1:
                    b[ind_i - 1][ind_j] = __a
                elif ind_i == from_row:
                    b[ind_i][ind_j] += __a
                elif ind_i == from_row + 1:
                    b[ind_i - 1][ind_j] += __a

        merge_list[0] = mb_map[from_row]
        merge_list[1] = mb_map[from_row + 1]

    elif from_row == ka:
        perm = np.arange(a.shape[1])
        np.random.shuffle(perm)
        for _ind in np.arange(a.shape[1]):
            mb_map[_ind] = perm[_ind]

        new_ka = ka
        new_kb = kb - 1
        a = a.transpose()

        a = a[perm]

        b = np.zeros([new_kb, new_ka])
        for ind_i, _a in enumerate(a):
            for ind_j, __a in enumerate(_a):
                if ind_i < from_row - new_ka:
                    b[ind_i][ind_j] = __a
                elif ind_i > from_row + 1 - new_ka:
                    b[ind_i - 1][ind_j] = __a
                elif ind_i == from_row - new_ka:
                    b[ind_i][ind_j] += __a
                elif ind_i == from_row + 1 - new_ka:
                    b[ind_i - 1][ind_j] += __a
        merge_list[0] = mb_map[from_row - ka] + ka
        merge_list[1] = mb_map[from_row + 1 - ka] + ka

    c = np.zeros([new_ka + new_kb, new_ka + new_kb])
    bt = b.transpose()
    if from_row == ka:
        b = bt
        bt = b.transpose()

    for ind_i, _c in enumerate(c):
        for ind_j, _ in enumerate(_c):
            if ind_j < new_ka <= ind_i:
                c[ind_i][ind_j] = bt[ind_i - new_ka][ind_j]
            elif ind_i < new_ka <= ind_j:
                c[ind_i][ind_j] = b[ind_i][ind_j - new_ka]

    assert new_ka + new_kb == c.shape[0], "new_ka = {}; new_kb = {}; new_mat.shape[0] = {}".format(
        new_ka, new_kb, c.shape[0]
    )
    assert new_ka + new_kb == c.shape[1], "new_ka = {}; new_kb = {}; new_mat.shape[1] = {}".format(
        new_ka, new_kb, c.shape[1]
    )
    assert np.all(c.transpose() == c), "Error: output m_e_rs matrix is not symmetric!"
    return new_ka, new_kb, c, merge_list


# ###############
# Parallelization
# ###############


def loky_executor(max_workers, timeout, func, feeds):
    assert type(feeds) is list, "[ERROR] feeds should be a Python list; here it is {}".format(str(type(feeds)))
    loky_executor = get_reusable_executor(max_workers=int(max_workers), timeout=int(timeout))
    results = loky_executor.map(func, feeds)
    return results
