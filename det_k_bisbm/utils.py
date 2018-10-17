""" utilities """
import numpy as np
from scipy.special import gammaln


def lbinom(n, k):
    """Return log of binom(n, k)."""
    if type(n) in [float, int]:
        n = np.array([n])
        k = np.array([k])
    return (gammaln(np.array([float(x) for x in n + 1])) -
            gammaln(np.array([float(x) for x in n - k + 1])) -
            gammaln(np.array([float(x) for x in k + 1])))


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


def fitting_entropy(edgelist, mb):
    """
    Calculate the entropy (a.k.a. negative log-likelihood) associated with the current block partition. It does not
    include the model entropy.

    Parameters
    ----------
    edgelist: `array-like`

    mb: `array-like`


    Returns
    -------
    ent: `float`
        the entropy.
    """
    ent = 0.
    num_edges = len(edgelist)
    ent += -num_edges

    m_e_rs = np.zeros((max(mb) + 1, max(mb) + 1))
    for i in edgelist:
        # Please do check the index convention of the edgelist
        source_group = int(mb[int(i[0])])
        target_group = int(mb[int(i[1])])
        m_e_rs[source_group][target_group] += 1
        m_e_rs[target_group][source_group] += 1

    italic_i = 0.
    m_e_r = np.sum(m_e_rs, axis=1)
    for ind, e_val in enumerate(np.nditer(m_e_rs)):
        ind_i = int(np.floor(ind / (m_e_rs.shape[0])))
        ind_j = ind % (m_e_rs.shape[0])
        if e_val != 0.0:
            italic_i += e_val * np.log(
                e_val / m_e_r[ind_i] / m_e_r[ind_j]
            )
    ent += -italic_i / 2

    n_k = get_n_k_from_edgelist(edgelist)
    ent_ = 0
    for deg, k in enumerate(n_k):
        if deg != 0 and k != 0:
            ent_ -= k * gammaln(deg)
    ent += ent_

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
            print("P(e) and P(b|n_r) are {} and {}".format(lbinom(x + e - 1, e), partition_entropy(ka=ka, kb=kb, na=na, nb=nb, nr=nr, allow_empty=allow_empty)))
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
    import numpy as np
    n_r = np.zeros(max(mb) + 1)
    for block_id in mb:
        n_r[block_id] += 1
    n_r = list(map(int, n_r))
    return n_r


def get_n_k_from_edgelist(edgelist):
    """
    Get n_k, or the number n_k of nodes of degree k.

    Parameters
    ----------
    edgelist: array-like

    Returns
    -------
    n_k: array-like

    """
    assert np.min(edgelist).__int__() == 0, "The index of a node must start from 0."
    max_ = np.max(edgelist).__int__()
    k = np.zeros(max_ + 1)
    for edge in edgelist:
        k[edge[0]] += 1
        k[edge[1]] += 1

    max_ = np.max(k).__int__()
    n_k = np.zeros(max_ + 1)
    for k_ in k:
        n_k[k_.__int__()] += 1
    return n_k


def compute_degree_entropy(edgelist, mb, degree_dl_kind="uniform"):
    ent = 0
    n_r = np.zeros(max(mb) + 1)
    for mb_ in mb:
        n_r[mb_] += 1

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
        raise NotImplementedError
    elif degree_dl_kind == "entropy":
        raise NotImplementedError
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


def get_desc_len_from_data(na, nb, n_edges, ka, kb, edgelist, mb, diff=True, nr=None, allow_empty=False):
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

    Returns
    -------
    desc_len_b: `float`
        Difference of the description length to the bipartite ER network, per edge.

    """
    edgelist = list(map(lambda e: [int(e[0]), int(e[1])], edgelist))

    italic_i = compute_profile_likelihood(edgelist, mb, ka=ka, kb=kb)
    desc_len = 0.

    # finally, we compute the description length
    if diff:  # todo: add more options to it; now, only uniform prior for P(e) is included.
        desc_len += (na * np.log(ka) + nb * np.log(kb) - n_edges * (italic_i - np.log(2))) / n_edges
        x = float(ka * kb) / n_edges
        desc_len += (1 + x) * np.log(1 + x) - x * np.log(x)
        desc_len -= (1 + 1 / n_edges) * np.log(1 + 1 / n_edges) - (1 / n_edges) * np.log(1 / n_edges)
    else:
        num_edges = len(edgelist)
        desc_len += fitting_entropy(edgelist, mb)
        print("desc len from fitting {}".format(desc_len))
        desc_len += model_entropy(num_edges, ka=ka, kb=kb, na=na, nb=nb, nr=nr, allow_empty=allow_empty)  # P(e | b) + P(b | K)
        desc_len += compute_degree_entropy(edgelist, mb)  # P(k |e, b) - this term is as correct as in graph_tool
        print("degree dl = {}".format(compute_degree_entropy(edgelist, mb)))
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
