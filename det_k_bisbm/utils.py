""" utilities """
import numpy as np
import math
import logging
import inspect


def gen_equal_partition(n, total):
    all_nodes = np.arange(total)
    n_blocks = list(map(len, np.array_split(all_nodes, n)))

    return n_blocks


def gen_equal_bipartite_partition(na, nb, ka, kb):
    n_blocks = map(int, gen_equal_partition(ka, na) + gen_equal_partition(kb, nb))
    n = []
    for idx, i in enumerate(n_blocks):
        n += [idx] * i
    return n


def get_n_r_from_mb(mb):
    '''
        Get n_r vector (number of nodes in each group) from the membership vector.
    '''
    assert type(mb) is list, "ERROR: the type of the input parameter should be a list"
    import numpy as np
    n_r = np.zeros(max(mb) + 1)
    for block_id in mb:
        n_r[block_id] += 1
    n_r = list(map(int, n_r))
    return n_r


def get_desc_len_from_data(na, nb, n_edges, ka, kb, edgelist, mb):
    assert type(edgelist) is list, "[ERROR] the type of the input parameter (edgelist) should be a list"
    assert type(mb) is list, "[ERROR] the type of the input parameter (mb) should be a list"
    # First, let's compute the m_e_rs from the edgelist and mb
    m_e_rs = np.zeros((max(mb) + 1, max(mb) + 1))
    for i in edgelist:
        # Please do check the index convention of the edgelist
        source_group = int(mb[int(i[0])])
        target_group = int(mb[int(i[1])])
        if source_group == target_group:
            raise ImportError("This is not a bipartite network!")
        m_e_rs[source_group][target_group] += 1
        m_e_rs[target_group][source_group] += 1

    # then, we compute the profile likelihood from the m_e_rs
    italic_i = 0.
    m_e_r = np.sum(m_e_rs, axis=1)
    num_edges = m_e_r.sum() / 2.
    for ind, e_val in enumerate(np.nditer(m_e_rs)):
        ind_i = int(math.floor(ind / (m_e_rs.shape[0])))
        ind_j = ind % (m_e_rs.shape[0])
        if e_val != 0.0:
            italic_i += e_val / 2. / num_edges * math.log(
                e_val / m_e_r[ind_i] / m_e_r[ind_j] * 2 * num_edges
            )
    assert m_e_rs.shape[0] == ka + kb, "[ERROR] m_e_rs dimension (={}) is not equal to ka (={}) + kb (={})!".format(
        m_e_rs.shape[0], ka, kb
    )

    # finally, we compute the description length
    desc_len_b = (na * math.log(ka) + nb * math.log(kb) - n_edges * italic_i) / n_edges
    x = float(ka * kb) / n_edges
    desc_len_b += (1 + x) * math.log(1 + x) - x * math.log(x)

    return desc_len_b
