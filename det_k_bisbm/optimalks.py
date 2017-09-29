import os
import math
import random
import logging
import numpy as np
from collections import OrderedDict
from loky import get_reusable_executor


class OptimalKs(object):
    """Base class for OptimalKs.

    Parameters
    ----------
    edgelist : list, required
        Edgelist (bipartite network) for model selection.

    types : list, required
        Types of each node specifying the type membership.

    init_ka : int, required
        Initial Ka for successive merging and searching for the optimum.

    init_kb : int, required
        Initial Ka for successive merging and searching for the optimum.

    i_th :  double, optional
        Threshold for the merging step (as described in the main text).

    n_sweeps : int, optional
        Number of calculations performed on each (Ka, Kb) point.

    is_parallel : bool, optional
        Whether the n_sweeps calculation is forked for parallel processes.

    n_cores : int, optional
        If is_parallel == True, the number of cores that is used.

    """

    def __init__(self,
                 engine,
                 edgelist,
                 types,
                 init_ka=10,
                 init_kb=10,
                 i_th=0.1,
                 logging_level="info"):

        self._engine = engine.engine  # TODO: check that engine is an object
        self.MAX_NUM_SWEEPS = engine.MAX_NUM_SWEEPS
        self.PARALLELIZATION = engine.PARALLELIZATION
        self.NUM_CORES = engine.NUM_CORES

        # params for the heuristic
        self.ka = int(init_ka)
        self.kb = int(init_kb)
        self.ITALIC_I_THRESHOLD = float(i_th)
        self.adaptive_ratio = 0.9  # adaptive parameter to make the "delta" smaller, if it's too large

        # TODO: "types" is only used to compute na and nb. Can be made more generic.
        self.types = types
        self.NUM_NODES_A = 0
        self.NUM_NODES_B = 0
        for _type in types:
            if _type in ["1", 1]:
                self.NUM_NODES_A += 1
            elif _type in ["2", 2]:
                self.NUM_NODES_B += 1

        assert self.NUM_NODES_A > 0, "[ERROR] Number of type-a nodes = 0, which is not allowed"
        assert self.NUM_NODES_B > 0, "[ERROR] Number of type-b nodes = 0, which is not allowed"
        self.NUM_NODES = len(types)

        self.edgelist = edgelist
        self.NUM_EDGES = len(self.edgelist)

        assert self.NUM_NODES == self.NUM_NODES_A + self.NUM_NODES_B, \
            "[ERROR] num_nodes ({}) does not equal to num_nodes_a ({}) plus num_nodes_b ({})".format(
                self.NUM_NODES, self.NUM_NODES_A, self.NUM_NODES_B
            )

        # These confident_* variable are used to store the "true" data
        # that is, not the sloppy temporarily results via matrix merging
        self.confident_desc_len = OrderedDict()
        self.confident_m_e_rs = OrderedDict()
        self.confident_italic_i = OrderedDict()

        # These trace_* variable are used to store the data that we temporarily go through
        self.trace_mb = OrderedDict()

        # for debug/temp variables
        self.debug_str = ""
        self.f_edgelist = "edgelist-" + str(random.random()) + ".tmp"

        # initialize other class attributes
        self.INIT_ITALIC_I = 0.
        self.exist_bookkeeping = True

        # logging
        self._logger = logging.Logger
        self.set_logging_level(logging_level)
        pass

    def set_logging_level(self, level):
        _level = 0
        if level.upper() == "INFO":
            _level = logging.INFO
        elif level.upper() == "WARNING":
            _level = logging.WARNING
        logging.basicConfig(
            level=_level,
            format="%(asctime)s:%(levelname)s:%(message)s"
        )
        self._logger = logging.getLogger(__name__)

    def set_params(self, init_ka=10, init_kb=10, i_th=0.1):
        # params for the heuristic
        self.ka = int(init_ka)
        self.kb = int(init_kb)
        self.ITALIC_I_THRESHOLD = float(i_th)
        self.INIT_ITALIC_I = 0.

    def set_adaptive_ratio(self, adaptive_ratio):
        self.adaptive_ratio = float(adaptive_ratio)

    def set_exist_bookkeeping(self, exist_bookkeeping):
        self.exist_bookkeeping = bool(exist_bookkeeping)
        if not exist_bookkeeping:
            self._logger.warning("Setting <exist_bookkeeping> to false makes bad performance.")

    def clean(self):
        self.confident_desc_len = OrderedDict()
        self.confident_m_e_rs = OrderedDict()
        self.confident_italic_i = OrderedDict()
        self.trace_mb = OrderedDict()

    @staticmethod
    def get_italic_i_from_m_e_rs(m_e_rs):
        assert type(m_e_rs) is np.ndarray, "[ERROR] input parameter (m_e_rs) should be of type numpy.ndarray"
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
        return italic_i

    def _cal_desc_len(self, ka, kb, italic_i):
        desc_len_b = (
                         self.NUM_NODES_A * math.log(ka) + self.NUM_NODES_B * math.log(kb) - self.NUM_EDGES * italic_i
                     ) / self.NUM_EDGES
        x = float(ka * kb) / self.NUM_EDGES
        desc_len_b += (1 + x) * math.log(1 + x) - x * math.log(x)
        return desc_len_b

    @staticmethod
    def get_m_e_rs_from_mb(edgelist, mb):
        assert type(edgelist) is list, \
            "[ERROR] the type of the first input should be a list; however, it is {}".format(str(type(edgelist)))
        assert type(mb) is list, \
            "[ERROR] the type of the second input should be a list; however, it is {}".format(str(type(mb)))
        # construct e_rs matrix
        m_e_rs = np.zeros((max(mb) + 1, max(mb) + 1))
        for i in edgelist:
            # Please do check the index convention of the edgelist
            source_group = int(mb[int(i[0])])
            target_group = int(mb[int(i[1])])
            if source_group == target_group:
                raise ImportError("[ERROR] This is not a bipartite network!")
            m_e_rs[source_group][target_group] += 1
            m_e_rs[target_group][source_group] += 1

        m_e_r = np.sum(m_e_rs, axis=1)
        return m_e_rs, m_e_r

    @staticmethod
    def merge_matrix(ka, kb, m_e_rs):
        """
        Merge the rows of the affinity matrix (dim = K) to gain a reduced matrix (dim = K - 1)

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

        from_row = random.sample([0] * ka + [ka] * kb, 1)[0]
        a = m_e_rs[0:ka, ka:ka + kb]

        merge_list = list([0, 0])  # which two mb label should be merged together?
        mb_map = OrderedDict()
        new_ka = 0
        new_kb = 0

        if ka == 1:
            # do not merge type-a rows (this happens when <i_th> is set too high)
            from_row = ka
        elif kb == 1:
            from_row = 0

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
        if not from_row == 0:
            b = bt
            bt = b.transpose()
        for ind_i, _c in enumerate(c):
            for ind_j, __c in enumerate(_c):
                if ind_i >= new_ka and ind_j < new_ka:
                    c[ind_i][ind_j] = bt[ind_i - new_ka][ind_j]
                elif ind_i < new_ka and ind_j >= new_ka:
                    c[ind_i][ind_j] = b[ind_i][ind_j - new_ka]

        assert new_ka + new_kb == c.shape[0], "new_ka = {}; new_kb = {}; new_mat.shape[0] = {}".format(
            new_ka, new_kb, c.shape[0]
        )
        assert new_ka + new_kb == c.shape[1], "new_ka = {}; new_kb = {}; new_mat.shape[1] = {}".format(
            new_ka, new_kb, c.shape[1]
        )
        assert np.all(c.transpose() == c), "Error: output m_e_rs matrix is not symmetric!"
        return new_ka, new_kb, c, merge_list

    def _calc_with_hook(self, ka, kb, old_desc_len=None):
        """
        Execute the partitioning code by spawning child processes in the shell; save its output afterwards.

        Parameters
        ----------
        ka : int
            Number of type-a communities that one wants to partition on the bipartite graph
        kb : int
            Number of type-b communities that one wants to partition on the bipartite graph

        Returns
        -------
        italic_i : float
            the profile likelihood of the found partition

        m_e_rs : numpy array
            the affinity matrix via the group membership vector found by the partitioning engine

        mb : list[int]
            group membership vector calculated by the partitioning engine

        """
        # each time when you calculate/search at particular ka and kb
        # the hood records relevant information for research
        try:
            self.confident_desc_len[(ka, kb)]
        except KeyError as _:
            pass
        else:
            if self.confident_desc_len[(ka, kb)] != 0:
                italic_i = self.confident_italic_i[(ka, kb)]
                m_e_rs = self.confident_m_e_rs[(ka, kb)]
                mb = self.trace_mb[(ka, kb)]
                self._logger.info("... fetch calculated data ...")
                return italic_i, m_e_rs, mb

        def run(ka, kb):
            mb = self._engine(self.f_edgelist, self.NUM_NODES_A, self.NUM_NODES_B, ka, kb)
            m_e_rs, _ = self.get_m_e_rs_from_mb(self.edgelist, mb)
            italic_i = self.get_italic_i_from_m_e_rs(m_e_rs)
            new_desc_len = self._cal_desc_len(ka, kb, italic_i)

            return m_e_rs, italic_i, new_desc_len, mb

        # Calculate the biSBM inference several times,
        # choose the maximum likelihood result.
        # In other words, we choose the state with minimum entropy.
        results = []
        if old_desc_len is None:
            if self.PARALLELIZATION:
                # automatically shutdown after idling for 2s
                results = list(
                    self.executor(self.NUM_CORES, 2, lambda x: run(ka, kb), list(range(self.MAX_NUM_SWEEPS)))
                )
            else:
                results = [run(ka, kb)]
        else:
            old_desc_len = float(old_desc_len)
            if not self.PARALLELIZATION:
                # if old_desc_len is passed
                # we compare the new_desc_len with the old one
                # --
                # this option is used when we want to decide whether
                # we should escape from the local minimum during the heuristic
                calculate_times = 0
                while calculate_times < self.MAX_NUM_SWEEPS:
                    result = run(ka, kb)
                    results.append(result)
                    new_desc_len = self._cal_desc_len(ka, kb, result[1])
                    if new_desc_len < old_desc_len:
                        # no need to go further
                        calculate_times = self.MAX_NUM_SWEEPS
                    else:
                        calculate_times += 1
            else:
                results = list(
                    self.executor(self.NUM_CORES, 2, lambda x: run(ka, kb), list(range(self.MAX_NUM_SWEEPS)))
                )

        result = min(results, key=lambda x: x[2])
        mb = result[3]
        italic_i = result[1]
        m_e_rs = result[0]

        return italic_i, m_e_rs, mb

    @staticmethod
    def executor(max_workers, timeout, func, feeds):
        assert type(feeds) is list, "[ERROR] feeds should be a Python list; here it is {}".format(str(type(feeds)))
        executor = get_reusable_executor(max_workers=int(max_workers), timeout=int(timeout))
        results = executor.map(func, feeds)

        return results

    def _moving_one_step_down(self, ka, kb):
        """
        Apply multiple merges of the original affinity matrix, return the one that least alters the entropy

        Parameters
        ----------
        ka : int
            number of type-a communities in the affinity matrix
        kb : int
            number of type-b communities in the affinity matrix

        Returns
        -------
        _ka : int
            the new number of type-a communities in the affinity matrix

        _kb : int
            the new number of type-b communities in the affinity matrix

        _m_e_rs : numpy array
            the new affinity matrix

        diff_italic_i : list(int, int)
            the difference of the new profile likelihood and the old one

        _mlist : list(int, int)
            the two row-indexes of the original affinity matrix that were finally chosen (and merged)

        """
        if self.INIT_ITALIC_I == 0:
            # This is an important step, where we
            # (1) Calculate the heavy biSBM at init (ka, kb)
            # (2) Initiate important variables for logging and drawing
            _, m_e_rs, italic_i = self._calc_and_update((ka, kb))

            self.INIT_ITALIC_I = italic_i
            self.m_e_rs = m_e_rs

            # these are used to track temporarily variables during the heuristic
            self.diff_italic_I_array = [0.]
            self.ka_array = [ka]
            self.kb_array = [kb]

        def _sample_and_merge():
            _ka, _kb, _m_e_rs, _mlist = self.merge_matrix(self.ka, self.kb, self.m_e_rs)
            _italic_I = self.get_italic_i_from_m_e_rs(_m_e_rs)
            diff_italic_i = _italic_I - self.INIT_ITALIC_I  # diff_italic_i is always negative;
            return _ka, _kb, _m_e_rs, diff_italic_i, _mlist

        # how many times that a sample merging takes place
        indexes_to_run_ = range(0, (ka + kb) * 2)  # NOTE that the 2 is a hard-coded parameter

        results = []
        for _ in indexes_to_run_:
            results.append(_sample_and_merge())

        _ka, _kb, _m_e_rs, _diff_italic_i, _mlist = max(results, key=lambda x: x[3])

        assert int(_m_e_rs.sum()) == int(self.NUM_EDGES * 2), "__m_e_rs.sum() = {}; self.NUM_EDGES * 2 = {}".format(
            str(int(_m_e_rs.sum())), str(self.NUM_EDGES * 2)
        )

        return _ka, _kb, _m_e_rs, _diff_italic_i, _mlist

    def iterator(self):
        with open(self.f_edgelist, "w") as f:
            for edge in self.edgelist:
                f.write(str(edge[0]) + "\t" + edge[1] + "\n")
        ka_moving = self.ka
        kb_moving = self.kb
        while ka_moving != 1 or kb_moving != 1:
            old_ka_moving = ka_moving
            old_kb_moving = kb_moving

            ka_moving, kb_moving, t_m_e_rs, diff_italic_i, mlist = self._moving_one_step_down(ka_moving, kb_moving)
            if abs(diff_italic_i) > self.ITALIC_I_THRESHOLD * self.INIT_ITALIC_I:
                old_desc_len, _, _ = self._calc_and_update((old_ka_moving, old_kb_moving))
                if self._bookkeeping_condition(old_desc_len):
                    ka_moving, kb_moving = self._back_to_where_desc_len_is_lowest(diff_italic_i)
                else:
                    candidate_desc_len, _, _ = self._calc_and_update((ka_moving, kb_moving), old_desc_len)
                    t_m_e_rs_cand = self.confident_m_e_rs[(ka_moving, kb_moving)]
                    if candidate_desc_len > old_desc_len:  # candidate move is not a good choice
                        tmp = [
                            ka_moving - old_ka_moving,
                            kb_moving - old_kb_moving
                        ]
                        tmp.reverse()
                        ka_moving = old_ka_moving + tmp[0]
                        kb_moving = old_kb_moving + tmp[1]
                        candidate_desc_len, _, _ = self._calc_and_update((ka_moving, kb_moving), old_desc_len)
                        t_m_e_rs_cand = self.confident_m_e_rs[(ka_moving, kb_moving)]
                        if candidate_desc_len > old_desc_len:  # candidate move is not a good choice
                            ka_moving = old_ka_moving - 1
                            kb_moving = old_kb_moving - 1
                            candidate_desc_len, _, _ = self._calc_and_update((ka_moving, kb_moving), old_desc_len)
                            t_m_e_rs_cand = self.confident_m_e_rs[(ka_moving, kb_moving)]
                            if candidate_desc_len > old_desc_len:  # candidate move is not a good choice
                                # Before we conclude anything,
                                # we check all the other points near here.
                                self._logger.info("check all the adjacent points near ({}, {})".format(
                                    old_ka_moving, old_kb_moving))
                                items = map(lambda x: (
                                    x[0] + old_ka_moving,
                                    x[1] + old_kb_moving
                                ), [(1, 0), (0, 1), (1, -1), (-1, 1), (+1, +1)]
                                            )
                                for item in items:
                                    self._calc_and_update(item, old_desc_len)

                                if self._bookkeeping_condition(old_desc_len):
                                    p_estimate = sorted(self.confident_desc_len, key=self.confident_desc_len.get)[0]
                                    self._logger.info(
                                        "Found ({}, {}) that gives a even lower description length ...".format(
                                            p_estimate[0], p_estimate[1])
                                    )
                                    self._back_to_where_desc_len_is_lowest(diff_italic_i)
                                else:
                                    # clean up
                                    try:
                                        os.remove(self.f_edgelist)
                                        os.remove("edgelist-*.tmp")
                                    finally:
                                        p_estimate = sorted(self.confident_desc_len, key=self.confident_desc_len.get)[0]
                                        self._logger.info("DONE: the MDL point is ({},{})".format(
                                            p_estimate[0], p_estimate[1]
                                        ))
                                        return self.confident_desc_len
                            else:
                                # continue moving with the
                                # new candidate's direction
                                self._update_current_state((ka_moving, kb_moving), t_m_e_rs_cand)
                        else:
                            # continue moving w/ the new candidate's direction
                            self._update_current_state((ka_moving, kb_moving), t_m_e_rs_cand)
                    else:
                        # candidate-1 might be a good choice, but ....
                        # continue moving w/ the new candidate's direction
                        self._update_current_state((ka_moving, kb_moving), t_m_e_rs_cand)
            else:
                old_of_g = self.trace_mb[(self.ka, self.kb)]
                new_of_g = list(np.zeros(self.NUM_NODES))

                mlist.sort()
                for _node_id, _g in enumerate(old_of_g):
                    if _g == mlist[1]:
                        new_of_g[_node_id] = mlist[0]
                    elif _g < mlist[1]:
                        new_of_g[_node_id] = _g
                    else:
                        new_of_g[_node_id] = _g - 1
                assert max(new_of_g) + 1 == ka_moving + kb_moving, \
                    "[ERROR] inconsistency between the membership indexes and the number of blocks."
                self.trace_mb[(ka_moving, kb_moving)] = new_of_g
                self._update_current_state((ka_moving, kb_moving), t_m_e_rs)
            # for drawing...
            self._iter_calc_hook(diff_italic_i)

        # clean up
        if self.ka == 1 and self.kb == 1:
            # if we reached (1, 1), check that it's the local optimal point, then we could return (1, 1).
            points_to_compute = [(1, 1), (1, 2), (2, 1), (2, 2)]
            for point in points_to_compute:
                self.compute_and_update(point[0], point[1], recompute=True)
            p_estimate = sorted(self.confident_desc_len, key=self.confident_desc_len.get)[0]
            if p_estimate != (1, 1):
                raise UserWarning(
                    "[WARNING] merging reached (1, 1); cannot go any further, please set a smaller <i_th>."
                )
        try:
            os.remove(self.f_edgelist)
            os.remove("edgelist-*.tmp")
        finally:
            p_estimate = sorted(self.confident_desc_len, key=self.confident_desc_len.get)[0]
            self._logger.info("DONE: the MDL point is ({},{})".format(
                p_estimate[0], p_estimate[1]
            ))
            return self.confident_desc_len

    def _bookkeeping_condition(self, old_desc_len):
        if self.exist_bookkeeping:
            return any(i < old_desc_len for i in self.confident_desc_len.values())
        else:
            return False

    def _back_to_where_desc_len_is_lowest(self, diff_italic_i):
        self._iter_calc_hook(diff_italic_i)
        self.ka = sorted(self.confident_desc_len, key=self.confident_desc_len.get, reverse=False)[0][0]
        self.kb = sorted(self.confident_desc_len, key=self.confident_desc_len.get, reverse=False)[0][1]
        self.ITALIC_I_THRESHOLD *= self.adaptive_ratio
        self.m_e_rs = self.confident_m_e_rs[(self.ka, self.kb)]
        return self.ka, self.kb

    def _iter_calc_hook(self, diff_italic_i):
        self.diff_italic_I_array.append(diff_italic_i)
        self.ka_array.append(self.ka)
        self.kb_array.append(self.kb)
        return

    def _update_current_state(self, point, m_e_rs):
        self.ka = point[0]
        self.kb = point[1]
        # this will be used in _moving_one_step_down function
        self.m_e_rs = m_e_rs
        return

    def _calc_and_update(self, point, old_desc_len=0.):
        self._logger.info("Now computing graph partition at ({}, {})".format(point[0], point[1]) + " ...")
        if old_desc_len == 0.:
            italic_i, m_e_rs, mb = self._calc_with_hook(point[0], point[1], old_desc_len=None)
        else:
            italic_i, m_e_rs, mb = self._calc_with_hook(point[0], point[1], old_desc_len=old_desc_len)
        candidate_desc_len = self._cal_desc_len(point[0], point[1], italic_i)
        self.confident_desc_len[point] = candidate_desc_len
        self.confident_italic_i[point] = italic_i
        self.confident_m_e_rs[point] = m_e_rs
        assert max(mb) + 1 == point[0] + point[1], \
            "[ERROR] inconsistency between the membership indexes and the number of blocks."
        self.trace_mb[point] = mb
        self._logger.info("... DONE.")
        return candidate_desc_len, m_e_rs, italic_i

    def compute_and_update(self, ka, kb, recompute=False):
        with open(self.f_edgelist, "w") as f:
            for edge in self.edgelist:
                f.write(str(edge[0]) + "\t" + edge[1] + "\n")
        if recompute:
            self.confident_desc_len[(ka, kb)] = 0
        self._calc_and_update((ka, kb))
        try:
            os.remove(self.f_edgelist)
        finally:
            pass
