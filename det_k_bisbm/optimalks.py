import os
import logging
import tempfile

from det_k_bisbm.utils import *


class OptimalKs(object):
    r"""Base class for OptimalKs.

    Parameters
    ----------
    engine : :class:`engine`, required
        The inference engine class.

    edgelist : iterable or :class:`numpy.ndarray`, required
        Edgelist (bipartite network) for model selection.

    types : iterable or :class:`numpy.ndarray`, required
        Types of each node specifying the type membership.

    verbose : bool, optional
        Logging level used.

    default_args :  bool, optional

    random_init_k : bool, optional

    bipartite_prior : bool, optional

    """

    def __init__(self,
                 engine,
                 edgelist,
                 types,
                 verbose=False,
                 default_args=True,
                 random_init_k=False,
                 bipartite_prior=True):

        self.engine_ = engine.engine  # TODO: check that engine is an object
        self.max_n_sweeps_ = engine.MAX_NUM_SWEEPS
        self.is_par_ = engine.PARALLELIZATION
        self.n_cores_ = engine.NUM_CORES
        self.algm_name_ = engine.ALGM_NAME
        self.virgin_run = True

        self.bm_state = dict()
        self.bm_state["types"] = types  # TODO: "types" is only used to compute na and nb. Can be made more generic.
        self.bm_state["n_a"] = 0
        self.bm_state["n_b"] = 0
        self.bm_state["n"] = 0
        self.bm_state["ref_dl"] = 0
        self.bm_state["e_rs"] = None
        self.bm_state["mb"] = list()

        for _type in types:
            self.bm_state["n"] += 1
            if _type in ["1", 1]:
                self.bm_state["n_a"] += 1
            elif _type in ["2", 2]:
                self.bm_state["n_b"] += 1

        self.edgelist = np.array(edgelist, dtype=np.uint32)
        self.bm_state["e"] = len(self.edgelist)
        self.i_0s = []
        if engine.ALGM_NAME == "mcmc" and default_args:
            engine.set_steps(self.bm_state["n"] * 1e5)
            engine.set_await_steps(self.bm_state["n"] * 2e3)
            engine.set_cooling("abrupt_cool")
            engine.set_cooling_param_1(self.bm_state["n"] * 1e3)
            engine.set_epsilon(1.)
        if default_args:
            self.bm_state["ka"] = int(self.bm_state["e"] ** 0.5 / 2)
            self.bm_state["kb"] = int(self.bm_state["e"] ** 0.5 / 2)
            self.i_0 = 1.
            self.adaptive_ratio = 0.95  # adaptive parameter to make the "i_0" smaller, if it's overshooting.
            self._k_th_nb_to_search = 2
            self._nm = 10
        else:
            self.bm_state["ka"] = self.bm_state["kb"] = self.i_0 = \
                self.adaptive_ratio = self._k_th_nb_to_search = self._nm = None
        if random_init_k:
            self.bm_state["ka"] = np.random.randint(1, self.bm_state["ka"] + 1)
            self.bm_state["kb"] = np.random.randint(1, self.bm_state["kb"] + 1)

        # These confident_* variable are used to store the "true" data
        # that is, not the sloppy temporarily results via matrix merging
        self.bookkeeping_dl = OrderedDict()
        self.bookkeeping_e_rs = OrderedDict()

        # These trace_* variable are used to store the data that we temporarily go through
        self.trace_mb = OrderedDict()

        # for debug/temp variables
        self.is_tempfile_existed = True
        self.f_edgelist = tempfile.NamedTemporaryFile(mode='w', delete=False)
        # self.f_edgelist = tempfile.NamedTemporaryFile(mode='w', dir='/scratch/Users/tzye5331/.tmp/', delete=False)
        # To prevent "TypeError: cannot serialize '_io.TextIOWrapper' object" when using loky
        self._f_edgelist_name = self._get_tempfile_edgelist()

        # logging
        if verbose:
            _logging_level = "INFO"
        else:
            _logging_level = "WARNING"
        self._logger = logging.Logger
        self.set_logging_level(_logging_level)
        self._summary = OrderedDict()
        self._summary["init_ka"] = self.bm_state["ka"]
        self._summary["init_kb"] = self.bm_state["kb"]
        self._summary["na"] = self.bm_state["n_a"]
        self._summary["nb"] = self.bm_state["n_b"]
        self._summary["e"] = self.bm_state["e"]
        self._summary["avg_k"] = 2 * self.bm_state["e"] / (self.bm_state["n_a"] + self.bm_state["n_b"])

        # look-up tables
        self.__q_cache_f_name = os.path.join(tempfile.mkdtemp(), '__q_cache.dat')  # for restricted integer partitions
        # self.__q_cache_f_name = os.path.join(tempfile.mkdtemp(dir='/scratch/Users/tzye5331/.tmp/'), '__q_cache.dat')
        self.__q_cache = np.array([], ndmin=2)
        self.__q_cache_max_e_r = self.bm_state["e"] if self.bm_state["e"] <= int(1e4) else int(1e4)

        max_e_r = self.__q_cache_max_e_r
        fp = np.memmap(self.__q_cache_f_name, dtype='uint32', mode="w+", shape=(max_e_r + 1, max_e_r + 1))
        self.__q_cache = init_q_cache(max_e_r, np.array([], ndmin=2))
        fp[:] = self.__q_cache[:]
        del fp
        # self.__q_cache = np.memmap(self.__q_cache_f_name, dtype='uint32', mode='r', shape=(max_e_r + 1, max_e_r + 1))

        self.bipartite_prior_ = bipartite_prior

    def iterator(self, bipartite_prior=True):
        self.bipartite_prior_ = bipartite_prior
        self._prerunning_checks()
        if not self.is_tempfile_existed:
            self._f_edgelist_name = self._get_tempfile_edgelist()

        self._compute_dl_and_update(1, 1)
        if self.algm_name_ == "mcmc" and self.virgin_run:
            self._natural_merge()

        if self._check_if_local_minimum(self.bm_state["ka"], self.bm_state["kb"]):
            self._clean_up()
            return self.bookkeeping_dl
        else:
            diff_dl = 0.
            while abs(diff_dl) < self.i_0 * self.bm_state["ref_dl"]:
                if self.bm_state["ka"] * self.bm_state["kb"] != 1:
                    ka_, kb_, diff_dl, mlist = self._merge_e_rs(self.bm_state["ka"], self.bm_state["kb"])
                    if self._determine_i_0(diff_dl):
                        break
                    mb_ = accept_mb_merge(self.bm_state["mb"], mlist)
                    e_rs = assemble_e_rs_from_mb(self.edgelist, mb_)
                    assert int(e_rs.sum()) == int(
                        self.bm_state["e"] * 2), '__m_e_rs.sum() = {}; self.bm_state["e"] * 2 = {}'.format(
                        str(int(e_rs.sum())), str(self.bm_state["e"] * 2)
                    )
                    self._update_bm_state(ka_, kb_, e_rs, mb_)
                    self._logger.info(f"Merging to ({ka_}, {kb_})")
                else:
                    break
            self._logger.info("Escape while-loop, Re-do iterator().")
            return self.iterator(bipartite_prior=self.bipartite_prior_)

    def summary(self):
        ka, kb = sorted(self.bookkeeping_dl, key=self.bookkeeping_dl.get)[0]
        self._summary["ka"] = ka
        self._summary["kb"] = kb
        self._summary["mdl"] = self.bookkeeping_dl[(ka, kb)]
        return self._summary

    def compute_and_update(self, ka, kb, recompute=False):
        try:
            os.remove(self._f_edgelist_name)
        except FileNotFoundError as e:
            self._logger.warning(f"FileNotFoundError: {e}")
        finally:
            self._f_edgelist_name = self._get_tempfile_edgelist()
            q_cache = np.array([], ndmin=2)
            q_cache = init_q_cache(self.__q_cache_max_e_r, q_cache)
            self.set__q_cache(q_cache)  # todo: add some checks here
            if recompute:
                self.bookkeeping_dl[(ka, kb)] = 0
            self._compute_dl_and_update(ka, kb, recompute=recompute)

    def compute_dl(self, ka, kb, recompute=False):
        r"""Execute the partitioning code by spawning child processes in the shell; saves its output afterwards.

        Parameters
        ----------
        ka : int
            Number of type-a communities that one wants to partition on the bipartite graph
        kb : int
            Number of type-b communities that one wants to partition on the bipartite graph

        Returns
        -------
        dl : float
            the description length of the found partition

        e_rs : :class:`numpy.ndarray`
            the affinity matrix via the group membership vector found by the partitioning engine

        mb : list[int]
            group membership vector calculated by the partitioning engine

        """
        # each time when you calculate/search at particular ka and kb
        # the hood records relevant information for research
        try:
            self.bookkeeping_dl[(ka, kb)]
        except KeyError as _:
            pass
        else:
            if self.bookkeeping_dl[(ka, kb)] > 0:
                return self.bookkeeping_dl[(ka, kb)], self.bookkeeping_e_rs[(ka, kb)], self.trace_mb[(ka, kb)]

        if ka == 1 and kb == 1:
            mb = np.array([0] * self.bm_state["n_a"] + [1] * self.bm_state["n_b"])
            res = self._compute_desc_len(self.bm_state["n_a"], self.bm_state["n_b"], self.bm_state["e"], ka, kb, mb)
            return res[0], res[1], res[2]

        if not recompute:
            ka_ = self._summary["init_ka"]
            kb_ = self._summary["init_kb"]
            dist = np.sqrt((ka_ - ka) ** 2 + (kb_ - kb) ** 2)
            if dist <= self._k_th_nb_to_search * np.sqrt(2):
                _mb = None
            else:
                self._logger.info(f"DIST={dist}; agg merge from ({ka_}, {kb_}) to ({ka}, {kb}).")
                _mb = self.trace_mb[(ka_, kb_)]
        else:
            _mb = None

        if self.algm_name_ == "mcmc":
            run = lambda a, b: self.engine_(self._f_edgelist_name, self.bm_state["n_a"], self.bm_state["n_b"], a, b,
                                            mb=_mb)
        else:
            run = lambda a, b: self.engine_(self._f_edgelist_name, self.bm_state["n_a"], self.bm_state["n_b"], a, b)

        # Calculate the biSBM inference several times,
        # choose the maximum likelihood (or minimum entropy) result.
        results = []
        if self.is_par_:
            # automatically shutdown after idling for 600s
            results = list(loky_executor(self.n_cores_, 600, lambda x: run(ka, kb), list(range(self.max_n_sweeps_))))
        else:
            for _ in range(self.max_n_sweeps_):
                results += [run(ka, kb)]

        result_ = [self._compute_desc_len(self.bm_state["n_a"], self.bm_state["n_b"], self.bm_state["e"], ka, kb, r) for
                   r in results]
        result = min(result_, key=lambda x: x[0])
        dl = result[0]
        e_rs = result[1]
        mb = result[2]
        return dl, e_rs, mb

    def natural_merge(self):
        run = lambda dummy: self.engine_(self._f_edgelist_name, self.bm_state["n_a"], self.bm_state["n_b"], 1, 1,
                                         mb=None, method="natural")  # Note: setting (ka, kb) = (1, 1) is redundant.
        results = []
        if self.is_par_:
            # automatically shutdown after idling for 600s
            results = list(loky_executor(self.n_cores_, 600, lambda x: run(0), list(range(self.max_n_sweeps_))))
        else:
            for _ in range(self.max_n_sweeps_):
                results += [run(0)]

        result_ = [self._compute_desc_len(
            self.bm_state["n_a"], self.bm_state["n_b"], self.bm_state["e"], r[0], r[1], r[2:]
        ) for r in results]
        result = min(result_, key=lambda x: x[0])
        dl = result[0]
        e_rs = result[1]
        mb = result[2]
        ka, kb = result[3]
        return dl, e_rs, mb, ka, kb

    def _natural_merge(self):
        dl, e_rs, mb, ka, kb = self.natural_merge()
        self._summary["init_ka"] = ka
        self._summary["init_kb"] = kb
        self._logger.info(f"Natural agglomerative merge to ({ka}, {kb}).")
        self.bookkeeping_dl[(ka, kb)] = dl
        self.bookkeeping_e_rs[(ka, kb)] = e_rs
        assert max(mb) + 1 == ka + kb, "[ERROR] inconsistency between mb. indexes and #blocks. {} != {}".format(
            max(mb) + 1, ka + kb)
        self.trace_mb[(ka, kb)] = mb
        self._update_bm_state(ka, kb, e_rs, mb)
        self.virgin_run = False

    def _determine_i_0(self, diff_dl):
        if self.i_0 < 1:
            return False
        i_0 = diff_dl / self.bm_state["ref_dl"]
        self.i_0s += [i_0]
        iqr = np.percentile(self.i_0s, 75) - np.percentile(self.i_0s, 25)
        if i_0 > 3 * iqr + np.percentile(self.i_0s, 75):
            self.i_0 = i_0
            return True
        else:
            return False

    def _compute_desc_len(self, n_a, n_b, e, ka, kb, mb):
        e_rs = assemble_e_rs_from_mb(self.edgelist, mb)
        nr = assemble_n_r_from_mb(mb)
        desc_len = get_desc_len_from_data(n_a, n_b, e, ka, kb, list(self.edgelist), mb, nr=nr, q_cache=self.__q_cache,
                                          is_bipartite=self.bipartite_prior_)
        return desc_len, e_rs, mb, (ka, kb)

    def _merge_e_rs(self, ka, kb):
        """Apply multiple merges of the original affinity matrix, return the one that least alters the entropy

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

        diff_italic_i : list(int, int)
            the difference of the new profile likelihood and the old one

        _mlist : list(int, int)
            the two row-indexes of the original affinity matrix that were finally chosen (and merged)

        """
        m = np.arange(ka + kb)

        mlist = set()
        while len(mlist) == 0:
            for _m in m:
                pool = random.choices(m, k=self._nm)
                _mlist = [[min(x, _m), max(x, _m)] for x in pool]
                for _ in _mlist:
                    cond = (_[0] != _[1]) and not (_[1] >= ka > _[0]) and not (_[0] == 0 and ka == 1) and not (
                            _[0] == ka and kb == 1)
                    if cond:
                        mlist.add(str(_[0]) + "+" + str(_[1]))

        diff_dl, _mlist = virtual_moves_ds(self.bm_state["e_rs"], mlist, self.bm_state["ka"])
        if max(_mlist) < self.bm_state["ka"]:
            ka = self.bm_state["ka"] - 1
            kb = self.bm_state["kb"]
        else:
            ka = self.bm_state["ka"]
            kb = self.bm_state["kb"] - 1
        return ka, kb, diff_dl, _mlist

    def _clean_up(self):
        try:
            os.remove(self._f_edgelist_name)
        except FileNotFoundError as e:
            self._logger.warning(f"FileNotFoundError: {e}")
        try:
            os.remove(self.__q_cache_f_name)
        except FileNotFoundError as e:
            self._logger.warning(f"FileNotFoundError: {e}")
        self.is_tempfile_existed = False

    def _rollback(self):
        dl = self.summary()["mdl"]
        ka = self.summary()["ka"]
        kb = self.summary()["kb"]
        e_rs = self.bookkeeping_e_rs[(ka, kb)]
        mb = self.trace_mb[(ka, kb)]
        self._update_bm_state(ka, kb, e_rs, mb)
        return ka, kb, e_rs, dl

    def _update_bm_state(self, ka, kb, e_rs, mb):
        self.bm_state["ka"] = ka
        self.bm_state["kb"] = kb
        self.bm_state["e_rs"] = e_rs
        self.bm_state["mb"] = mb

    def _compute_dl_and_update(self, ka, kb, recompute=False):
        dl, e_rs, mb = self.compute_dl(ka, kb, recompute=recompute)
        self.bookkeeping_dl[(ka, kb)] = dl
        self.bookkeeping_e_rs[(ka, kb)] = e_rs
        assert max(mb) + 1 == ka + kb, "[ERROR] inconsistency between mb. indexes and #blocks. {} != {}".format(
            max(mb) + 1, ka + kb)
        self.trace_mb[(ka, kb)] = mb
        self.bm_state["ref_dl"] = self.summary()["mdl"] if self.bm_state["ref_dl"] != 0 else dl
        return dl, e_rs, mb

    # ###########
    # Checkpoints
    # ###########
    def _is_mdl_so_far(self, desc_len):
        """Check if `desc_len` is the minimal value so far."""
        return not any([i < desc_len for i in self.bookkeeping_dl.values()])

    def _check_if_local_minimum(self, ka, kb):
        """The `neighborhood search` as described in the paper."""
        self.is_tempfile_existed = True
        k_th = self._k_th_nb_to_search
        self._logger.info(f"Checking if {(ka, kb)} is a local minimum.")
        _dl, _e_rs, _mb = self._compute_dl_and_update(ka, kb)
        if _dl > self.bookkeeping_dl[(1, 1)]:
            self._logger.info("DL at ({}, {}) is larger than that of (1, 1), which is {} compared to {}".format(
                ka, kb, _dl, self.bookkeeping_dl[(1, 1)])
            )
            self._update_bm_state(ka, kb, _e_rs, _mb)
            return False

        if _dl > self.summary()["mdl"]:
            self.i_0 *= self.adaptive_ratio
            ka, kb, _, _dl = self._rollback()
            self._logger.info("There's already a point with lower dl; we are overshooting. Let's reduce i_0.")
            self._logger.info(f"Re-Checking if {(ka, kb)} is a local minimum.")

        nb_points = map(lambda x: (x[0] + ka, x[1] + kb), product(range(-k_th, k_th + 1), repeat=2))
        # if any item has values less than 1, delete it. Also, exclude the suspected point (i.e., [ka, kb]).
        nb_points = [(i, j) for i, j in nb_points if i >= 1 and j >= 1 and (i, j) != (ka, kb)]

        for _ka, _kb in nb_points:
            dl, _, _ = self._compute_dl_and_update(_ka, _kb)
            if self._is_mdl_so_far(dl):
                self._logger.info(f"Found {(_ka, _kb)} that gives an even lower description length ...")
                self._rollback()
                self._logger.info("rollback but NOT reduce i_0")
                break
        if _dl != self.summary()["mdl"]:
            self._logger.info(f"No, {(ka, kb)} is NOT a local minimum")
            return False
        else:
            self._logger.info(f"Yes, {(ka, kb)} is a local minimum; we are done.")
            return True

    def _prerunning_checks(self):
        assert self.bm_state["n_a"] > 0, "[ERROR] Number of type-a nodes = 0, which is not allowed"
        assert self.bm_state["n_b"] > 0, "[ERROR] Number of type-b nodes = 0, which is not allowed"
        assert self.bm_state["n"] == self.bm_state["n_a"] + self.bm_state["n_b"], \
            "[ERROR] num_nodes ({}) does not equal to num_nodes_a ({}) plus num_nodes_b ({})".format(
                self.bm_state["n"], self.bm_state["n_a"], self.bm_state["n_b"]
            )
        if self.bm_state["ka"] is None or self.bm_state["kb"] is None or self.i_0 is None:
            raise AttributeError("Arguments missing! Please assign `init_ka`, `init_kb`, and `i_0`.")
        if self.adaptive_ratio is None:
            raise AttributeError("Arguments missing! Please assign `adaptive_ratio`.")
        if self._k_th_nb_to_search is None:
            raise AttributeError("Arguments missing! Please assign `k_th_nb_to_search`.")
        if self._nm is None:
            raise AttributeError("Arguments missing! Please assign `size_rows_to_run`.")

    # #######################
    # Set & Get of parameters
    # #######################
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

    def set_params(self, init_ka=10, init_kb=10, i_0=0.01):
        # params for the heuristic
        self.bm_state["ka"] = int(init_ka)
        self.bm_state["kb"] = int(init_kb)
        self.i_0 = float(i_0)
        assert 0. <= self.i_0 < 1, "[ERROR] Allowed range for i_0 is [0, 1)."
        assert self.bm_state["ka"] <= self.bm_state[
            "n_a"], "[ERROR] Number of type-a communities must be smaller than the # nodes in type-a."
        assert self.bm_state["kb"] <= self.bm_state[
            "n_b"], "[ERROR] Number of type-b communities must be smaller than the # nodes in type-b."
        self._summary["init_ka"] = self.bm_state["ka"]
        self._summary["init_kb"] = self.bm_state["kb"]

    def set_adaptive_ratio(self, adaptive_ratio):
        self.adaptive_ratio = float(adaptive_ratio)

    def set_k_th_neighbor_to_search(self, k):
        self._k_th_nb_to_search = int(k)

    def set_nm(self, s):
        self._nm = int(s)

    def get__q_cache(self):
        return self.__q_cache

    def set__q_cache(self, q_cache):
        self.__q_cache = q_cache
        fp = np.memmap(self.__q_cache_f_name, dtype='uint32', mode="w+", shape=(q_cache.shape[0], q_cache.shape[1]))
        fp[:] = self.__q_cache[:]
        del fp

    def _get_tempfile_edgelist(self):
        try:
            self.f_edgelist.seek(0)
        except AttributeError:
            self.f_edgelist = tempfile.NamedTemporaryFile(mode='w', delete=False)
        finally:
            for edge in self.edgelist:
                self.f_edgelist.write(str(edge[0]) + "\t" + str(edge[1]) + "\n")
            self.f_edgelist.flush()
            f_edgelist_name = self.f_edgelist.name
            del self.f_edgelist
        return f_edgelist_name
