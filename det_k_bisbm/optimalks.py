import os
import logging
import tempfile
import random
from collections import OrderedDict

from det_k_bisbm.utils import *


class OptimalKs(object):
    """Base class for OptimalKs.

    Parameters
    ----------
    engine : :class:`engine` (required)
        The inference engine class.

    edgelist : ``iterable`` or :class:`numpy.ndarray`, required
        Edgelist (bipartite network) for model selection.

    types : ``iterable`` or :class:`numpy.ndarray`, required
        Types of each node specifying the type membership.

    verbose : ``bool`` (optional, default: ``False``)
        Logging level used. If ``True``, progress information will be shown.

    default_args :  ``bool`` (optional, default: ``True``)
        Arguments for initiating the heuristic. If ``False``, we should use :func:`set_params`,
        :func:`set_adaptive_ratio`, :func:`set_k_th_neighbor_to_search`, and :func:`set_nm` to set the default
        parameters.

    random_init_k : ``bool`` (optional, default: `False`)

    bipartite_prior : ``bool`` (optional, default: ``True``)
        Whether to use the bipartite prior for edge counts for the algorithm. If ``False``, we will pretend that we
        do not assume a bipartite structure at the level of the :math:`e_{rs}` matrix. This is what has been used in
        :func:`graph_tool.inference.minimize.minimize_blockmodel_dl`, even if its ``state_args`` is customized to be bipartite.

    tempdir : ``str`` (optional, default: ``None``)
        The directory entry for generated temporary network data, which will be removed immediately after the
        class is deleted. If ``tempdir`` is unset or ``None``, we will search for a standard list of directories
        and sets tempdir to the first one which the calling user can create files in, as :func:`tempfile.gettempdir`
        dictates. We will pass the file to the inference :mod:`engines`.

    """

    def __init__(self,
                 engine,
                 edgelist,
                 types,
                 verbose=False,
                 default_args=True,
                 random_init_k=False,
                 bipartite_prior=True,
                 tempdir=None):

        self.engine_ = engine.engine  # TODO: check that engine is an object
        self.max_n_sweeps_ = engine.MAX_NUM_SWEEPS
        self.is_par_ = engine.PARALLELIZATION
        self.n_cores_ = engine.NUM_CORES
        self.algm_name_ = engine.ALGM_NAME
        self._virgin_run = True

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

        self.edgelist = np.array(edgelist, dtype=np.uint64)
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
            self.adaptive_ratio = 0.9  # adaptive parameter to make the "i_0" smaller, if it's overshooting.
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
        self.trace_k = []  # only for painter.paint_trace

        # for debug/temp variables
        self.tempdir = tempdir

        self.__del__no_call = True
        if self.is_par_:
            # To prevent "TypeError: cannot serialize '_io.TextIOWrapper' object" when using loky
            self.f_edgelist = tempfile.NamedTemporaryFile(mode='w+b', dir=tempdir, delete=False)
        else:
            self.f_edgelist = tempfile.NamedTemporaryFile(mode='w+b', dir=tempdir, delete=True)
        self._f_edgelist_name = self._get_tempfile_edgelist()

        # logging
        if verbose:
            _logging_level = "INFO"
        else:
            _logging_level = "WARNING"
        self._logger = logging.Logger
        self._set_logging_level(_logging_level)
        self._summary = dict()
        self._summary["algm_args"] = {}
        self._summary["algm_args"]["init_ka"] = self.bm_state["ka"]
        self._summary["algm_args"]["init_kb"] = self.bm_state["kb"]
        self._summary["na"] = self.bm_state["n_a"]
        self._summary["nb"] = self.bm_state["n_b"]
        self._summary["e"] = self.bm_state["e"]
        self._summary["avg_k"] = 2 * self.bm_state["e"] / (self.bm_state["n_a"] + self.bm_state["n_b"])

        # look-up tables
        self.__q_cache_max_e_r = self.bm_state["e"] if self.bm_state["e"] <= int(1e4) else int(1e4)
        self.__q_cache = init_q_cache(self.__q_cache_max_e_r)

        self.bipartite_prior_ = bipartite_prior

    def minimize_bisbm_dl(self, bipartite_prior=True):
        """Fit the bipartite stochastic block model, by minimizing its description length using an agglomerative
        heuristic.

        Parameters
        ----------
        bipartite_prior : ``bool`` (optional, default ``True``)

        Returns
        -------
        OptimalKs.bookkeeping_dl : :py:class:`collections.OrderedDict`

        References
        ----------
        .. [yen-bipartite-2019] Tzu-Chi Yen and Daniel B. Larremore, "Blockmodeling a Bipartite Network with Bipartite Priors", in preparation.

        """
        self.bipartite_prior_ = bipartite_prior
        self._prerunning_checks()

        self._compute_dl_and_update(1, 1)
        if self.algm_name_ == "mcmc" and self._virgin_run:
            self._natural_merge()

        if self._check_if_local_minimum(self.bm_state["ka"], self.bm_state["kb"]):
            self.trace_k += [("mdl", self.bm_state["ka"], self.bm_state["kb"])]
            return self.bookkeeping_dl
        else:
            dS = 0.
            while abs(dS) < self.i_0 * self.bm_state["ref_dl"]:
                ka, kb = self.bm_state["ka"], self.bm_state["kb"]
                self.trace_k += [("merge_or_rollback", ka, kb)]
                if ka * kb != 1:
                    ka_, kb_, dS, mlist = self._merge_e_rs(ka, kb)
                    if self._determine_i_0(dS):
                        ka__, kb__, _ = self.summary(mode="simple")
                        self._logger.info(
                            f"Tried {(ka, kb)} ~~-> {(ka_, kb_)}, "
                            f"but *DL{(ka_, kb_)} deviates too much from *DL{(ka__, kb__)}, "
                            f"which is {dS}."
                        )
                        break
                    mb_ = accept_mb_merge(self.bm_state["mb"], mlist)
                    e_rs = assemble_e_rs_from_mb(self.edgelist, mb_)
                    self._update_bm_state(ka_, kb_, e_rs, mb_, record_merge=True)
                    self._logger.info(f"{(ka, kb)} ~~-> {(ka_, kb_)}")
                else:
                    break
            ka, kb = self.bm_state["ka"], self.bm_state["kb"]
            self.trace_k += [("escape_to", ka, kb)]
            self._logger.info(f"Escape the loop of agglomerative merges. Now {(ka, kb)} looks suspicious.")
            return self.minimize_bisbm_dl(bipartite_prior=self.bipartite_prior_)

    def summary(self, mode=None):
        """Return a summary of the algorithmic outcome.

        Returns
        -------
        OptimalKs._summary : ``dict``
            A summary of the algorithmic outcome with minimal description length (in nats).

        """
        ka, kb = sorted(self.bookkeeping_dl, key=self.bookkeeping_dl.get)[0]
        self._summary["mdl"] = self.bookkeeping_dl[(ka, kb)]
        if mode == "simple":
            return ka, kb, self._summary["mdl"]

        self._summary["ka"] = ka
        self._summary["kb"] = kb
        self._summary["dl"] = self.summary_dl(ka, kb)
        del self._summary["dl"]["dl"]

        return self._summary

    def summary_dl(self, ka, kb):
        _summary = dict()
        na, nb, e = self.bm_state["n_a"], self.bm_state["n_b"], self.bm_state["e"]
        try:
            mb = self.trace_mb[(ka, kb)][1]
        except KeyError:
            raise KeyError(f"Did you compute the partition at {(ka, kb)}?")
        nr = assemble_n_r_from_mb(mb)
        _summary["adjacency"] = float(adjacency_entropy(self.edgelist, mb))
        _summary["partition"] = float(partition_entropy(ka=ka, kb=kb, na=na, nb=nb, nr=nr))
        _summary["degree"] = float(degree_entropy(self.edgelist, mb, __q_cache=self.__q_cache))
        _summary["edges"] = float(
            model_entropy(e, ka=ka, kb=kb, na=na, nb=nb, nr=nr, is_bipartite=self.bipartite_prior_) -
            _summary["partition"])
        _summary["dl"] = sum(_summary.values())
        return _summary

    def compute_and_update(self, ka, kb, recompute=True):
        """Infer the partitions at a specific :math:`(K_a, K_b)` and then update the base class.

        Parameters
        ----------
        ka : ``int``
            Number of type-`a` communities that we want to partition.

        kb : ``int``
            Number of type-`b` communities that we want to partition.

        recompute : ``bool`` (optional, default: ``True``)

        """
        if recompute:
            self.bookkeeping_dl[(ka, kb)] = 0
        self._compute_dl_and_update(ka, kb, recompute=recompute)

    def compute_dl(self, ka, kb, recompute=False):
        """Execute the partitioning code by spawning child processes in the shell; saves its output afterwards.

        Parameters
        ----------
        ka : ``int``
            Number of type-`a` communities that we want to partition.

        kb : ``int``
            Number of type-`b` communities that we want to partition.

        recompute : ``bool`` (optional, default: ``False``)
            TODO.

        Returns
        -------
        dl : ``float``
            The description length of the partition found.

        e_rs : :class:`numpy.ndarray`
            the affinity matrix via the group membership vector found by the partitioning engine

        mb : ``list[int]``
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
                return self.bookkeeping_dl[(ka, kb)], self.bookkeeping_e_rs[(ka, kb)], self.trace_mb[(ka, kb)][1]
        na, nb, e = self.bm_state["n_a"], self.bm_state["n_b"], self.bm_state["e"]
        if ka == 1 and kb == 1:
            mb = np.array([0] * na + [1] * nb, dtype=np.int_)
            res = self._compute_desc_len(na, nb, e, ka, kb, mb)
            return res[0], res[1], res[2]

        if not recompute:
            ka_ = self._summary["algm_args"]["init_ka"]
            kb_ = self._summary["algm_args"]["init_kb"]
            na = self._summary["na"]
            nb = self._summary["nb"]
            dist = np.sqrt((ka_ - ka) ** 2 + (kb_ - kb) ** 2)
            if dist <= self._k_th_nb_to_search * np.sqrt(2):
                self._logger.info(f"({na}, {nb}) ~~-> ({ka}, {kb}); Use that partition to start MCMC@({ka}, {kb}).")
                _mb = None
            else:
                self._logger.info(f"({ka_}, {kb_}) ~~-> ({ka}, {kb}); Use that partition to start MCMC@({ka}, {kb}).")
                _mb = self.trace_mb[(ka_, kb_)][1]
        else:
            _mb = None

        def run(a, b):
            return self.engine_(self._f_edgelist_name, na, nb, a, b, mb=_mb)

        # Calculate the biSBM inference several times,
        # choose the maximum likelihood (or minimum entropy) result.
        results = []
        if self.is_par_:
            # automatically shutdown after idling for 600s
            self.__del__no_call = True
            results = list(loky_executor(self.n_cores_, 600, lambda x: run(ka, kb), list(range(self.max_n_sweeps_))))
            self.__del__no_call = False
        else:
            for _ in range(self.max_n_sweeps_):
                results += [run(ka, kb)]

        result_ = [self._compute_desc_len(na, nb, e, ka, kb, r) for r in results]
        result = min(result_, key=lambda x: x[0])
        dl = result[0]
        e_rs = result[1]
        mb = result[2]
        return dl, e_rs, mb

    def natural_merge(self):
        """Phase 1 natural e_rs-block merge"""
        na, nb, e = self.bm_state["n_a"], self.bm_state["n_b"], self.bm_state["e"]

        def run(_):
            # Note: setting (ka, kb) = (1, 1) is redundant.
            return self.engine_(self._f_edgelist_name, na, nb, 1, 1, mb=None, method="natural")

        results = []
        if self.is_par_:
            # automatically shutdown after idling for 600s
            self.__del__no_call = True
            results = list(loky_executor(self.n_cores_, 600, lambda x: run(0), list(range(self.max_n_sweeps_))))
            self.__del__no_call = False
        else:
            for _ in range(self.max_n_sweeps_):
                results += [run(0)]

        result_ = [self._compute_desc_len(na, nb, e, r[0], r[1], r[2:]) for r in results]
        result = min(result_, key=lambda x: x[0])
        dl = result[0]
        e_rs = result[1]
        mb = result[2]
        ka, kb = result[3]
        return dl, e_rs, mb, ka, kb, na, nb

    def _natural_merge(self):
        dl, e_rs, mb, ka, kb, na, nb = self.natural_merge()
        assert max(mb) + 1 == ka + kb, "[ERROR] inconsistency between mb. indexes and #blocks. {} != {}".format(
            max(mb) + 1, ka + kb)
        self._summary["algm_args"]["init_ka"] = ka
        self._summary["algm_args"]["init_kb"] = kb
        self._logger.info(f"Natural agglomerative merge {(na, nb)} ~~-> {(ka, kb)}.")
        self.bookkeeping_dl[(ka, kb)] = dl
        self.bookkeeping_e_rs[(ka, kb)] = e_rs
        self.trace_mb[(ka, kb)] = ("merge", mb)
        self._update_bm_state(ka, kb, e_rs, mb)
        self._virgin_run = False

    def _determine_i_0(self, dS):
        if self.i_0 < 1:
            return False
        i_0 = dS / self.bm_state["ref_dl"]
        self.i_0s += [i_0]
        iqr = np.percentile(self.i_0s, 75) - np.percentile(self.i_0s, 25)
        if i_0 > 3 * iqr + np.percentile(self.i_0s, 75) >= 1e-4:
            self.i_0 = i_0
            self._summary["algm_args"]["i_0"] = i_0
            self._logger.info(f"Determining \u0394 at {i_0}.")  # \u0394 = Delta = i_0
            return True
        else:
            return False

    def _compute_desc_len(self, n_a, n_b, e, ka, kb, mb):
        e_rs = assemble_e_rs_from_mb(self.edgelist, mb)
        nr = assemble_n_r_from_mb(mb)
        desc_len = get_desc_len_from_data(n_a, n_b, e, ka, kb, self.edgelist, mb, nr=nr, q_cache=self.__q_cache,
                                          is_bipartite=self.bipartite_prior_)
        return desc_len, e_rs, mb, (ka, kb)

    def _merge_e_rs(self, ka, kb):
        """Apply multiple merges of the original affinity matrix, return the one that least alters the entropy

        Parameters
        ----------
        ka : ``int``
            Number of type-a communities in the affinity matrix
        kb : ``int``
            Number of type-b communities in the affinity matrix

        Returns
        -------
        _ka : ``int``
            New number of type-a communities in the affinity matrix

        _kb : ``int``
            New number of type-b communities in the affinity matrix

        dS : ``list(int, int)``
            Difference of the new entropy and the old one

        _mlist : ``list(int, int)``
            The two row-indexes of the original affinity matrix that were finally chosen (and merged)

        """
        m = np.arange(ka + kb)

        mlists = set()
        while len(mlists) == 0:
            for _m in m:
                pool = random.choices(m, k=self._nm)
                _mlist = [[min(x, _m), max(x, _m)] for x in pool]
                for _ in _mlist:
                    cond = (_[0] != _[1]) and not (_[1] >= ka > _[0]) and not (_[0] == 0 and ka == 1) and not (
                            _[0] == ka and kb == 1)
                    if cond:
                        mlists.add(str(_[0]) + "+" + str(_[1]))

        dS, _mlist = virtual_moves_ds(self.bm_state["e_rs"], mlists, self.bm_state["ka"])
        if np.max(_mlist) < self.bm_state["ka"]:
            ka = self.bm_state["ka"] - 1
            kb = self.bm_state["kb"]
        else:
            ka = self.bm_state["ka"]
            kb = self.bm_state["kb"] - 1
        return ka, kb, dS, _mlist

    def _rollback(self):
        ka, kb, dl = self.summary(mode="simple")
        e_rs = self.bookkeeping_e_rs[(ka, kb)]
        mb = self.trace_mb[(ka, kb)][1]
        self._update_bm_state(ka, kb, e_rs, mb)
        return ka, kb, e_rs, dl

    def _update_bm_state(self, ka, kb, e_rs, mb, record_merge=False):
        self.bm_state["ka"], self.bm_state["kb"] = ka, kb
        self.bm_state["e_rs"] = e_rs
        self.bm_state["mb"] = mb
        if record_merge:
            self.trace_mb[(ka, kb)] = ("merge", mb)

    def _compute_dl_and_update(self, ka, kb, recompute=False):
        dl, e_rs, mb = self.compute_dl(ka, kb, recompute=recompute)
        assert max(mb) + 1 == ka + kb, "[ERROR] inconsistency between mb. indexes and #blocks. {} != {}".format(
            max(mb) + 1, ka + kb)
        self.bookkeeping_dl[(ka, kb)] = dl
        self.bookkeeping_e_rs[(ka, kb)] = e_rs
        self.trace_mb[(ka, kb)] = ("mcmc", mb)
        self.trace_k += [("mcmc", ka, kb)]
        self.bm_state["ref_dl"] = self.summary(mode="simple")[2] if self.bm_state["ref_dl"] != 0 else dl
        return dl, e_rs, mb

    # ###########
    # Checkpoints
    # ###########
    def _is_mdl_so_far(self, desc_len):
        """Check if `desc_len` is the minimal value so far."""
        return not any([i < desc_len for i in self.bookkeeping_dl.values()])

    def _check_if_local_minimum(self, ka, kb):
        """The `neighborhood search` as described in the paper."""
        self._logger.info(f"Is {(ka, kb)} a local minimum? Let's check.")
        _dl, _e_rs, _mb = self._compute_dl_and_update(ka, kb)
        null_dl = self.bookkeeping_dl[(1, 1)]
        if _dl > self.bookkeeping_dl[(1, 1)]:
            self._logger.info("DL({}, {}) > DL(1, 1), which is {} compared to {}".format(ka, kb, _dl, null_dl))
            self._logger.info(f"~~~- Keep merging -~~~")
            self._update_bm_state(ka, kb, _e_rs, _mb)
            return False

        if _dl > self.summary(mode="simple")[2]:
            ka, kb, _, _dl = self._rollback()
            _ = self.adaptive_ratio
            self.i_0 *= _
            self._logger.info(f"Overshooting! There's already a point with lower DL. Let's reduce \u0394 by {_}.")
            self._logger.info(f"Move to {(ka, kb)} and re-checking if it is a local minimum.")

        nb_points = self._get_neighbor_points(ka, kb)

        for _ka, _kb in nb_points:
            self._compute_dl_and_update(_ka, _kb)
            if not self._is_mdl_so_far(_dl):
                _, _, _, mdl = self._rollback()
                self._logger.info(
                    f"Warning. DL{(_ka, _kb)} = {mdl} < DL{(ka, kb)}. We move to {(_ka, _kb)} but NOT reduce \u0394.")
                break

        if _dl != self.summary(mode="simple")[2]:
            self._logger.info(f"Bummer. {(ka, kb)} is NOT a local minimum.")
            return False
        else:
            self._logger.info(f"YES! {(ka, kb)} is a local minimum with DL = {_dl}. We are done.")
            return True

    def _get_neighbor_points(self, ka, kb):
        k_th = self._k_th_nb_to_search
        nb_points = [(x + ka, y + kb) for (x, y) in product(range(-k_th, k_th + 1), repeat=2)]
        # if any item has values less than 1, delete it. Also, exclude the suspected point (i.e., [ka, kb]).
        na = self.bm_state["n_a"]
        nb = self.bm_state["n_b"]
        nb_points = [(i, j) for i, j in nb_points if na >= i >= 1 and nb >= j >= 1 and (i, j) != (ka, kb)]
        _ = sorted(nb_points, key=lambda x: x[0] - ka + x[1] - kb, reverse=True)
        nb_points = [_.pop(0), _.pop(-1)]
        random.shuffle(_)
        nb_points += _
        return nb_points

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
    def set_params(self, init_ka=10, init_kb=10, i_0=0.005):
        """Set the parameters for the heuristic.

        Parameters
        ----------
        init_ka : ``int`` (required, default: ``10``)

        init_kb : ``int`` (required, default: ``10``)

        i_0 : ``float`` (required, default: ``0.005``)

        Notes
        -----
        TODO.

        .. warning::

           If :math:`i_0` is set too small, the heuristic will be slow and tends to get trapped in
           a local minimum (in the description length landscape) where :math:`K_a` and :math:`K_b` are large.

        """
        # params for the heuristic
        self.bm_state["ka"] = int(init_ka)
        self.bm_state["kb"] = int(init_kb)
        self.i_0 = float(i_0)
        assert 0. <= self.i_0 < 1, "[ERROR] Allowed range for i_0 is [0, 1)."
        assert self.bm_state["ka"] <= self.bm_state[
            "n_a"], "[ERROR] Number of type-a communities must be smaller than the # nodes in type-a."
        assert self.bm_state["kb"] <= self.bm_state[
            "n_b"], "[ERROR] Number of type-b communities must be smaller than the # nodes in type-b."
        self._summary["algm_args"]["init_ka"] = self.bm_state["ka"]
        self._summary["algm_args"]["init_kb"] = self.bm_state["kb"]
        self._summary["algm_args"]["i_0"] = float(i_0)

    def set_adaptive_ratio(self, adaptive_ratio=0.95):
        """Set the adaptive ratio (``float`` between 0 to 1, defaults to ``0.95``)."""
        assert 0. < adaptive_ratio < 1, "[ERROR] Allowed range for adaptive_ratio is (0, 1)."
        self.adaptive_ratio = float(adaptive_ratio)

    def set_k_th_neighbor_to_search(self, k):
        self._k_th_nb_to_search = int(k)

    def set_nm(self, s=10):
        """Set the :math:`n_m` parameter (defaults to ``10``)."""
        self._nm = int(s)

    def get_f_edgelist_name(self):
        return self._f_edgelist_name

    def get__q_cache(self):
        return self.__q_cache

    def _set_logging_level(self, level):
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

    def _get_tempfile_edgelist(self):
        try:
            self.f_edgelist.seek(0)
        except AttributeError:
            if self.is_par_:
                delete = False
            else:
                delete = True
            self.f_edgelist = tempfile.NamedTemporaryFile(mode='wb', dir=self.tempdir, delete=delete)
        finally:
            for edge in self.edgelist:
                content = str(edge[0]) + "\t" + str(edge[1]) + "\n"
                self.f_edgelist.write(content.encode())
            self.f_edgelist.flush()
            f_edgelist_name = self.f_edgelist.name
        if self.is_par_:
            del self.f_edgelist
        return f_edgelist_name

    def __del__(self):
        if self.__del__no_call:
            return
        if self.is_par_:
            os.remove(self._f_edgelist_name)
