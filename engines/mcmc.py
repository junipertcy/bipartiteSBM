import os
import numpy as np
import subprocess


class MCMC(object):
    """Base class for the Markov chain Monte Carlo algorithm.

    Parameters
    ----------
    f_engine : ``str`` (required, default: `"engines/bipartiteSBM-MCMC/bin/mcmc"`)
        Path to the graph partitioning binary.

    n_sweeps : ``int`` (required, default: `1`)
        Number of partitioning computations for each :math:`(K_a, K_b)` data point.

    is_parallel : ``bool`` (required, default: `False`)
        Whether to compute the partitioning in parallel.

    n_cores : ``int`` (required, default: `1`)
        The number of cores used when `is_parallel is True`.

    algm_name : ``str`` (required, default: `mcmc`)
        The name of the algorithm.

    mcmc_steps : ``int`` (required, default: `1e5`)
        Number of sweeps to perform. During each sweep, a move attempt is made for each node.

    mcmc_await_steps : ``int`` (required, default: `2e3`)
        Number of iterations to wait for a record-breaking event. The algorithm will stop if there is no record-breaking
        event within the interval or the overall MCMC sweeps exceed ``mcmc_steps``, whichever happens earlier.

    mcmc_cooling : ``str`` (required, default: `abrupt_cool`)
        Annealing scheme used, which can be either ``exponential``, ``logarithm``, ``linear``, ``constant``,
        or ``abrupt_cool``.

    mcmc_cooling_param_1 : ``int`` (required, default: `1e3`)
        Parameter 1 for the annealing.

    mcmc_cooling_param_2 : ``float`` (required, default: `0.1`)
        Parameter 2 for the annealing.

    mcmc_epsilon : ``float`` (required, default: `1.`)
        The :math:`\epsilon` parameter used in the proposal moves.

    """
    def __init__(self,
                 f_engine="engines/bipartiteSBM-MCMC/bin/mcmc",
                 n_sweeps=1,
                 is_parallel=False,
                 n_cores=1,
                 algm_name="mcmc",
                 mcmc_steps=1e5,
                 mcmc_await_steps=2e3,
                 mcmc_cooling="abrupt_cool",
                 mcmc_cooling_param_1=1e3,
                 mcmc_cooling_param_2=0.1,
                 mcmc_epsilon=1.):

        self.MAX_NUM_SWEEPS = int(n_sweeps)
        self.PARALLELIZATION = bool(is_parallel)
        self.NUM_CORES = int(n_cores)
        self.ALGM_NAME = str(algm_name)

        # for MCMC
        if not os.path.isfile(f_engine):
            raise BaseException("[ERROR] MCMC engine binary not found!")

        self.f_engine = f_engine
        self.mcmc_steps_ = int(mcmc_steps)
        self.mcmc_await_steps_ = int(mcmc_await_steps)
        self.mcmc_cooling_ = str(mcmc_cooling)
        self.mcmc_cooling_param_1 = mcmc_cooling_param_1
        self.mcmc_cooling_param_2 = mcmc_cooling_param_2
        self.mcmc_epsilon_ = mcmc_epsilon

        pass

    def set_steps(self, steps):
        self.mcmc_steps_ = int(steps)

    def set_await_steps(self, await_steps):
        self.mcmc_await_steps_ = int(await_steps)

    def set_cooling(self, cooling):
        self.mcmc_cooling_ = str(cooling)

    def set_cooling_param_1(self, cooling_param_1):
        self.mcmc_cooling_param_1 = cooling_param_1

    def set_cooling_param_2(self, cooling_param_2):
        self.mcmc_cooling_param_2 = cooling_param_2

    def set_epsilon(self, epsilon):
        self.mcmc_epsilon_ = epsilon

    def prepare_engine(self, f_edgelist, na, nb, ka, kb, mb=None, method=None):
        """Output shell commands for graph partitioning calculation.

        Parameters
        ----------
        ka : ``int`` (required)
            Number of communities for type-`a` nodes to partition.

        kb : ``int`` (required)
            Number of communities for type-`b` nodes to partition.

        Returns
        -------
        action_str : ``str``
            the command line string that enables execution of the code

        """
        params_ = ""
        if self.mcmc_cooling_ in ["exponential", "linear", "logarithmic"]:
            params_ = str(self.mcmc_cooling_param_1) + " " + str(self.mcmc_cooling_param_2)
        elif self.mcmc_cooling_ in ["constant", "abrupt_cool"]:
            params_ = str(self.mcmc_cooling_param_1)

        if mb is None:
            means_ = "-g"
        else:
            means_ = "--mb" + " " + " ".join(map(str, mb))

        if method == "natural":
            means_ = "-g -u"

        # n_blocks_ = " ".join(
        #     self._constrained_sum_sample_pos(ka, na)
        # ) + " " + " ".join(
        #     self._constrained_sum_sample_pos(kb, nb)
        # )
        n_blocks_ = self._gen_init_n_blocks(na, nb, ka, kb)
        n_types_ = str(na) + " " + str(nb)

        action_list = [
            self.f_engine,
            "-e",
            f_edgelist,
            "-n",
            n_blocks_,
            "-t",
            str(self.mcmc_steps_),
            "-x",
            str(self.mcmc_await_steps_),
            "-c",
            self.mcmc_cooling_,
            "-a",
            params_,
            "-y",
            n_types_,
            "-z",
            str(ka) + " " + str(kb),
            "-E",
            str(self.mcmc_epsilon_),
            means_
        ]

        action_str = ' '.join(action_list)
        #print(action_str)
        return action_str

    def engine(self, f_edgelist, na, nb, ka, kb, mb=None, method=None):  # TODO: bug when assigned verbose=False
        """Run the shell code.

        Parameters
        ----------
        f_edgelist : ``str``

        na : ``int``

        nb : ``int``

        ka : ``int``, required
            Number of communities for type-*a* nodes to partition.

        kb : ``int``, required
            Number of communities for type-*b* nodes to partition.

        mb : :class:`numpy.ndarray`

        method :

        Returns
        -------
        of_group : :class:`numpy.ndarray`

        """
        of_group = []
        action_str = self.prepare_engine(f_edgelist, na, nb, ka, kb, mb=mb, method=method)

        num_sweeps_ = 1

        def _run_engine(_):
            p = subprocess.Popen(
                action_str.split(' '),
                bufsize=2048,
                stdout=subprocess.PIPE
            )
            out, err = p.communicate()
            p.wait()
            return out, err, p

        num_sweep_ = 0

        while num_sweep_ < num_sweeps_:
            out, err, p = _run_engine("")
            if p.returncode == -11:  # when Exception raises from the mcmc code
                raise RuntimeError("Exception from C++ program during inference! -- " + action_str)
            elif p.returncode == 0:
                num_sweep_ += 1
                of_group = out.replace(b' \n', b'').split(b' ')  # Note the space before the line break
                of_group = list(map(int, of_group))

        return np.array(of_group, dtype=np.int_)

    @staticmethod
    def _gen_init_n_blocks(na, nb, ka, kb):
        num_nodes_a = np.arange(na)
        n_blocks_a = map(len, np.array_split(num_nodes_a, ka))
        num_nodes_b = np.arange(nb)
        n_blocks_b = map(len, np.array_split(num_nodes_b, kb))

        n_blocks_ = " ".join(map(str, n_blocks_a)) + " " + " ".join(map(str, n_blocks_b))

        return n_blocks_

    @staticmethod
    def _constrained_sum_sample_pos(n, total):
        # in this setting, there will be no empty groups generated by this function
        n = int(n)
        total = int(total)
        normalized_list = [int(total) + 1]
        while sum(normalized_list) > total and np.greater_equal(normalized_list, np.zeros(n)).all():
            indicator = True
            while indicator:
                normalized_list = list(map(round, map(lambda x: x * total, np.random.dirichlet(np.ones(n), 1).tolist()[0])))
                normalized_list = list(map(int, normalized_list))
                indicator = len(normalized_list) - np.count_nonzero(normalized_list) != 0
            sum_ = 0
            for ind, q in enumerate(normalized_list):
                if ind < len(normalized_list) - 1:
                    sum_ += q
            # TODO: there is a bug here; sometimes it assigns -1 to the end of the array, but pass the while condition
            normalized_list[len(normalized_list) - 1] = abs(total - sum_)
        assert sum(normalized_list) == total, "ERROR: the constrainedSumSamplePos-sampled list does not sum to #edges."
        return map(str, normalized_list)

    @staticmethod
    def gen_types(na, nb):
        types = [1] * int(na) + [2] * int(nb)
        return types
