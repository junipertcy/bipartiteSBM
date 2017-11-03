from det_k_bisbm.ioutils import *
from det_k_bisbm.optimalks import *

from engines.mcmc import *


mcmc = MCMC(f_engine="engines/bipartiteSBM-MCMC/bin/mcmc",
            n_sweeps=1,
            is_parallel=True,
            n_cores=1,
            mcmc_steps=10000000,
            mcmc_await_steps=1000000,
            mcmc_cooling="constant",
            mcmc_cooling_param_1=0.1,
            mcmc_cooling_param_2=0.1,
            mcmc_epsilon=10.
        )

edgelist = get_edgelist("dataset/bisbm-n_1000-ka_4-kb_6-r-1.0-Ka_30-Ir_1.75.gt.edgelist", "\t")
types = mcmc.gen_types(500, 500)

oks = OptimalKs(mcmc, edgelist, types)

oks.set_params(init_ka=50, init_kb=50, i_th=0.1)
oks.set_k_th_neighbor_to_search(2)


def test_answer():
    confident_desc_len = oks.iterator()
    p_estimate = sorted(confident_desc_len, key=confident_desc_len.get)[0]
    # YES. We may not obtain (4, 6), as non-identifiable blocks may exist.
    assert p_estimate in [(4, 6), (4, 5), (4, 7)]
