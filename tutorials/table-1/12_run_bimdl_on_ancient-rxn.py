# 12_run_bimdl_on_ancient-rxn.py

import os
import sys
module_path = os.path.abspath(os.path.join('../../'))
if module_path not in sys.path:
    sys.path.append(module_path)


from engines.mcmc import *
from det_k_bisbm.optimalks import *
from det_k_bisbm.ioutils import *


filename = "../../dataset/empirical/ancient-rxn_5651-metabolites_5252.edgelist"
na = 5651
nb = 5252
avg_deg = 4.2

n = na + nb

k = int( (n * avg_deg / 2) ** 0.5)


mcmc = MCMC(f_engine="../../engines/bipartiteSBM-MCMC/bin/mcmc",
            n_sweeps=10,
            is_parallel=True,
            n_cores=10,
            mcmc_steps=10000*n,
            mcmc_await_steps=1000*n,
            mcmc_cooling="abrupt_cool",
            mcmc_epsilon=0.01
)


edgelist = get_edgelist(filename, " ")
types= mcmc.gen_types(na, nb)
oks = OptimalKs(mcmc, edgelist, types)

oks.set_params(init_ka=k, init_kb=k, i_th=0.1)
oks.set_adaptive_ratio(0.9)
oks.set_exist_bookkeeping(True)
oks.set_logging_level("info")
oks.set_k_th_neighbor_to_search(1)

oks.iterator()

