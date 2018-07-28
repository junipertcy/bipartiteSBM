# 05_run_bimdl_on_malaria.py

import os
import sys
module_path = os.path.abspath(os.path.join('../../'))
if module_path not in sys.path:
    sys.path.append(module_path)


from engines.mcmc import *
from det_k_bisbm.optimalks import *
from det_k_bisbm.ioutils import *


filename = "../../dataset/empirical/malaria-gene_297-substring_806.edgelist"
na = 297
nb = 806
avg_deg = 5.4

n = na + nb

k = int( (n * avg_deg / 2) ** 0.5)


mcmc = MCMC(f_engine="../../engines/bipartiteSBM-MCMC/bin/mcmc",
            n_sweeps=10,
            is_parallel=True,
            n_cores=2,
            mcmc_steps=10000*n,
            mcmc_await_steps=1000*n,
            mcmc_cooling="abrupt_cool",
            mcmc_cooling_param_1=1000*n,
            mcmc_epsilon=0.0001
)


edgelist = get_edgelist(filename, "\t")
types= mcmc.gen_types(na, nb)
oks = OptimalKs(mcmc, edgelist, types)

oks.set_params(init_ka=k, init_kb=k, i_th=0.3)
oks.set_adaptive_ratio(0.9)
oks.set_exist_bookkeeping(True)
oks.set_logging_level("info")
oks.set_k_th_neighbor_to_search(1)

oks.iterator()

mdl_pair = sorted(oks.confident_desc_len, key=oks.confident_desc_len.get)[0]
print("MDL at ({}, {}): {}".format(mdl_pair[0], mdl_pair[1], oks.confident_desc_len[mdl_pair]))

