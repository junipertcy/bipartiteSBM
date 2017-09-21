from det_k_bisbm.ioutils import *
from det_k_bisbm.optimalks import *

from engines.kl import *


kl = KL(f_engine="engines/bipartiteSBM-KL/biSBM",
        n_sweeps=1,
        is_parallel=True,
        n_cores=1,
        kl_edgelist_delimiter="\t",
        kl_itertimes=4,
        f_kl_output="engines/bipartiteSBM-KL/f_kl_output"
    )

edgelist = get_edgelist("dataset/bisbm-n_1000-ka_4-kb_6-r-1.0-Ka_30-Ir_1.75.gt.edgelist", "\t")
types = kl.gen_types(500, 500)

oks = OptimalKs(kl, edgelist, types)
oks.set_params(init_ka=10, init_kb=10, i_th=0.1)


def test_answer():
    confident_desc_len = oks.iterator()
    p_estimate = sorted(confident_desc_len, key=confident_desc_len.get)[0]
    assert p_estimate == (4, 6)
