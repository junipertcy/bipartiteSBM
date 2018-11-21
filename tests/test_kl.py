from det_k_bisbm.ioutils import *
from det_k_bisbm.optimalks import *

from engines.kl import *


kl = KL(f_engine="engines/bipartiteSBM-KL/biSBM",
        n_sweeps=1,
        is_parallel=True,
        n_cores=1,
        kl_edgelist_delimiter="\t",
        kl_steps=5,
        kl_itertimes=1,
        f_kl_output="engines/bipartiteSBM-KL/f_kl_output"
    )

edgelist = get_edgelist("dataset/test/southernWomen.edgelist", "\t")
types = get_types("dataset/test/southernWomen.types")

oks = OptimalKs(kl, edgelist, types, default_args=True, random_init_k=False)


def test_answer():
    confident_desc_len = oks.iterator()
    p_estimate = sorted(confident_desc_len, key=confident_desc_len.get)[0]
    assert p_estimate == (1, 1)
