from biSBM.ioutils import *
from biSBM.optimalks import *

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
    oks.minimize_bisbm_dl()
    ka = oks.summary()["ka"]
    kb = oks.summary()["kb"]
    assert (ka, kb) == (2, 3)  # there exists community structure in the southernWomen dataset
