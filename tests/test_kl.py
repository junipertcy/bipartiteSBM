import biSBM as bm


kl = bm.engines.KL(
    f_engine="engines/bipartiteSBM-KL/biSBM",
    n_sweeps=1,
    is_parallel=True,
    n_cores=1,
    kl_edgelist_delimiter="\t",
    kl_steps=5,
    kl_itertimes=1,
    f_kl_output="engines/bipartiteSBM-KL/f_kl_output"
)

edgelist = bm.get_edgelist("dataset/test/southernWomen.edgelist", "\t")
types = bm.get_types("dataset/test/southernWomen.types")

oks = bm.OptimalKs(kl, edgelist, types, default_args=True, random_init_k=False)


def test_answer():
    oks.minimize_bisbm_dl()
    ka = oks.summary()["ka"]
    kb = oks.summary()["kb"]
    assert (ka, kb) == (1, 1)  # there exists no community structure in the southernWomen dataset
