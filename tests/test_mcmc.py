import pytest
import biSBM as bm


mcmc = bm.engines.MCMC(
    f_engine="engines/bipartiteSBM-MCMC/bin/mcmc",
    n_sweeps=1,
    is_parallel=False,
    n_cores=1
)

edgelist = bm.get_edgelist("dataset/test/bisbm-n_1000-ka_4-kb_6.edgelist")
types = mcmc.gen_types(500, 500)

oks = bm.OptimalKs(mcmc, edgelist, types, default_args=True, random_init_k=False)


def test_answer():
    oks.minimize_bisbm_dl()
    ka = oks.summary()["ka"]
    kb = oks.summary()["kb"]
    # Note that we may not obtain (4, 6), as non-identifiable blocks may exist.
    assert (ka, kb) in [(4, 6)]


def test_summary_dl_at_1_1():
    oks.compute_and_update(1, 1)
    dl = oks.summary_dl(1, 1)
    assert dl["dl"] == 56078.5634561319
    assert dl["adjacency"] == 51884.81583464478
    assert dl["partition"] == 12.429216196844383
    assert dl["degree"] == 4181.318405290275
    assert dl["edges"] == 0.0


def test_summary_dl_at_500_500():
    oks.compute_and_update(500, 500)
    dl = oks.summary_dl(500, 500)
    assert dl["dl"] == pytest.approx(62871.75555606898)
    assert dl["adjacency"] == pytest.approx(0., abs=1e-9)
    assert dl["partition"] == pytest.approx(5235.090133117156)
    assert dl["degree"] == pytest.approx(0., abs=1e-9)
    assert dl["edges"] == pytest.approx(57636.66542295214)


def test_issue_12():
    edgelist = [[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [1, 5], [2, 6], [2, 7], [2, 8]]
    types = [1, 1, 1, 2, 2, 2, 2, 2, 2]
    oks = bm.OptimalKs(mcmc, edgelist, types)
    oks.minimize_bisbm_dl()
    dl = oks.summary()
    assert dl["mdl"] == pytest.approx(15.615238196841506)
