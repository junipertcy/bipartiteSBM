import pytest
import sys

from det_k_bisbm.ioutils import *
from det_k_bisbm.optimalks import *

from engines.mcmc import *

mcmc = MCMC(f_engine="engines/bipartiteSBM-MCMC/bin/mcmc",
            n_sweeps=10,
            is_parallel=True,
            n_cores=2,
            mcmc_steps=100000,
            mcmc_await_steps=10000,
            mcmc_cooling="exponential",
            mcmc_cooling_param_1=10,
            mcmc_cooling_param_2=0.1,
            mcmc_epsilon=0.01
        )

edgelist = get_edgelist("dataset/southernWomen.edgelist", "\t")
types = get_types("dataset/southernWomen.types")

oks = OptimalKs(mcmc, edgelist, types)

oks.set_params(init_ka=10, init_kb=10, i_th=0.1)


@pytest.mark.skipif(sys.platform == 'linux', reason="TODO: not gonna work in linux")
@pytest.fixture
def confident_desc_len():
    return oks.iterator()


@pytest.mark.skipif(sys.platform == 'linux', reason="TODO: not gonna work in linux")
def test_answer(confident_desc_len):
    p_estimate = sorted(confident_desc_len, key=confident_desc_len.get)[0]
    assert p_estimate == (1, 1)
