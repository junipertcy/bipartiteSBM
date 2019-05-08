# det_k_bisbm
[![Twitter: @oneofyen](https://img.shields.io/badge/contact-@oneofyen-blue.svg?style=flat)](https://twitter.com/oneofyen) 
[![License](https://img.shields.io/badge/license-GPL-green.svg?style=flat)](https://github.com/junipertcy/det_k_bisbm/blob/master/LICENSE)
[![Build Status](https://travis-ci.org/junipertcy/det_k_bisbm.svg?branch=master)](https://travis-ci.org/junipertcy/det_k_bisbm)

Python implementation of a fast community number determination heuristic of the bipartite Stochastic Block Model (biSBM), 
using the [MCMC sampler](https://github.com/junipertcy/bipartiteSBM-MCMC)
or the [Kernighan-Lin algorithm](https://github.com/junipertcy/bipartiteSBM-KL) as the inference engine. 

<p align="center">
  <a href="https://github.com/junipertcy/det_k_bisbm">
    <img width="800" src="http://wiki.junipertcy.info/images/1/10/Det_k_bisbm-logo.png">
  </a>
</p>

---

`det_k_bisbm` utilizes the Minimum Description Length principle to determine a point estimate of the
numbers of communities in the biSBM that best compresses the model and data. Several test examples are included.

If you want to sample the whole marginal distribution of the number of communities, rather than a point estimate,
please check the companion [Markov Chain Monte Carlo](https://github.com/junipertcy/bipartiteSBM-MCMC) program.

Supported and tested on Python>=3.6.

## Documentation
The project documentation is at https://docs.netscied.tw/det_k_bisbm/index.html.

## Companion article
If you find `det_k_bisbm` useful for your research, please consider citing the following paper:

Tzu-Chi Yen and Daniel B. Larremore, "Blockmodeling on a Bipartite Network with Bipartite Prior," *in preparation*.
