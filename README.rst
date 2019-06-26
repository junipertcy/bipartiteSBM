det_k_bisbm
===========

.. image:: https://img.shields.io/badge/contact-@oneofyen-blue.svg?style=flat
   :target: https://twitter.com/oneofyen
   :alt: Twitter: @oneofyen
.. image:: https://img.shields.io/badge/license-GPL-green.svg?style=flat
   :target: https://github.com/junipertcy/det_k_bisbm/blob/master/LICENSE
   :alt: License
.. image:: https://travis-ci.org/junipertcy/det_k_bisbm.svg?branch=master
   :target: https://travis-ci.org/junipertcy/det_k_bisbm
   :alt: Build Status

The ``det_k_bisbm`` is a Python library of a fast community inference heuristic of the bipartite Stochastic Block Model (biSBM),
using the `MCMC sampler`_ or the `Kernighan-Lin algorithm`_. It estimates the number of communities (as well as the partition) for a bipartite network.

.. figure::  https://wiki.junipertcy.info/images/1/10/Det_k_bisbm-logo.png
   :align:   center

   The ``det_k_bisbm`` helps you infer the number of communities in a bipartite network.

The ``det_k_bisbm`` utilizes the Minimum Description Length principle to determine a point estimate of the
numbers of communities in the biSBM that best compress the model and data.
Several test examples are included.

Supported and tested on Python>=3.6.

Documentation
-------------
The project documentation is at https://docs.netscied.tw/det_k_bisbm/index.html.

Companion article
-----------------
If you find ``det_k_bisbm`` useful for your research, please consider citing the following paper:

Tzu-Chi Yen and Daniel B. Larremore, "Blockmodeling on a Bipartite Network with Bipartite Prior," `in preparation`.


.. _`MCMC sampler`: https://github.com/junipertcy/bipartiteSBM-MCMC
.. _`Kernighan-Lin algorithm`: https://github.com/junipertcy/bipartiteSBM-KL
