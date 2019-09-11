bipartiteSBM
===========

.. image:: https://img.shields.io/badge/contact-@oneofyen-blue.svg?style=flat
   :target: https://twitter.com/oneofyen
   :alt: Twitter: @oneofyen
.. image:: https://img.shields.io/badge/license-GPL-green.svg?style=flat
   :target: https://github.com/junipertcy/bipartiteSBM/blob/master/LICENSE
   :alt: License
.. image:: https://travis-ci.org/junipertcy/bipartiteSBM.svg?branch=master
   :target: https://travis-ci.org/junipertcy/bipartiteSBM
   :alt: Build Status

The ``bipartiteSBM`` is a Python library of a fast community inference heuristic of the bipartite Stochastic Block Model (biSBM),
using the `MCMC sampler`_ or the `Kernighan-Lin algorithm`_. It estimates the number of communities (as well as the partition) for a bipartite network.

.. figure::  https://wiki.junipertcy.info/images/1/10/Det_k_bisbm-logo.png
   :align:   center

   The ``bipartiteSBM`` helps you infer the number of communities in a bipartite network.

The ``bipartiteSBM`` utilizes the Minimum Description Length principle to determine a point estimate of the
numbers of communities in the biSBM that best compress the model and data.
Several test examples are included.

Supported and tested on Python>=3.6.

Documentation
-------------
The project documentation is at https://docs.netscied.tw/bipartiteSBM/index.html.

Companion article
-----------------
If you find ``bipartiteSBM`` useful for your research, please consider citing the following paper:

Tzu-Chi Yen and Daniel B. Larremore, "Blockmodeling a Bipartite Network with Bipartite Prior," `in preparation`.


.. _`MCMC sampler`: https://github.com/junipertcy/bipartiteSBM-MCMC
.. _`Kernighan-Lin algorithm`: https://github.com/junipertcy/bipartiteSBM-KL
