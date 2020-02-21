bipartiteSBM
============

.. image:: https://img.shields.io/badge/contact-@oneofyen-blue.svg?style=flat
   :target: https://twitter.com/oneofyen
   :alt: Twitter: @oneofyen
.. image:: https://img.shields.io/badge/license-GPL-green.svg?style=flat
   :target: https://github.com/junipertcy/bipartiteSBM/blob/master/LICENSE
   :alt: License
.. image:: https://travis-ci.org/junipertcy/bipartiteSBM.svg?branch=master
   :target: https://travis-ci.org/junipertcy/bipartiteSBM
   :alt: Build Status

This code and data repository accompanies the paper

* *Community Detection in Bipartite Networks with Stochastic Blockmodels*, `Tzu-Chi Yen`_ and `Daniel B. Larremore`_, arXiv: 2001.11818 (2020).

Read it on: [`arXiv`_].

The code is tested on Python>=3.6. For questions, please email tzuchi at tzuchi.yen@colorado.edu, or via the `issues`_!

Introduction
------------

The ``bipartiteSBM`` implements a fast community inference algorithm for the bipartite Stochastic Block Model (biSBM)
using the `MCMC sampler`_ or the `Kernighan-Lin algorithm`_ as the core optimization engine.
It searches through the space with dynamic programming, and estimates the number of communities
(as well as the partition) for a bipartite network.

.. figure::  https://wiki.junipertcy.info/images/1/10/Det_k_bisbm-logo.png
   :align:   center

   The ``bipartiteSBM`` helps you infer the number of communities in a bipartite network. (``det_k_bisbm`` is a deprecated name for the same library.)

The ``bipartiteSBM`` utilizes the Minimum Description Length principle to determine a point estimate of the
bipartite partition that best compresses the model and data. In other words, we formulate priors and maximize the
corresponding posterior likelihood function.

Several test examples are included. Read on in the `docs`_!

Documentation
-------------
* The project documentation is at https://docs.netscied.tw/bipartiteSBM/index.html.
* For installation instructions, see https://docs.netscied.tw/bipartiteSBM/usage/installation.html. You'll need `CMake`_ and `Boost`_ libraries, and a compiler that supports C++14.

.. _`MCMC sampler`: https://github.com/junipertcy/bipartiteSBM-MCMC
.. _`Kernighan-Lin algorithm`: https://github.com/junipertcy/bipartiteSBM-KL
.. _`CMake`: https://cmake.org/
.. _`Boost`: https://www.boost.org/
.. _`Tzu-Chi Yen`: https://junipertcy.info/
.. _`Daniel B. Larremore`: https://larremorelab.github.io/
.. _`arXiv`: https://arxiv.org/abs/2001.11818
.. _`issues`: https://github.com/junipertcy/bipartiteSBM/issues
.. _`docs`: https://docs.netscied.tw/bipartiteSBM/index.html