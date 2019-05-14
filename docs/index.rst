:mod:`det_k_bisbm` User Guide
=====================================================
Documentation covering the use of ``det_k_bisbm`` library.

``det_k_bisbm`` is a Python implementation of a fast community inference heuristic of the bipartite Stochastic Block Model (biSBM),
using the `MCMC sampler`_ or the `Kernighan-Lin algorithm`_. It estimates the number of communities (as well as the partition) for a bipartite network.

If you have any questions, please contact tzuchi@netscied.tw.

.. toctree::
   :maxdepth: 2
   :caption: Contents

.. toctree::
   :maxdepth: 2
   :caption: Quick start

   usage/why-should-i-use-det_k_bisbm
   usage/installation
   usage/kl-inference
   usage/mcmc-inference
   usage/more-mcmc-inference
   usage/explore-consistency
   usage/generate-synthetic-networks
   usage/make-plots
   usage/parallel
   usage/interpret-entropy

.. toctree::
   :maxdepth: 2
   :caption: Dataset

   dataset/format
   dataset/southern-women
   dataset/malaria
   dataset/ancient-metabolic

.. toctree::
   :maxdepth: 1
   :caption: Module documentation

   src/det_k_bisbm
   src/engines

.. toctree::
   :maxdepth: 2
   :caption: Frequently Asked Questions (FAQ)

   frequently-asked-questions/local-minima
   frequently-asked-questions/choosing-parameters
   frequently-asked-questions/detectability


.. toctree::
   :maxdepth: 2
   :caption: Additional resources

   additional-resources/slides

.. _`MCMC sampler`: https://github.com/junipertcy/bipartiteSBM-MCMC
.. _`Kernighan-Lin algorithm`: https://github.com/junipertcy/bipartiteSBM-KL
