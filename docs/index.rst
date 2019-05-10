Welcome to ``det_k_bisbm``'s documentation
=====================================================
``det_k_bisbm`` is a Python implementation of a fast community number determination heuristic of the bipartite Stochastic Block Model (biSBM),
using the `MCMC sampler`_ or the `Kernighan-Lin algorithm`_ as the inference engine.

.. toctree::
   :maxdepth: 2
   :caption: Contents

.. toctree::
   :maxdepth: 2
   :caption: Quick start

   usage/installation
   usage/kl-inference
   usage/mcmc-inference
   usage/explore-consistency

.. toctree::
   :maxdepth: 2
   :caption: Dataset

   dataset/format
   dataset/southern-women

.. toctree::
   :maxdepth: 2
   :caption: Module documentation

   src/det_k_bisbm
   src/engines

.. toctree::
   :maxdepth: 2
   :caption: Frequently Asked Questions (FAQ)

   frequently-asked-questions/detectability
   frequently-asked-questions/local-minima
   frequently-asked-questions/choosing-parameters

.. toctree::
   :maxdepth: 2
   :caption: Additional resources

   additional-resources/slides

.. _`MCMC sampler`: https://github.com/junipertcy/bipartiteSBM-MCMC
.. _`Kernighan-Lin algorithm`: https://github.com/junipertcy/bipartiteSBM-KL
