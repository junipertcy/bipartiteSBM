=================================
The :mod:`det_k_bisbm` User Guide
=================================
Documentation covering the use of the ``det_k_bisbm`` library.

The ``det_k_bisbm`` is a Python implementation of a fast community inference heuristic of the bipartite Stochastic Block Model (biSBM),
using the `MCMC sampler`_ or the `Kernighan-Lin algorithm`_. It estimates the number of communities (as well as the partition) for a bipartite network.

If you have any questions, please contact tzuchi@netscied.tw.

.. toctree::
   :maxdepth: 2
   :caption: Contents

.. toctree::
   :maxdepth: 1
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
   :maxdepth: 1
   :caption: Dataset

   dataset/format
   dataset/node-type-ordering
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

Acknowledgements
----------------

The `det_k_bisbm` is inspired and supported by the following great humans,
`Daniel B. Larremore`_, `Tiago de Paula Peixoto`_, `Jean-Gabriel Young`_, `Pan Zhang`_, and `Jie Tang`_.


.. _`MCMC sampler`: https://github.com/junipertcy/bipartiteSBM-MCMC
.. _`Kernighan-Lin algorithm`: https://github.com/junipertcy/bipartiteSBM-KL
.. _`Daniel B. Larremore`: http://danlarremore.com/
.. _`Tiago de Paula Peixoto`: https://skewed.de/tiago
.. _`Jean-Gabriel Young`: https://www.jgyoung.ca/
.. _`Jie Tang`: http://keg.cs.tsinghua.edu.cn/jietang/
.. _`Pan Zhang`: http://lib.itp.ac.cn/html/panzhang/
