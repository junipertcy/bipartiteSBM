==================================
The :mod:`bipartiteSBM` User Guide
==================================
.. include:: ../README.rst
   :start-line: 13
   :end-before: Documentation

If you have any questions, please contact tzuchi@netscied.tw.

.. toctree::
   :maxdepth: 2
   :caption: Contents

.. toctree::
   :maxdepth: 1
   :caption: Quick start

   usage/why-should-i-use-bipartiteSBM
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

   src/biSBM
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

The `bipartiteSBM` is inspired and supported by the following great humans,
`Daniel B. Larremore`_, `Tiago de Paula Peixoto`_, `Jean-Gabriel Young`_, `Pan Zhang`_, and `Jie Tang`_.
Thanks `Valentin Haenel`_ who helped debug and fix the Numba code.


.. _`MCMC sampler`: https://github.com/junipertcy/bipartiteSBM-MCMC
.. _`Kernighan-Lin algorithm`: https://github.com/junipertcy/bipartiteSBM-KL
.. _`Daniel B. Larremore`: http://danlarremore.com/
.. _`Tiago de Paula Peixoto`: https://skewed.de/tiago
.. _`Jean-Gabriel Young`: https://www.jgyoung.ca/
.. _`Jie Tang`: http://keg.cs.tsinghua.edu.cn/jietang/
.. _`Pan Zhang`: http://lib.itp.ac.cn/html/panzhang/
.. _`Valentin Haenel`: http://haenel.co/
