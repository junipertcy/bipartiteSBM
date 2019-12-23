Local minima
============
* The algorithm gets trapped in a local minimum easily in my data. What can I do?

  This happens when the modular structure in your data is not strong.

  Try increase the neighborhood range for the determination of minimum via
  `oks.set_k_th_neighbor_to_search(h)`, where h being larger than 2 (default to be 2).

  In practice, you should run the algorithm several times to determine the global optimum.
