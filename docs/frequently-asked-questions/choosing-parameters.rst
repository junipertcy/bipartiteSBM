Choosing parameters
===================
* There are "setters" for some parameters, what are the most impactful knobs to tune?

  To increase accuracy, you might want to increase `nm` (defaults to 10) or the neighborhood size (defaults to 2)
  with ``set_nm`` and ``set_k_th_neighbor_to_search``, respectively. The value of `nm` will affect the number of
  merges per block in the off-by-one :math:`e`-matrix merging step. And the `k_th_neighbor_to_search` (call it k)
  parameter controls the number of points to check around a suspected local minimum. For example, setting `k=2`
  means we have to do the (more demanding) graph partitioning steps for `(2k+1) * (2k+1) = 25` grid points, with the
  suspected point lying at the center.

  To increase efficiency, you might want to increase `c` (defaults to 3) with ``set_c``. This will enable the algorithm
  to skip multiple graph partitioning steps, at the cost of being prone to overshoot and getting trapped in a local
  optimum.
|
* How could I choose `init_ka`, `init_kb`, and `i_th`?

  You do not need to do that. These values are automatically determined by the algorithm.
  But they are helpful for debug.  :-)
