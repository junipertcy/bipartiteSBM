# Choosing parameters
* How could I choose `init_ka`, `init_kb`, and `i_th`?

  Without other prior information, one could choose `init_ka = init_kb = sqrt(e) / 2`, where `e` is the number of edges.
  Roughly above this value, the algorithm will suffer a resolution limit, 
  and it doesn’t make any difference whichever value is set. 
  
  Regarding the `i_th`,
  it is a parameter to avoid the (comparatively) more computationally expensive graph partitioning calculation. 
  Setting it close to 1 enables the algorithm to skip multiple graph partitioning steps,
  at the cost of being prone to overshoot and getting trapped in a local optimum.
  Valid values for `i_th` is `[0, 1)`, one could simply start with `i_th = 0.1`.  
