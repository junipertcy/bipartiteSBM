# Local minima?
* The algorithm gets trapped in a local minimum easily in my data. What can I do?

  This happens when the modular structure in your data is not strong.
  Try `using a lower starting (Ka, Kb)` or 
  increase the checking range for the determination of minimum via `setting oks.set_k_th_neighbor_to_search(K)`,
  where K is a larger number (default to be 1). 
  In principle, you are suggested to run the algorithm several times to determine the global optimum.