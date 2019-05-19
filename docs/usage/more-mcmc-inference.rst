More MCMC inference
===================

We can customize the MCMC inference by initiating the engine instance differently.


Annealing schemes
-----------------
We provide 5 annealing schemes that possibly relax the Markov chain to the global minimum on the description length
landscape. Nevertheless, as suggested by [peixoto-efficient-2014]_ the default annealing scheme is set
to ``abrupt_cool``.

For ``abrupt_cool``, only the ``mcmc_cooling_param_1`` is useful. It corresponds to the number of sweeps for
equilibrium (:math:`T=1`) after which an abrupt cooling (:math:`T=0`) is performed.

As inspired by `sbm_canonical_mcmc <https://github.com/jg-you/sbm_canonical_mcmc>`_, the other 4 cooling schedules are:
``exponential``, ``linear``, ``logarithmic`` and ``constant``.

The inverse temperature functions are defined as ::

   beta(t) = 1/T_0 * alpha^(-t)                (Exponential)
   beta(t) = 1/T_0 * [1 - eta * t / T_0]^(-1)  (Linear)
   beta(t) = log(t + d) / c                    (Logarithmic)
   beta(t) = 1 / T_0                           (Constant),

where :math:`t` is the MCMC step. The parameters of these cooling schedules are passed like this: ::

   T_0 alpha    (Exponential)
   T_0 eta      (Linear)
   c d          (Logarithmic)
   T_0          (Constant),

where the first argument fulfills the ``mcmc_cooling_param_1`` in :class:`engines.MCMC` and the second one corresponds
to ``mcmc_cooling_param_2``. Note that the second parameter is meaningless for ``logarithmic`` and ``constant``.


References
~~~~~~~~~~
.. [peixoto-efficient-2014] Tiago P. Peixoto, "Efficient Monte Carlo and
   greedy heuristic for the inference of stochastic block models", Phys.
   Rev. E 89, 012804 (2014), :doi:`10.1103/PhysRevE.89.012804`,
   :arxiv:`1310.4378`