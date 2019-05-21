Example MCMC inference
======================
As an example, we are going to find the partition that gives the minimum description length on a small graph
in `dataset/southernWomen.edgelist`.

To begin, let's first import the functions that we need. ::

   from det_k_bisbm.ioutils import get_edgelist, get_types
   from det_k_bisbm.optimalks import *

Here, we will fit the biSBM via the `Markov Chain Monte Carlo <https://github.com/junipertcy/bipartiteSBM-MCMC>`_ algorithm. ::

   from engines.mcmc import *
   mcmc = MCMC(f_engine="engines/bipartiteSBM-MCMC/bin/mcmc")

The :class:`engines.mcmc` is a wrapper class for the C++ engine. It generates the commandline input that runs as a spawned
process in the background. In this example, it will generate a string that tells the C++ program to do graph partition
of the `southern women dataset` at :math:`K_a=3` and :math:`K_b=2`. ::

   mcmc.prepare_engine("dataset/test/southernWomen.edgelist", 18, 14, 3, 2)
   # Out[*]: 'engines/bipartiteSBM-MCMC/bin/mcmc -e dataset/test/southernWomen.edgelist -n 6 6 6 7 7 -t 1000000 -x 100000 -c abrupt_cool -a 100000.0 -y 18 14 -z 3 2 -E 0.001 -g

In addition, we have to tell the program which are type-`a` nodes and which are type-`b`.
We assume that the node indices in the dataset run in a specific order; that is,
nodes of type-`a` are indexed first and then followed by nodes of type-`b`.
For example, we have the number (i.e., size) of type-1 nodes as :math:`n_a=18`,
while number of type-2 nodes is :math:`n_b=14`.

This means that the node indices :math:`0 \dots 18` are type-`a` nodes and :math:`18 \dots 32` are type-`b` nodes.
Once we specified the engine that we liked, it's time to bake the dataset!  ::

   edgelist = get_edgelist("dataset/test/southernWomen.edgelist", "\t")
   types = get_types("dataset/test/southernWomen.types")

For `types`, we can also use,  ::

   types = mcmc.gen_types(n1, n2)  # n1=18 & n2=14

We can then feed these three variables into the main class.  ::

   oks = OptimalKs(mcmc, edgelist, types)

Now, we can start the heuristic search!  ::

    oks.minimize_bisbm_dl()
    # Out[*]:
    # OrderedDict([((1, 1), 191.72536162138402),
    #              ((6, 7), 227.47573446636372),
    #              ((2, 1), 199.25454713207995),
    #              ((1, 2), 197.11255878689138),
    #              ((1, 3), 202.30070134785217),
    #              ((2, 2), 199.5005556514051),
    #              ((2, 3), 196.01966191156208),
    #              ((3, 1), 203.69566837215007),
    #              ((3, 2), 204.73016933010257),
    #              ((3, 3), 201.34320240110824)])

We should expect to wait for a minute or two for the program to complete (depending on the size of the network).
For the `sourthernWomen` dataset, we will see that there are no statistically significant communities other than the
act of being a bipartite network. That is, we reached a trivial conclusion that :math:`K_a=1` and :math:`K_b=1`.

If you are interested to run the heuristic again,
just to check the consistency of the result,
you can re-initiate the :class:`det_k_bisbm.optimalks.OptimalKs` class and then re-do the `minimize_bisbm_dl()`.

Once the algorithm stops, it will output the trace and each respective entropy (a.k.a, description length).
We may simply run,  ::

    oks.summary()
    # Out[*]:
    # {'algm_args': {'init_ka': 6, 'init_kb': 7, 'i_0': 0.3065136807765042},
    #  'na': 18,
    #  'nb': 14,
    #  'e': 89,
    #  'avg_k': 5.5625,
    #  'ka': 1,
    #  'kb': 1,
    #  'mdl': 191.72536162138402}

We conclude a trivial bipartite partition for the southern women dataset!
