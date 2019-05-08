# Example MCMC inference
As an example, we'll compute the partition that gives minimum description length on a small graph in `dataset/southernWomen.edgelist`.
To begin, let's first import the class and functions.
```python
from det_k_bisbm.ioutils import *
from det_k_bisbm.optimalks import *
```
If you want to do the graph partitioning using [Markov Chain Monte Carlo](https://github.com/junipertcy/bipartiteSBM-MCMC), do:
```python
from engines.mcmc import *
mcmc = MCMC(f_engine="engines/bipartiteSBM-MCMC/bin/mcmc",  # path to the graph partitioning binary
            n_sweeps=4,                                     # number of partitioning computations for each (K1, K2) data point
            is_parallel=True,                               # whether to compute the partitioning in parallel
            n_cores=2,                                      # if `is_parallel == True`, the number of cores used   
            mcmc_steps=1e5,                                 # [MCMC] the number of sweeps
            mcmc_await_steps=1e4,                           # [MCMC] the number of sweeps to await until stopping the algorithm, if max(entropy) and min(entropy) show no change therein  
            mcmc_cooling="abrupt_cool",                     # [MCMC] annealing scheme used. enum: ["exponential", "logarithm", "linear", "constant", "abrupt_cool"].
            mcmc_cooling_param_1=1e4,                       # [MCMC] parameter 1 for the annealing
            mcmc_cooling_param_2=0.1,                       # [MCMC] parameter 2 for the annealing
            mcmc_epsilon=0.01                               # [MCMC] the "epsilon" value used in the algorithm
        )
```
The `mcmc` is a wrapper class for the C++ engine. It can also generate strings that are useful to run in the command line. 
For example, this code generates a string that tells the C++ program to do graph partition of the `southern women dataset` at `K1=3` and `K2=2`. 

```python
mcmc.prepare_engine("dataset/test/southernWomen.edgelist", 18, 14, 3, 2)
# Out[*]: 'engines/bipartiteSBM-MCMC/bin/mcmc -e dataset/southernWomen.edgelist0 -n 6 6 6 7 7 -t 100 -x 1000 --maximize -c exponential -a 10 0.1 -y 18 14 -z 3 2 -E 0.01 --randomize
``` 

In addition, we have to tell the program which are type-1 nodes and which are type-2. 
We assume that the node index in the dataset runs from all nodes in type-1 first and then type-2. 
Here, we have the number of type-1 nodes as `n1=18`, while number of type-2 nodes is `n2=14`, meaning that the node index
`0 .. 18` are type-1 nodes and `18 .. 32` are type-2 nodes.  

Once we specified the engine that we liked, it's time to prepare the dataset.
```python
edgelist = get_edgelist("dataset/test/southernWomen.edgelist", "\t")
types = get_types("dataset/test/southernWomen.types")
```
We can then feed these three variables into the main class.
```python
oks = OptimalKs(mcmc, edgelist, types)
```
Although there are default parametric values in the heuristic, we suggest you set new ones on your own. 
Here, we set `init_ka=10`, `init_kb=10` and `i_th=0.1`.
```python
oks.set_params(init_ka=10, init_kb=10, i_th=0.1)
```  
We now start the heuristic search via,
```python
oks.iterator()
```
We should expect to wait for a while for the program to finish (if the network is large).
For the `sourthernWomen` dataset, we will see that there are no statistically significant communities other than the fact of being a bipartite network.
That is, we reached a trivial conclusion that `K1=1` and `K2=1`.

If you are interested to run the heuristic again, there's a function that easily clean up the traces that we have gone so far.
```python
oks.clean()
```
Now we can reset the parameters and run the algorithm again! 

In addition, in any case, if one wants to calculate the description length of the data at a single point, `(ka, kb)`, without running through the whole heuristic, one can use,
```python
oks.compute_and_update(ka, kb)
``` 
and then check this inherent variable, which faithfully stores the description lengths that we have computed,
```python
oks.confident_desc_len
```

We have kept a book-keeping of other useful data, too. 
They are `confident_italic_i`, `confident_m_e_rs`, and `trace_mb`. 
We will make a quick tutorial with them in a Jupyter Notebook soon later.
