<p align="center">
  <a href="https://github.com/junipertcy/det_k_bisbm">
    <img width="800" src="http://wiki.junipertcy.info/images/1/10/Det_k_bisbm-logo.png">
  </a>
</p>

# det_k_bisbm

Python implementation of a fast community number determination heuristic of the bipartite Stochastic Block Model (biSBM), 
using the MCMC sampler or the Kernighan-Lin algorithm as the inference engine. 

This program utilizes the Minimum Description Length principle to determine a point estimate of the
numbers of communities in the biSBM that best compresses the model and data. Several test examples are included.

## Table of content

1. [Usage](#usage)
    1. [Installation](#installation)
    2. [Example MCMC inference](#example-mcmc)
    3. [Example Kerininghan-Lin inference](#example-kl)
2. [Datasets](#datasets)
2. [Versions](#versions)  
3. [Companion article](#companion-article)

## Usage

### Installation

If you are new to Python, we suggest you install [Anaconda](https://www.anaconda.com/download/). It will provide most scientific libraries that we need here.

If you want to enable parallelization for the engine, please additionally install `pathos` via:
```bash
pip install git+https://github.com/uqfoundation/pathos.git@master
```
We provide two reference engines to illustrate the applicability of our method. They have been added as submodules to this repository. To clone this project along with the submodules, do:
```bash
git clone git@github.com:junipertcy/det_k_bisbm.git --recursive --init
```
Now enter the directory `det_k_bisbm`. Since the submodules we cloned in the `engines` folder are still empty, let's run this command to ensure we have all the submodule's content:
```bash
git submodule update
```
Since both of the two modules are C++ subroutines for graph partitioning. To compile these C++ codes, please run the shell script:
```bash
sh start.sh
```
If you are good so far, then we are now ready!

### Example MCMC inference

To use this library, let's first import the class and functions.
```python
from det_k_bisbm.ioutils import *
from det_k_bisbm.optimalks import *
```
If you want to do the graph partitioning using Markov Chain Monte Carlo, do:
```python
from engines.mcmc import *
mcmc = MCMC(f_engine="engines/bipartiteSBM-MCMC/bin/mcmc",  # path to the graph partitioning binary
            n_sweeps=2,                                     # number of partitioning computations for each (K1, K2) data point
            is_parallel=True,                               # whether to compute the partitioning in parallel
            n_cores=2,                                      # if `is_parallel == True`, the number of cores used   
            mcmc_steps=100,                                 # [MCMC] the number of sweeps
            mcmc_await_steps=1000,                          # [MCMC] the number of sweeps to await to stop the algorithm, if max(entropy) and min(entropy) show no change therein  
            mcmc_cooling="exponential",                     # [MCMC] annealing scheme used. enum: ["exponential", "logarithm", "linear", "constant"].
            mcmc_cooling_param_1=10,                        # [MCMC] parameter 1 for the annealing
            mcmc_cooling_param_2=0.1,                       # [MCMC] parameter 2 for the annealing
            mcmc_epsilon=0.01                               # [MCMC] the "epsilon" value used in the algorithm
        )                                
```
The `mcmc` is a wrapper class for the C++ engine. It can also generate strings that are useful to run in the command line. 
For example, this code generates a string that tells the C++ program to do graph partition of the `southern women dataset` at `K1=3` and `K2=2`. 
```python
mcmc.prepare_engine("dataset/southernWomen.edgelist", 18, 14, 3, 2)
# Out[*]: 'engines/bipartiteSBM-MCMC/bin/mcmc -e dataset/southernWomen.edgelist0 -n 6 6 6 7 7 -t 100 -x 1000 --maximize -c exponential -a 10 0.1 -y 18 14 -z 3 2 -E 0.01 --randomize
``` 
In addition, we have to tell the program which are type-1 nodes and which are type-2. 
We assume that the node index in the dataset runs from all nodes in type-1 first and then type-2. 
Here, we have the number of type-1 nodes as `n1=18`, while number of type-2 nodes is `n2=14`, meaning that the node index
`0 .. 18` are type-1 nodes and `18 .. 32` are type-2 nodes.  

Once we specified the engine that we liked, it's time to prepare the dataset.
```python
edgelist = get_edgelist("dataset/southernWomen.edgelist", " ")
types= get_types("dataset/southernWomen.types")
```
We can then feed these three variables into the main class.
```python
oks = OptimalKs(mcmc, edgelist, types)
```
Although there are default parametric values the heuristic, we suggest you set new ones on your own. 
Here, we set `init_ka=50`, `init_kb=50` and `i_th=0.1`.
```python
ops.set_params(init_ka=50, init_kb=50, i_th=0.1)
```  
We now start the heuristic search via,
```python
oks.iterator()
```
We should expect for a while for the program to finish.

If you want to run the heuristic again, there's a function that easily clean up the traces that we have gone so far.
```python
oks.clean()
```
Now we can set new parameters and re-run the algorithm!

### Example Kerninghan-Lin inference

The algorithm for bipartite community detection is independent to the graph partitioning algorithm used. 
In principle, one could switch to a different partitioning engine and infer the number of communities in a similar manner.
Here, we illustrate the use of the Kerninghan-Lin algorithm in the heuristic.

If one wants to do the graph partitioning using Kerninghan-Lin, one initiates a different engine class:
```python
from engines.kl import *
kl = KL(f_engine="engines/bipartiteSBM-KL/biSBM",
        n_sweeps=2,
        is_parallel=True,
        n_cores=2,
        kl_edgelist_delimiter="\t",                        # [KL] due to the KL code accepts 1-indexed nodes by default, we used the delimiter to transform our 0-indexed input.  
        kl_itertimes=1,                                    # [KL] the number of random initializations 
        f_kl_output="engines/bipartiteSBM-KL/f_kl_output"  # [KL] path to the KL output dir; recommended to be in the same folder as the binary
    )
```
Similarly, one can generate the string for command line computation. 
Note that since all outputs will appear in one specified `f_kl_output` and we may have `kl_itertimes` random initializations,
hence the subfolders in `f_kl_output` is hashed in order to place the output data nicely.
```python
kl.prepare_engine("dataset/southernWomen.edgelist", 18, 14, 2, 3)
```
The remaining codes are similar.
```python
edgelist = get_edgelist("dataset/southernWomen.edgelist", ",")
types= kl.gen_types(18, 14)

oks = OptimalKs(kl, edgelist, types)
ops.set_params(init_ka=10, init_kb=10, i_th=0.1)
oks.iterator()
```
We might expect to wait longer in larger networks since Kerninghan-Lin is slower than the MCMC algorithm.


## Datasets

(to be written)

## Versions

`Version 1.1` - `2017-08-31` - Moved to branch `master`.

`Version 1.0` - `2017-06-20` - A `pre-release` version that does model selection for bipartite SBM.

## Companion article

**Estimating the Number of Communitites in a Bipartite Network**

Tzu-Chi Yen and Daniel B. Larremore, *in preparation*.
