# det_k_bisbm
[![Twitter: @oneofyen](https://img.shields.io/badge/contact-@oneofyen-blue.svg?style=flat)](https://twitter.com/oneofyen) 
[![License](https://img.shields.io/badge/license-GPL-green.svg?style=flat)](https://github.com/junipertcy/det_k_bisbm/blob/master/LICENSE)
[![Build Status](https://travis-ci.org/junipertcy/det_k_bisbm.svg?branch=master)](https://travis-ci.org/junipertcy/det_k_bisbm)

Python implementation of a fast community number determination heuristic of the bipartite Stochastic Block Model (biSBM), 
using the [MCMC sampler](https://github.com/junipertcy/bipartiteSBM-MCMC)
or the [Kernighan-Lin algorithm](https://github.com/junipertcy/bipartiteSBM-KL) as the inference engine. 

<p align="center">
  <a href="https://github.com/junipertcy/det_k_bisbm">
    <img width="800" src="http://wiki.junipertcy.info/images/1/10/Det_k_bisbm-logo.png">
  </a>
</p>

---

`det_k_bisbm` utilizes the Minimum Description Length principle to determine a point estimate of the
numbers of communities in the biSBM that best compresses the model and data. Several test examples are included.

If you want to sample the whole marginal distribution of the number of communities, rather than a point estimate,
please check the companion [Markov Chain Monte Carlo](https://github.com/junipertcy/bipartiteSBM-MCMC) program.

Both Python 2.7 and 3.6 are supported and tested.

## Table of content

- [Usage](#usage)
    - [Installation](#installation)
    - [Example MCMC inference](#example-mcmc)
    - [Example Kerininghan-Lin inference](#example-kl)
- [Dataset](#dataset)
- [Versions](#versions)  
- [Companion article](#companion-article)

## Usage

### Installation

If you are new to Python, we suggest you install [Anaconda](https://www.anaconda.com/download/). It will provide most scientific libraries that we need here.

If you want to enable parallelization for the engine, please additionally install `loky` via:
```bash
pip install loky
```
We provide two reference engines to illustrate the applicability of our method. They have been added as submodules to this repository. To clone this project along with the submodules, do:
```bash
git clone git@github.com:junipertcy/det_k_bisbm.git --recursive
```
Now enter the directory `det_k_bisbm`. Since the submodules we cloned in the `engines` folder may be out-dated, let's run this command to ensure we have all the newest submodule's content:
```bash
git submodule update
```
Since both of the two modules are C++ subroutines for graph partitioning. To compile these C++ codes, please run the shell script:
```bash
sh scripts/compile_engines.sh
```
If you are good so far, then we are now ready!

### <a id="example-mcmc"></a>Example MCMC inference

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
            n_sweeps=10,                                    # number of partitioning computations for each (K1, K2) data point
            is_parallel=True,                               # whether to compute the partitioning in parallel
            n_cores=2,                                      # if `is_parallel == True`, the number of cores used   
            mcmc_steps=100000,                              # [MCMC] the number of sweeps
            mcmc_await_steps=10000,                         # [MCMC] the number of sweeps to await until stopping the algorithm, if max(entropy) and min(entropy) show no change therein  
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
edgelist = get_edgelist("dataset/southernWomen.edgelist", "\t")
types = get_types("dataset/southernWomen.types")
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
They are `confident_italic_i`, `confident_m_e_rs`, and `confident_mb`. 
We will make a quick tutorial with them in a Jupyter Notebook soon later.


### <a id="example-kl"></a>Example Kerninghan-Lin inference

The algorithm for bipartite community detection is independent to the graph partitioning algorithm used. 
In principle, one could switch to a different partitioning engine and infer the number of communities in a similar manner.
Here, we illustrate the use of the [Kerninghan-Lin algorithm](https://github.com/junipertcy/bipartiteSBM-KL) in the heuristic.

If one wants to do the graph partitioning using Kerninghan-Lin, one initiates a different engine class:
```python
from engines.kl import *
kl = KL(f_engine="engines/bipartiteSBM-KL/biSBM",
        n_sweeps=2,                                        # Note that this will generate <n_sweeps> output sub-folders in <f_kl_output>
        is_parallel=True,
        n_cores=2,
        kl_edgelist_delimiter="\t",                        # [KL] due to the KL code accepts 1-indexed nodes by default, we used the delimiter to transform our 0-indexed input.  
        kl_steps=4,                                        # [KL] the number of random initializations (see the README_cplusplus.txt file)
        kl_itertimes=1,                                    # [KL] the number of KL runs (within each <outputFOLDER>) for returning an optimal result
        f_kl_output="engines/bipartiteSBM-KL/f_kl_output"  # [KL] path to the KL output dir; recommended to be in the same folder as the binary
    )
```
It performs `<n_sweeps> * <kl_itertimes>` KL runs before returning the membership assignment with the highest likelihood.
Note that since all outputs will appear in one specified `f_kl_output` and we may have `n_sweeps` (parallel) runs,
hence the subfolders in `f_kl_output` are hashed in order to place the output data nicely.

Similarly, one can generate the string for command line computation.
This time we test the algorithm on an example graph in `dataset/bisbm-n_1000-ka_4-kb_6-r-1.0-Ka_30-Ir_1.75.gt.edgelist`.
This is a synthetic network with `K1=4` and `K2=6`, generated by the bipartite SBM. 

```python
kl.prepare_engine("dataset/bisbm-n_1000-ka_4-kb_6-r-1.0-Ka_30-Ir_1.75.gt.edgelist", 500, 500, 6, 7, delimiter="\t")
```

The remaining codes are similar.
```python
edgelist = get_edgelist("dataset/bisbm-n_1000-ka_4-kb_6-r-1.0-Ka_30-Ir_1.75.gt.edgelist", "\t")
types = kl.gen_types(500, 500)

oks = OptimalKs(kl, edgelist, types)
oks.set_params(init_ka=10, init_kb=10, i_th=0.1)
oks.iterator()
```
We will see that it correctly finds `K1=4` and `K2=6` as a result.
Note that Kerninghan-Lin is generally slower than the MCMC algorithm when the number of communities is large.

## Dataset

This program accepts input data as a text file of graph adjacencies, say `graph.edgelist`, which contains one edge per line. Each line follows an out-neighbor adjacency list format; that is, a 2-tuple of node indexes of the form,
```ini
<node_source_id_1> <out_neighbor_id_1>
<node_source_id_1> <out_neighbor_id_2>
...
<node_source_id_1> <out_neighbor_id_<outdegree>>
...
```
The `get_edgelist` function can resolve text files with custom delimiters.

## Versions

`Version 1.1` - `2017-08-31` - Moved to branch `master`.

`Version 1.0` - `2017-06-20` - A `pre-release` version that does model selection for bipartite SBM.

## Companion article

If you find `det_k_bisbm` useful for your research, please consider citing the following paper:

**Estimating the Number of Communitites in a Bipartite Network**

Tzu-Chi Yen and Daniel B. Larremore, *in preparation*.
