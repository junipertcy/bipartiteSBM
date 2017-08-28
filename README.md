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

If you are new to Python, we recommend you install `Anaconda`. It will provide most scientific libraries that we need here.

If you want to enable parallelization for the engine, please additionally install `pathos` via:
```python
pip install git+https://github.com/uqfoundation/pathos.git@master
```
We provide two reference engines to illustrate the applicability of our method. They have been added as submodules to this repository. To clone this project along with the submodules, do:
```python
git clong git@github.com:junipertcy/det_k_bisbm.git --recursive
```
Both of the two modules are C++ subroutines for graph partitioning. To compile these C++ codes, please run the shell script:
```bash
sh start.sh
```
If you are good so far, then we are now ready!

## Example MCMC inference

To use this library, let's first import the class and functions.
```python
from det_k_bisbm.ioutils import *
from det_k_bisbm.optimalks import *
```
If you want to do the graph partitioning using Markov Chain Monte Carlo, do:
```python
from engines.mcmc import *
mcmc = MCMC(f_engine="engines/bipartiteSBM-MCMC/bin/mcmc",
            n_sweeps=2,
            is_parallel=True,
            n_cores=2,
            mcmc_steps=100,
            mcmc_await_steps=1000,
            mcmc_cooling="exponential",
            mcmc_cooling_param_1=10,
            mcmc_cooling_param_2=0.1,
            mcmc_epsilon=0.01)
```
The `mcmc` is a wrapper class for the C++ engine. It can also generate string that are useful to run in the command line. 
For example, this code generates a string that tells the C++ program to do graph partition of the `southern women dataset` at `K1=2` and `K2=3`. 
In addition, we have to tell the program which are type-1 nodes and which are type-2. 
We assume that the node index in the dataset runs from all nodes in type-1 first and then type-2. 
Here, we have the number of type-1 nodes as `n1=18`, while number of type-2 nodes is `n2=14`, meaning that the node index
`0 .. 18` are type-1 nodes and `18 .. 32` are type-2 nodes.  
```python
mcmc.prepare_engine("dataset/southernWomen.edgelist0", 18, 14, 2, 3)
``` 
Once we specified the engine that we liked, it's time to prepare the dataset.
```python
edgelist = get_edgelist("dataset/southernWomen.edgelist0", " ")
types= get_types("dataset/southernWomen.types")
```

We can then feed these three variables into the main class.
```python
oks = OptimalKs(mcmc,
                edgelist, 
                types,
                init_Ka=50,
                init_Kb=50,
                i_th=0.1)
```
We now start the heuristic search via,
```python
oks.iterator()
```
We should expect a while for the program to finish.

## Example Kerininghan-Lin inference

(to be written)

## Datasets

(to be written)

## Versions

`Version 1.0` - `2017-06-20` - A pre-release version that does model selection for bipartite SBM.

## Companion article

**Estimating the Number of Communitites in a Bipartite Network**

Tzu-Chi Yen and Daniel B. Larremore, *in preparation*.


--

More to update soon.

TC