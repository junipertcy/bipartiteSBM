# Installation

We provide two reference engines to illustrate the applicability of our method. They have been added as submodules to this repository. To clone this project along with the submodules, do:
```bash
git clone https://github.com/junipertcy/det_k_bisbm.git --recursive
```
Now enter the directory `det_k_bisbm`. Since the submodules we cloned in the `engines` folder may be out-dated, let's run this command to ensure we have all the newest submodule's content:
```bash
git submodule update
```
Since both of the two modules are C++ subroutines for graph partitioning. To compile these C++ codes, please run the shell script:
```bash
sh scripts/compile_engines.sh
```
If you are new to Python, we suggest you install [Anaconda](https://www.anaconda.com/download/). It will provide most scientific libraries that we need here.

Let's install the libraries that we will use in this heuristic via:
```bash
pip install -r requirements.txt
```
If you are good so far, then we are now ready!

