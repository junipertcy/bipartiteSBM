Installation
============

:mod:`biSBM` is a Python library, and it depends on C++ programs for faster SBM inference.
We provide two reference engines for this purpose; they have been added as submodules to this repository.
But you will still need `CMake <https://cmake.org/>`_ and `Boost Libraries <https://www.boost.org/>`_ to compile them.
Please refer to their official pages for installation instructions.
For macOS users, you may want to run `brew install` for ``cmake`` and ``boost``.

To clone this project along with the submodules, do: ::

   git clone https://github.com/junipertcy/bipartiteSBM.git --recursive

Now enter the directory ``bipartiteSBM``.
Since the submodules we cloned in the :mod:`engines` folder may be out-dated,
let's run this command to ensure we have all the newest submodule's content: ::

   git submodule update

These modules are C++ subroutines for graph partitioning. To compile them, please run this shell script: ::

   sh scripts/compile_engines.sh

Lastly, we have to install Python library dependencies, by simply running this command: ::

   python -m pip install -r requirements.txt

If you are good so far, then we are good to go!
