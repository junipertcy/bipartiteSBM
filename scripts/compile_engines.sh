#!/bin/bash

python3.7-config --ldflags
cd engines/bipartiteSBM-MCMC/ && cmake . && make && cd -
cd engines/bipartiteSBM-KL/ && g++ -O3 -Wall -g -pedantic -o biSBM biSBM.cpp && cd -
