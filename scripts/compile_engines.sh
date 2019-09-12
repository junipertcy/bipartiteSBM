#!/bin/bash

cd engines/bipartiteSBM-MCMC/ && cmake -DBOOST_ROOT=/usr/local/opt/boost@1.60 . && make && cd -
cd engines/bipartiteSBM-KL/ && g++ -O3 -Wall -g -pedantic -o biSBM biSBM.cpp && cd -
