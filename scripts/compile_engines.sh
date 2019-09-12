#!/bin/bash

ls /usr/local/opt/
cd engines/bipartiteSBM-MCMC/ && cmake -DBOOST_ROOT=/usr/local/opt/boost . && make && cd -
cd engines/bipartiteSBM-KL/ && g++ -O3 -Wall -g -pedantic -o biSBM biSBM.cpp && cd -
