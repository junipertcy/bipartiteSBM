#!/bin/bash

pushd engines/bipartiteSBM-MCMC/ && cmake . && make && popd
pushd engines/bipartiteSBM-KL/ && g++ -O3 -Wall -g -pedantic -o biSBM biSBM.cpp && popd
