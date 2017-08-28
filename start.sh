#!/bin/bash

pushd engines/mcmc/ && cmake . && make && popd
pushd engines/kl/ && g++ -O3 -Wall -g -pedantic -o biSBM biSBM.cpp && popd
