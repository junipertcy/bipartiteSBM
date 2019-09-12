#!/bin/bash

# Add the miniconda bin directory to $PATH
#export PATH=$HOME/miniconda3/bin:$PATH
echo $PATH

# Use the miniconda installer for setup of conda itself
pushd .
cd
mkdir -p download
cd download
if [[ ! -f $HOME/miniconda3/bin/activate ]]
    then
    if [[ ! -f miniconda.sh ]]
        then
        if [[ "$TRAVIS_OS_NAME" == "linux" ]]
        then
            wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
        elif [[ "$TRAVIS_OS_NAME" == "osx" ]]
        then
             echo $PATH
#            wget http://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
            brew update
#            brew upgrade
            brew install boost-python cmake
            python3.7-config --ldflags
        fi
    fi
#    chmod +x miniconda.sh && ./miniconda.sh -b -f
#    conda update --yes conda
#    conda create -n testenv36 --yes python=3.6
#    conda create -n testenv37 --yes python=3.7
fi
cd ..
popd

export INSTALL_TEST_REQUIREMENTS="true"
echo $INSTALL_TEST_REQUIREMENTS