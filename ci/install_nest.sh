#!/bin/bash

set -e  # stop execution in case of errors

pip install cython
wget http://www.nest-simulator.org/downloads/gplreleases/nest-2.6.0.tar.gz
tar xzf nest-2.6.0.tar.gz
mkdir -p build/nest
cd build/nest
export VENV=`python -c "import sys; print sys.prefix"`
../../nest-2.6.0/configure --with-mpi --prefix=$VENV
make
make install
cd ../..
