#!/bin/bash

set -e  # stop execution in case of errors

    export NEST_VERSION="master"
    export NEST="nest-simulator-$NEST_VERSION"
    pip install cython
    wget https://github.com/nest/nest-simulator/archive/$NEST_VERSION.tar.gz -O $HOME/$NEST.tar.gz;
    pushd $HOME;
    tar xzf $NEST.tar.gz;
    popd;

    mkdir -p $HOME/build/$NEST
    pushd $HOME/build/$NEST
    export VENV=`python -c "import sys; print sys.prefix"`;
    echo $VENV
    echo $PWD
    cmake -DCMAKE_INSTALL_PREFIX=$VENV -Dwith-mpi=ON $HOME/$NEST;
    make;
    make install
    popd
