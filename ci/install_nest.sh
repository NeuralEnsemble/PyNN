#!/bin/bash

set -e  # stop execution in case of errors

if [[ "$TRAVIS_PYTHON_VERSION" == "2.7_with_system_site_packages" ]]; then

    export NEST_VERSION="nest-2.8.0"
    pip install cython
    if [ ! -f "$HOME/$NEST_VERSION/configure" ]; then
        wget https://github.com/nest/nest-simulator/releases/download/v2.8.0/$NEST_VERSION.tar.gz -O $HOME/$NEST_VERSION.tar.gz;
        pushd $HOME;
        tar xzf $NEST_VERSION.tar.gz;
        popd;
    else
        echo 'Using cached version of NEST sources.';
    fi
    mkdir -p $HOME/build/$NEST_VERSION
    pushd $HOME/build/$NEST_VERSION
    if [ ! -f "$HOME/build/$NEST_VERSION/config.log" ]; then
        export VENV=`python -c "import sys; print sys.prefix"`;
        $HOME/$NEST_VERSION/configure --with-mpi --prefix=$VENV;
        make;
    else
        echo 'Using cached NEST build directory.';
        echo "$HOME/$NEST_VERSION";
        ls $HOME/$NEST_VERSION;
        echo "$HOME/build/$NEST_VERSION";
        ls $HOME/build/$NEST_VERSION;
    fi
    make install
    popd

fi