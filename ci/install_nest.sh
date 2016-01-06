#!/bin/bash

set -e  # stop execution in case of errors

if [[ "$TRAVIS_PYTHON_VERSION" == "2.7_with_system_site_packages" ]]; then

    export NEST_VERSION="2.6.0"
    export NEST="nest-$NEST_VERSION"
    pip install cython
    if [ ! -f "$HOME/$NEST_VERSION/configure" ]; then
        wget https://github.com/nest/nest-simulator/releases/download/v$NEST_VERSION/$NEST.tar.gz -O $HOME/$NEST.tar.gz;
        pushd $HOME;
        tar xzf $NEST.tar.gz;
        popd;
    else
        echo 'Using cached version of NEST sources.';
    fi
    mkdir -p $HOME/build/$NEST
    pushd $HOME/build/$NEST
    if [ ! -f "$HOME/build/$NEST/config.log" ]; then
        export VENV=`python -c "import sys; print sys.prefix"`;
        $HOME/$NEST/configure --with-mpi --prefix=$VENV;
        make;
    else
        echo 'Using cached NEST build directory.';
        echo "$HOME/$NEST";
        ls $HOME/$NEST;
        echo "$HOME/build/$NEST";
        ls $HOME/build/$NEST;
    fi
    make install
    popd

fi
