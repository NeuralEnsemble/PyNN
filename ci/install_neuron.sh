#!/bin/bash

set -e  # stop execution in case of errors

if [ "$TRAVIS_PYTHON_VERSION" == "3.7" ]; then
    echo -e "\n========== Installing NEURON ==========\n"
    export NRN_VERSION="nrn-7.7"
    if [ ! -f "$HOME/$NRN_VERSION/configure" ]; then
        wget http://www.neuron.yale.edu/ftp/neuron/versions/v7.7/$NRN_VERSION.tar.gz -O $HOME/$NRN_VERSION.tar.gz;
        pushd $HOME;
        tar xzf $NRN_VERSION.tar.gz;
        popd;
    else
        echo 'Using cached version of NEURON sources.';
    fi
    mkdir -p $HOME/build/$NRN_VERSION
    pushd $HOME/build/$NRN_VERSION
    export VENV=`python -c "import sys; print(sys.prefix)"`;
    if [ ! -f "$HOME/build/$NRN_VERSION/config.log" ]; then
        $HOME/$NRN_VERSION/configure --with-paranrn --with-nrnpython=$VENV/bin/python --prefix=$VENV --disable-rx3d --without-iv;
        make;
    else
        echo 'Using cached NEURON build directory.';
    fi
    make install
    cd src/nrnpython
    python setup.py install

    pip install nrnutils  # must be installed after NEURON

    # compile PyNN NMODL mechanisms
    cd $VENV/bin;
    ls -l;
    ln -sf ../x86_64/bin/nrnivmodl;

    echo $TRAVIS_BUILD_DIR
    cd $TRAVIS_BUILD_DIR/pyNN/neuron/nmodl
    nrnivmodl

    popd

fi