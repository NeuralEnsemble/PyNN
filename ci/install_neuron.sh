#!/bin/bash

set -e  # stop execution in case of errors

if [ "$TRAVIS_PYTHON_VERSION" == "3.9" ]; then
    echo -e "\n========== Installing NEURON ==========\n"
    export NRN_VERSION="nrn-8.1"
    export VENV=`python -c "import sys; print(sys.prefix)"`

    if [ ! -f "$HOME/$NRN_VERSION/build/CMakeCache.txt" ]; then
        echo 'Cloning NEURON sources from GitHub'
        git clone https://github.com/neuronsimulator/nrn -b release/8.1 $HOME/$NRN_VERSION
        mkdir -p $HOME/$NRN_VERSION/build
    else
        echo 'Using cached NEURON build directory.'
    fi
    pushd $HOME/$NRN_VERSION/build

    cmake .. -DNRN_ENABLE_INTERVIEWS=OFF -DNRN_ENABLE_MPI=ON -DNRN_ENABLE_RX3D=OFF -DCMAKE_INSTALL_PREFIX=$VENV -DNRN_MODULE_INSTALL_OPTIONS=""
    cmake --build . --target install

    pip install nrnutils  # must be installed after NEURON

    # compile PyNN NMODL mechanisms
    echo $TRAVIS_BUILD_DIR
    cd $TRAVIS_BUILD_DIR/pyNN/neuron/nmodl
    nrnivmodl

    popd

fi