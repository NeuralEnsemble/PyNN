#!/bin/bash

set -e  # stop execution in case of errors

if [ "$TRAVIS_PYTHON_VERSION" == "3.9" ]; then
    echo -e "\n========== Installing NEURON ==========\n"
    export NRN_VERSION="nrn-8.0.0"

    if [ ! -f "$HOME/$NRN_VERSION/build/config.log" ]; then

        git clone https://github.com/neuronsimulator/nrn -b 8.0.0 $HOME/$NRN_VERSION

        mkdir -p $HOME/$NRN_VERSION/build
        pushd $HOME/$NRN_VERSION/build
        export VENV=`python -c "import sys; print(sys.prefix)"`

        cmake .. -DNRN_ENABLE_INTERVIEWS=OFF -DNRN_ENABLE_MPI=ON -DNRN_ENABLE_RX3D=OFF -DCMAKE_INSTALL_PREFIX=$VENV
        cmake --build . --target install
        ls
    else
        echo 'Using cached NEURON build directory.'
    fi

    pip install nrnutils  # must be installed after NEURON

    # compile PyNN NMODL mechanisms
    echo $TRAVIS_BUILD_DIR
    cd $TRAVIS_BUILD_DIR/pyNN/neuron/nmodl
    nrnivmodl

    popd

fi