#!/bin/bash

set -e  # stop execution in case of errors

if [ "$TRAVIS_PYTHON_VERSION" == "3.9" ]; then
    echo -e "\n========== Installing NEST ==========\n"
    # Specify which version of NEST to install
    #export NEST_VERSION="master"
    export NEST_VERSION="3.0"

    pip install cython

    if [ "$NEST_VERSION" = "master" ]; then
      export NEST="nest-simulator-$NEST_VERSION"
      wget https://github.com/nest/nest-simulator/archive/$NEST_VERSION.tar.gz -O $HOME/$NEST.tar.gz;
    else
      export NEST="nest-simulator-$NEST_VERSION"
      wget https://github.com/nest/nest-simulator/archive/v$NEST_VERSION.tar.gz -O $HOME/$NEST.tar.gz;
    fi

    pushd $HOME;
    tar xzf $NEST.tar.gz;
    ls;
    popd;

    mkdir -p $HOME/build/$NEST
    pushd $HOME/build/$NEST
    export VENV=`python -c "import sys; print(sys.prefix)"`;
    export PYLIB_GLOBAL=`find /opt/python/${TRAVIS_PYTHON_VERSION}/lib/ -name "libpython${TRAVIS_PYTHON_VERSION}*.so"`;
    ln -s ${PYLIB_GLOBAL} $VENV/lib/libpython${TRAVIS_PYTHON_VERSION}.so;
    export PYTHON_INCLUDE_DIR=$VENV/include/python${TRAVIS_PYTHON_VERSION}
    cython --version;
    cmake --version;
    cmake -DCMAKE_INSTALL_PREFIX=$VENV \
          -Dwith-mpi=ON  \
          -DPYTHON_EXECUTABLE=$VENV/bin/python \
          -DCYTHON_EXECUTABLE=$VENV/bin/cython \
          -DPYTHON_LIBRARY=$VENV/lib/libpython${TRAVIS_PYTHON_VERSION}.so \
          -DPYTHON_INCLUDE_DIR=${PYTHON_INCLUDE_DIR} \
          $HOME/$NEST;
    make;
    make install;
    popd;
    python -c "import nest";
fi
