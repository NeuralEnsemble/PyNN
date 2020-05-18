#!/bin/bash

set -e  # stop execution in case of errors

if [ "$TRAVIS_PYTHON_VERSION" == "2.7" ] || [ "$TRAVIS_PYTHON_VERSION" == "3.7" ]; then
    echo -e "\n========== Installing NEST ==========\n"
    # Specify which version of NEST to install
    #export NEST_VERSION="master"
    export NEST_VERSION="2.20.0"

    pip install cython  #==0.28.1

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
    if [ "$TRAVIS_PYTHON_VERSION" == "2.7" ]; then
      ln -s /opt/python/2.7/lib/libpython2.7.so $VENV/lib/libpython2.7.so;
      export PYTHON_INCLUDE_DIR=$VENV/include/python${TRAVIS_PYTHON_VERSION}
    fi
    if [ "$TRAVIS_PYTHON_VERSION" == "3.7" ]; then
      ln -s /opt/python/3.7/lib/libpython3.7m.so $VENV/lib/libpython3.7.so;
      export PYTHON_INCLUDE_DIR=$VENV/include/python${TRAVIS_PYTHON_VERSION}m
    fi
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
