#!/bin/bash

set -e  # stop execution in case of errors

if [ "$TRAVIS_PYTHON_VERSION" == "3.8" ]; then
    echo -e "\n========== Installing Brian 2 ==========\n"
    pip install cython;
    pip install brian2;
fi
