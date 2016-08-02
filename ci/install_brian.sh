#!/bin/bash

set -e  # stop execution in case of errors

if [ "$TRAVIS_PYTHON_VERSION" == "2.7" ]; then
    echo -e "\n========== Installing Brian ==========\n"
    pip install scipy;
    pip install sympy;
    pip install brian;
fi
