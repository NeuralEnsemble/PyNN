#!/bin/bash

set -e  # stop execution in case of errors

if [ "$PY_MINOR_VERSION" == "8" ]; then
    echo -e "\n========== Installing Brian 2 ==========\n"
    pip install cython;
    pip install brian2;
fi
