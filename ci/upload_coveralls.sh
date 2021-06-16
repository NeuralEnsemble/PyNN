#!/bin/bash

set -e  # stop execution in case of errors

if [ "$PY_MINOR_VERSION" == "8" ]; then
    coveralls;
fi
