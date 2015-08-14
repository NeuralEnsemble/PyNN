#!/bin/bash

set -e  # stop execution in case of errors

if [[ "$TRAVIS_PYTHON_VERSION" == "2.7_with_system_site_packages" ]]; then
    pip install sympy;
    pip install brian;
fi
