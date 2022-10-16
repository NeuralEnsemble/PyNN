#!/bin/bash

set -e  # stop execution in case of errors

if [ "$TRAVIS_PYTHON_VERSION" == "3.9" ]; then
    pytest --verbose --cov=pyNN
else
    pytest --verbose test/unittests
fi

exit $?
