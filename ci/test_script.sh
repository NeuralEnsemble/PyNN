#!/bin/bash

set -e  # stop execution in case of errors

if [ "$PY_MINOR_VERSION" == "8" ]; then
    python setup.py nosetests --with-coverage --cover-package=pyNN -v --tests=test;
else
    python setup.py nosetests -e backends -v --tests=test/unittests;
fi
