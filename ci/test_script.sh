#!/bin/bash

set -e  # stop execution in case of errors

if [ "$TRAVIS_PYTHON_VERSION" == "3.8" ]; then
    python setup.py nosetests --nologcapture --with-coverage --cover-package=pyNN -v --tests=test;
else
    python setup.py nosetests --nologcapture -e backends -v --tests=test/unittests;
fi
