#!/bin/bash

set -e  # stop execution in case of errors

if [ "$TRAVIS_PYTHON_VERSION" == "3.9" ]; then
    python setup.py nosetests --verbose --nologcapture --with-coverage --cover-package=pyNN --tests=test;
else
    python setup.py nosetests --verbose --nologcapture -e backends --tests=test/unittests
fi
