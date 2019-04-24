#!/bin/bash

set -e  # stop execution in case of errors

if [ "$TRAVIS_PYTHON_VERSION" == "2.7" ] || [ "$TRAVIS_PYTHON_VERSION" == "3.6" ]; then
    python setup.py nosetests --with-coverage --cover-package=pyNN -e backends -v --tests=test/unittests;
    python setup.py nosetests --with-coverage --cover-package=pyNN -v --tests=test/system;
else
    python setup.py nosetests -e backends -v --tests=test/unittests;
fi
