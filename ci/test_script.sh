#!/bin/bash

set -e  # stop execution in case of errors

if [ "$TRAVIS_PYTHON_VERSION" == "2.7" ] || [ "$TRAVIS_PYTHON_VERSION" == "3.6" ]; then
    nosetests --with-coverage --cover-package=pyNN -e backends -v test/unittests;
    nosetests --with-coverage --cover-package=pyNN -v test/system;
else
    nosetests -e backends test/unittests;
fi