#!/bin/bash

set -e  # stop execution in case of errors

nosetests --with-coverage --cover-package=pyNN -e backends test/unittests
if [[ "$TRAVIS_PYTHON_VERSION" == "2.7" ]]; then
    nosetests --with-coverage --cover-package=pyNN -v test/system;
fi