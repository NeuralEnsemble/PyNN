#!/bin/bash

set -e  # stop execution in case of errors

if [[ "$TRAVIS_PYTHON_VERSION" == "2.7_with_system_site_packages" ]]; then
    nosetests --with-coverage --cover-package=pyNN -e backends test/unittests;
    nosetests --with-coverage --cover-package=pyNN -v test/system;
else
    nosetests -e backends test/unittests;
fi