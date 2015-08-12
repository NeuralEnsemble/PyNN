#!/bin/bash

set -e  # stop execution in case of errors

pushd test/unittests
#nosetests -v -w test/unittests --with-coverage --cover-package=pyNN -c test/unittests/setup.cfg
nosetests -v --with-coverage --cover-package=pyNN -c setup.cfg
popd
pushd test/system
#nosetests -v -w test/system --with-coverage --cover-package=pyNN test_nest.py
nosetests -v --with-coverage --cover-package=pyNN test_nest.py
popd
