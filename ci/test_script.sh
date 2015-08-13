#!/bin/bash

set -e  # stop execution in case of errors

nosetests --with-coverage --cover-package=pyNN -e backends test/unittests
nosetests --with-coverage --cover-package=pyNN -v test/system
