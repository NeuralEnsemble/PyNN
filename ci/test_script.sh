#!/bin/bash

set -e  # stop execution in case of errors

nosetests -v -w test/unittests -c test/unittests/setup.cfg
nosetests -v -w test/system test_nest.py
