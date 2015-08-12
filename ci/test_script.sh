#!/bin/bash

set -e  # stop execution in case of errors

nosetests -w test/unittests -c test/unittests/setup.cfg
nosetests -w test/system test_nest.py
