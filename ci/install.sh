#!/usr/bin/env bash

set -e  # stop execution in case of errors

pip install -r requirements.txt
pip install coverage coveralls
pip install nose-testconfig
source ci/install_brian.sh
source ci/install_nest.sh
source ci/install_neuron.sh
python setup.py install
