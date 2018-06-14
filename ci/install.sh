#!/usr/bin/env bash

set -e  # stop execution in case of errors

source ci/install_brian.sh
source ci/install_nest.sh
source ci/install_neuron.sh

python setup.py install
