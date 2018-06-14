#!/usr/bin/env bash

set -e  # stop execution in case of errors

./ci/install_brian.sh
./ci/install_nest.sh
./ci/install_neuron.sh

python setup.py install
