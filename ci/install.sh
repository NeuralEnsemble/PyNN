#!/usr/bin/env bash

set -e  # stop execution in case of errors

pip install -r requirements.txt
pip install coverage coveralls
pip install nose-testconfig
export PY_MAJOR_VERSION=`python3 -c "import sys; print(sys.version_info.major)"`
export PY_MINOR_VERSION=`python3 -c "import sys; print(sys.version_info.minor)"`
source ci/install_brian.sh
source ci/install_nest.sh
source ci/install_neuron.sh
python setup.py install
