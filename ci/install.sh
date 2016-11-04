#!/usr/bin/env bash

set -e  # stop execution in case of errors

sudo apt-get install -qq python-numpy python3-numpy python-scipy python3-scipy libgsl0-dev  openmpi-bin libopenmpi-dev

# need to install neo-0.5alpha1 from Github at this point
wget https://github.com/NeuralEnsemble/python-neo/archive/master.zip -O $HOME/neo-master.zip
pushd $HOME
unzip $HOME/neo-master.zip
pip install $HOME/python-neo-master
popd
pip install -r requirements.txt
pip install coverage coveralls
source ci/install_brian.sh
source ci/install_nest.sh
source ci/install_neuron.sh
python setup.py install
