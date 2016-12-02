#!/usr/bin/env bash

set -e  # stop execution in case of errors

sudo apt-get install -qq python-numpy python3-numpy python-scipy python3-scipy libgsl0-dev  openmpi-bin libopenmpi-dev

pip install -r requirements.txt
pip install coverage coveralls
source ci/install_brian.sh
source ci/install_nest.sh
source ci/install_neuron.sh
python setup.py install

#echo "Python 3.5 site packages"
#ls /home/travis/virtualenv/python3.5.2/lib/python3.5/site-packages
#echo "Python 2.7 site packages"
#ls /home/travis/virtualenv/python2.7.12/lib/python2.7/site-packages
