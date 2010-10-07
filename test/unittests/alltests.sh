#/bin/bash
coverage erase

coverage run commontests.py
coverage run -a generictests.py nest
coverage run -a generictests.py neuron
coverage run -a generictests.py brian
coverage run -a generictests.py pcsim
coverage run -a rngtests.py
coverage run -a utilitytests.py
coverage run -a multisimtests.py

coverage report ~/dev/pyNN_trunk/src/*.py ~/dev/pyNN_trunk/src/*/*.py
coverage html ~/dev/pyNN_trunk/src/*.py ~/dev/pyNN_trunk/src/*/*.py