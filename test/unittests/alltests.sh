#/bin/bash
python-coverage -e

python-coverage -x commontests.py
python-coverage -x facetsmltest.py
python-coverage -x generictests.py nest2
python-coverage -x generictests.py neuron2
python-coverage -x generictests.py brian
#python-coverage -x generictests.py pcsim
#python-coverage -x nest1tests.py
#python-coverage -x nest2oldtests.py
python-coverage -x briantests.py
python-coverage -x nest2tests.py
python-coverage -x neuromltest.py
python-coverage -x neuron2tests.py
#python-coverage -x neurontests.py
#python-coverage -x pcsimtests_lowlevel.py
#python-coverage -x pcsimtests_population.py
python-coverage -x recordingtests.py
python-coverage -x rngtests.py
python-coverage -x utilitytests.py

python-coverage -r ~/dev/pyNN/*.py ~/dev/pyNN/nest2/*.py ~/dev/pyNN/neuron2/*.py ~/dev/pyNN/brian/*.py
python-coverage -a ~/dev/pyNN/*.py ~/dev/pyNN/nest2/*.py ~/dev/pyNN/neuron2/*.py ~/dev/pyNN/brian/*.py