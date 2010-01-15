#/bin/bash
python-coverage -e

python-coverage -x commontests.py
python-coverage -x generictests.py nest
python-coverage -x generictests.py neuron
python-coverage -x generictests.py brian
python-coverage -x generictests.py pcsim
python-coverage -x recordingtests.py
python-coverage -x rngtests.py
python-coverage -x utilitytests.py

python-coverage -r ~/dev/pyNN/*.py ~/dev/pyNN/nest/*.py ~/dev/pyNN/neuron/*.py ~/dev/pyNN/brian/*.py ~/dev/pyNN/pcsim/*.py
python-coverage -a ~/dev/pyNN/*.py ~/dev/pyNN/nest/*.py ~/dev/pyNN/neuron/*.py ~/dev/pyNN/brian/*.py ~/dev/pyNN/pcsim/*.py