#!/usr/bin/env python

from distutils.core import setup

setup(
    name = "PyNN",
    version = "0.4.0alpha",
    package_dir={'pyNN': 'src'},
    packages = ['pyNN','pyNN.nest2','pyNN.pcsim','pyNN.neuron'],
    author = "The NeuralEnsemble Community",
    author_email = "pynn@neuralensemble.org",
    description = "A Python package for simulator-independent specification of neuronal network models",
    license = "CeCILL",
    keywords = "computational neuroscience simulation neuron nest pcsim neuroml",
    url = "http://neuralensemble.org/PyNN/",
     )
