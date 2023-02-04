PyNN
====

PyNN (pronounced '*pine*') is a simulator-independent language for building
neuronal network models.

In other words, you can write the code for a model once, using the PyNN API and
the Python programming language, and then run it without modification on any
simulator that PyNN supports (currently NEURON, NEST and Brian 2) and
on a number of neuromorphic hardware systems.

The PyNN API aims to support modelling at a high-level of abstraction
(populations of neurons, layers, columns and the connections between them) while
still allowing access to the details of individual neurons and synapses when
required. PyNN provides a library of standard neuron, synapse and synaptic
plasticity models, which have been verified to work the same on the different
supported simulators. PyNN also provides a set of commonly-used connectivity
algorithms (e.g. all-to-all, random, distance-dependent, small-world) but makes
it easy to provide your own connectivity in a simulator-independent way.

Even if you don't wish to run simulations on multiple simulators, you may
benefit from writing your simulation code using PyNN's powerful, high-level
interface. In this case, you can use any neuron or synapse model supported by
your simulator, and are not restricted to the standard models.


- Home page: http://neuralensemble.org/PyNN/
- Documentation: http://neuralensemble.org/docs/PyNN/
- Mailing list: https://groups.google.com/forum/?fromgroups#!forum/neuralensemble
- Bug reports: https://github.com/NeuralEnsemble/PyNN/issues


:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

.. image:: https://travis-ci.org/NeuralEnsemble/PyNN.png?branch=master
   :target: https://travis-ci.org/NeuralEnsemble/PyNN
   :alt: Unit Test Status

.. image:: https://coveralls.io/repos/NeuralEnsemble/PyNN/badge.svg?branch=master&service=github
   :target: https://coveralls.io/github/NeuralEnsemble/PyNN?branch=master
   :alt: Test coverage
