#!/usr/bin/env python

from distutils.core import setup
from distutils.command.build import build as _build
import os

class build(_build):
    """Add nrnivmodl to the end of the build process."""

    def run(self):
        _build.run(self)
        nrnivmodl = self.find_nrnivmodl()
        if nrnivmodl:
            print "nrnivmodl found at", nrnivmodl
            import subprocess
            p = subprocess.Popen(nrnivmodl, shell=True, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         close_fds=True, cwd=os.path.join(os.getcwd(), self.build_lib, 'pyNN/neuron/nmodl'))
            stdout = p.stdout.readlines()
            result = p.wait()
            # test if nrnivmodl was successful
            if result != 0:
                print "Unable to compile NEURON extensions. Output was:"
                print '  '.join([''] + stdout) # indent error msg for easy comprehension
            else:
                print "Successfully compiled NEURON extensions."
        else:
            print "Unable to find nrnivmodl. It will not be possible to use the pyNN.neuron module."
        
    def find_nrnivmodl(self):
        """Try to find the nrnivmodl executable."""
        path = os.environ.get("PATH", "").split(os.pathsep)
        nrnivmodl = ''
        for dir_name in path:
            abs_name = os.path.abspath(os.path.normpath(os.path.join(dir_name, "nrnivmodl")))
            if os.path.isfile(abs_name):
                nrnivmodl = abs_name
                break
        return nrnivmodl
      
setup(
    name = "PyNN",
    version = "0.8.0dev",
    package_dir={'pyNN': 'src'},
    packages = ['pyNN','pyNN.nest', 'pyNN.pcsim', 'pyNN.neuron', 'pyNN.nineml',
                'pyNN.brian','pyNN.nemo', 'pyNN.common',
                'pyNN.recording', 'pyNN.standardmodels', 'pyNN.descriptions',
                'pyNN.nest.standardmodels', 'pyNN.pcsim.standardmodels',
                'pyNN.neuron.standardmodels', 'pyNN.brian.standardmodels', 'pyNN.nemo.standardmodels'],
    package_data = {'pyNN': ['neuron/nmodl/*.mod', "descriptions/templates/*/*"]},
    author = "The PyNN team",
    author_email = "pynn@neuralensemble.org",
    description = "A Python package for simulator-independent specification of neuronal network models",
        long_description = """In other words, you can write the code for a model once, using the PyNN API and the Python programming language, and then run it without modification on any simulator that PyNN supports (currently NEURON, NEST, PCSIM and Brian).

The API has two parts, a low-level, procedural API (functions ``create()``, ``connect()``, ``set()``, ``record()``, ``record_v()``), and a high-level, object-oriented API (classes ``Population`` and ``Projection``, which have methods like ``set()``, ``record()``, ``setWeights()``, etc.). 

The low-level API is good for small networks, and perhaps gives more flexibility. The high-level API is good for hiding the details and the book-keeping, allowing you to concentrate on the overall structure of your model.

The other thing that is required to write a model once and run it on multiple simulators is standard cell and synapse models. PyNN translates standard cell-model names and parameter names into simulator-specific names, e.g. standard model ``IF_curr_alpha`` is ``iaf_neuron`` in NEST and ``StandardIF`` in NEURON, while ``SpikeSourcePoisson`` is a ``poisson_generator`` in NEST and a ``NetStim`` in NEURON.

Even if you don't wish to run simulations on multiple simulators, you may benefit from writing your simulation code using PyNN's powerful, high-level interface. In this case, you can use any neuron or synapse model supported by your simulator, and are not restricted to the standard models.

PyNN is a work in progress, but is already being used for several large-scale simulation projects.""",
    license = "CeCILL http://www.cecill.info",
    keywords = "computational neuroscience simulation neuron nest pcsim brian neuroml",
    url = "http://neuralensemble.org/PyNN/",
    classifiers = ['Development Status :: 4 - Beta',
                   'Environment :: Console',
                   'Intended Audience :: Science/Research',
                   'License :: Other/Proprietary License',
                   'Natural Language :: English',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering'],
    cmdclass = {'build': build},
)

