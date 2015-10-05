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
            print("nrnivmodl found at", nrnivmodl)
            import subprocess
            p = subprocess.Popen(nrnivmodl, shell=True, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         close_fds=True, cwd=os.path.join(os.getcwd(), self.build_lib, 'pyNN/neuron/nmodl'))
            stdout = p.stdout.readlines()
            result = p.wait()
            # test if nrnivmodl was successful
            if result != 0:
                print("Unable to compile NEURON extensions. Output was:")
                print('  '.join([''] + stdout)) # indent error msg for easy comprehension
            else:
                print("Successfully compiled NEURON extensions.")
        else:
            print("Unable to find nrnivmodl. It will not be possible to use the pyNN.neuron module.")
        
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
    version = "0.8.0",
    packages = ['pyNN','pyNN.nest', 'pyNN.neuron',
                'pyNN.brian', 'pyNN.common', 'pyNN.mock',
                'pyNN.recording', 'pyNN.standardmodels', 'pyNN.descriptions',
                'pyNN.nest.standardmodels',
                'pyNN.neuron.standardmodels', 'pyNN.brian.standardmodels',
                'pyNN.utility'],
    package_data = {'pyNN': ['neuron/nmodl/*.mod', "descriptions/templates/*/*"]},
    author = "The PyNN team",
    author_email = "andrew.davison@unic.cnrs-gif.fr",
    description = "A Python package for simulator-independent specification of neuronal network models",
    long_description = open("README.rst").read(),
    license = "CeCILL http://www.cecill.info",
    keywords = "computational neuroscience simulation neuron nest brian neuromorphic",
    url = "http://neuralensemble.org/PyNN/",
    classifiers = ['Development Status :: 4 - Beta',
                   'Environment :: Console',
                   'Intended Audience :: Science/Research',
                   'License :: Other/Proprietary License',
                   'Natural Language :: English',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 2.6',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.3',
                   'Programming Language :: Python :: 3.4',
                   'Programming Language :: Python :: 3.5',
                   'Topic :: Scientific/Engineering'],
    cmdclass = {'build': build},
)

