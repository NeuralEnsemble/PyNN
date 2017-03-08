"""

Export of PyNN models to NeuroML 2

Contact Padraig Gleeson for more details

:copyright: Copyright 2006-2017 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

from pyNN import common
from pyNN import __version__

import logging

name = "NeuroML2Converter"

import neuroml
import pyneuroml
from pyneuroml.lems.LEMSSimulation import LEMSSimulation

nml_doc = None
lems_sim = None

logger = logging.getLogger("PyNN_NeuroML")

comment = "\n    This %s file has been generated from: \n" + \
          "        PyNN v%s\n"%__version__ + \
          "        libNeuroML v%s\n"%neuroml.__version__ + \
          "        pyNeuroML v%s\n    "%pyneuroml.__version__


def _get_nml_doc(reference="PyNN_NeuroML2_Export",reset=False):
    """Return the main NeuroMLDocument object being created"""
    global nml_doc
    global comment
    if nml_doc == None or reset:
        nml_doc = neuroml.NeuroMLDocument(id=reference)
        nml_doc.notes = comment%'NeuroML 2'
        
    return nml_doc


def _get_main_network():
    """Return the main NeuroML network object being created"""
    return _get_nml_doc().networks[0]

def _get_lems_sim(reference=None,reset=False):
    """Return the main LEMSSimulation object being created"""
    global lems_sim
    global comment
    if reference == None:
        reference = _get_nml_doc().id
    if lems_sim == None or reset:
        # Note: values will be over written
        lems_sim = LEMSSimulation("Sim_%s"%reference, 100, 0.01, target="network",comment=comment%'LEMS')
    return lems_sim

class ID(int, common.IDMixin):
    def __init__(self, n):
        """Create an ID object with numerical value `n`."""
        int.__init__(n)
        common.IDMixin.__init__(self)

class State(common.control.BaseState):
    def __init__(self):
        logger.debug("State initialised!")
        common.control.BaseState.__init__(self)
        self.mpi_rank = 0
        self.num_processes = 1
        self.clear()
        self.dt = 0.1
    def run(self, simtime):
        self.t += simtime
        self.running = True
    def run_until(self, tstop):
        logger.debug("run_until() called with %s"%tstop)
        lems_sim = _get_lems_sim()
        lems_sim.duration = float(tstop)
        
        self.t = tstop
        self.running = True
    def clear(self):
        self.recorders = set([])
        self.id_counter = 42
        self.segment_counter = -1
        self.reset()
    def reset(self):
        """Reset the state of the current network to time t = 0."""
        self.running = False
        self.t = 0
        self.t_start = 0
        self.segment_counter += 1

state = State()
