"""
Support cell types defined in 9ML with NEURON.

Requires the 9ml2nmodl script to be on the path.

Classes:
    NineMLCell       - a single neuron instance
    NineMLCellType   - base class for cell types, not used directly

Functions:
    nineml_cell_type - return a new NineMLCellType subclass

Constants:
    NMODL_DIR        - subdirectory to which NMODL mechanisms will be written

:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

from __future__ import absolute_import # Not compatible with Python 2.4
import subprocess
import neuron

from pyNN.nineml.cells import join, _add_prefix, _build_nineml_celltype, NineMLCellType
import logging
import os

h = neuron.h
logger = logging.getLogger("PyNN")

NMODL_DIR = "nineml_mechanisms"

class NineMLCell(object):
    
    def __init__(self, **parameters):
        self.type = parameters.pop("type")
        self.source_section = h.Section()
        self.source = getattr(h, self.type.model_name)(0.5, sec=self.source_section)
        for param, value in parameters.items():
            setattr(self.source, param, value)
        # for recording
        self.spike_times = h.Vector(0)
        self.traces = {}
        self.recording_time = False
    
    def __getattr__(self, name):
        try:
            return self.__getattribute__(name)
        except AttributeError:
            if name in self.type.synapse_types:
                return self.source # source is also target
            else:
                raise AttributeError("'NineMLCell' object has no attribute or synapse type '%s'" % name)

    def record(self, active):
        if active:
            rec = h.NetCon(self.source, None)
            rec.record(self.spike_times)
        else:
            self.spike_times = h.Vector(0)

    def memb_init(self):
        # this is a bit of a hack
        for var in self.type.recordable:
            if hasattr(self, "%s_init" % var):
                initial_value = getattr(self, "%s_init" % var)
                logger.debug("Initialising %s to %g" % (var, initial_value))
                setattr(self.source, var, initial_value)


def _compile_nmodl(nineml_component, weight_variables): # weight variables should really be within component
    """
    Generate NMODL code for the 9ML component, run "nrnivmodl" and then load
    the mechanisms into NEURON.
    """
    if not os.path.exists(NMODL_DIR):
        os.makedirs(NMODL_DIR)
    cwd = os.getcwd()
    os.chdir(NMODL_DIR)
    xml_file = "%s.xml" % nineml_component.name
    logger.debug("Writing NineML component to %s" % xml_file)
    nineml_component.write(xml_file)
    nineml2nmodl = __import__("9ml2nmodl")
    nineml2nmodl.write_nmodl(xml_file, weight_variables) # weight variables should really come from xml file
    p = subprocess.check_call(["nrnivmodl"])
    os.chdir(cwd)
    neuron.load_mechanisms(NMODL_DIR)


def nineml_cell_type(name, neuron_model, port_map={}, weight_variables={}, **synapse_models):
    """
    Return a new NineMLCellType subclass.
    """
    return _build_nineml_celltype(name, (NineMLCellType,),
                                  {'neuron_model': neuron_model,
                                   'synapse_models': synapse_models,
                                   'port_map': port_map,
                                   'weight_variables': weight_variables})
