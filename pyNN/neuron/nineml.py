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

:copyright: Copyright 2006-2016 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

from __future__ import absolute_import
import logging
import os
import neuron
from pyNN.models import BaseCellType
from pyNN.nineml.cells import build_nineml_celltype
from nineml2nmodl import write_nmodl, write_nmodldirect, call_nrnivmodl


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
        self.rec = h.NetCon(self.source, None)
        self.spike_times = h.Vector(0)
        self.traces = {}
        self.recording_time = False

    def __getattr__(self, name):
        try:
            return self.__getattribute__(name)
        except AttributeError:
            if name in self.type.receptor_types:
                return self.source  # source is also target
            else:
                raise AttributeError("'NineMLCell' object has no attribute or receptor type '%s'" % name)

    def memb_init(self):
        # this is a bit of a hack
        for var in self.type.recordable:
            if hasattr(self, "%s_init" % var):
                initial_value = getattr(self, "%s_init" % var)
                logger.debug("Initialising %s to %g" % (var, initial_value))
                setattr(self.source, var, initial_value)


class NineMLCellType(BaseCellType):
    model = NineMLCell

    def __init__(self, **parameters):
        BaseCellType.__init__(self, **parameters)
        self.extra_parameters = {"type": self}  # self.__class__?


def _compile_nmodl(nineml_component, weight_variables, hierarchical_mode=None):  # weight variables should really be within component
    """
    Generate NMODL code for the 9ML component, run "nrnivmodl" and then load
    the mechanisms into NEURON.
    """
    wdir = os.path.join(NMODL_DIR, nineml_component.name)
    if not os.path.exists(wdir):
        os.makedirs(wdir)
    cwd = os.getcwd()
    os.chdir(wdir)

    xml_file = "%s.xml" % nineml_component.name
    logger.debug("Writing NineML component to %s" % xml_file)
    nineml_component.write(xml_file)
    mod_filename = nineml_component.name + ".mod"
    write_nmodldirect(component=nineml_component, mod_filename=mod_filename, weight_variables=weight_variables)
    #write_nmodl(xml_file, weight_variables) # weight variables should really come from xml file

    print("Running 'nrnivmodl' from %s" % wdir)
    call_nrnivmodl()
    os.chdir(cwd)
    neuron.load_mechanisms(wdir)


def nineml_cell_type(name, combined_model, weight_vars):
    """
    Return a new NineMLCellType subclass.
    """
    return build_nineml_celltype(name, (NineMLCellType,),
                                 {'combined_model': combined_model,
                                  'weight_variables': weight_vars,
                                  'builder': _compile_nmodl})

