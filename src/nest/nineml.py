"""
Support cell types defined in 9ML with NEST.

Requires the 9ml nestbuilder script to be on the import path.

Classes:
    NineMLCellType   - base class for cell types, not used directly

Functions:
    nineml_cell_type_from_model - return a new NineMLCellType subclass

Constants:
    NEST_DIR        - subdirectory to which NEST mechanisms will be written (TODO: not implemented)

:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

from __future__ import absolute_import # Not compatible with Python 2.4
import subprocess
import neuron
from pyNN.models import BaseCellType
#from pyNN.nineml.cells import _build_nineml_celltype
from pyNN.nineml.cells import CoBaSyn 
import logging
import os
from itertools import chain
#from pyNN.neuron import simulator
#from pyNN import common, recording
from pyNN.nest.cells import NativeCellType
#common.simulator = simulator
#recording.simulator = simulator

logger = logging.getLogger("PyNN")

# TODO: This should go to a evironment variable, like PYNN_9ML_DIR
# and then a sub-dir for nest, neuron, etc.
# but default to ~/.pyNN or something to that regard.
NEST_DIR = "nest_models"

class NineMLCellType(NativeCellType):
    
    def __init__(self, parameters):
        NativeCellType.__init__(self, parameters)







def nineml_celltype_from_model(name, nineml_model, synapse_components):
    """
    Return a new NineMLCellType subclass from a NineML model.
    """
    
    dct = {'nineml_model':nineml_model,
           'synapse_components':synapse_components,
           'builder': _compile_nmodl} 
    return _nest_build_nineml_celltype(name, (NineMLCellType,), dct)



class _nest_build_nineml_celltype(type):
    """
    Metaclass for building NineMLCellType subclasses
    Called by nineml_celltype_from_model
    """
    def __new__(cls, name, bases, dct):
        
        
        #Extract Parameters Back out from Dict:
        nineml_model = dct['nineml_model']
        synapse_components = dct['synapse_components']

        # Reduce the model:                    
        from nineml.abstraction_layer import models               
        #reduction_process = models.ModelToSingleComponentReducer(nineml_model, componentname=name)
        #reduced_component = reduction_process.reducedcomponent
        reduced_component = models.reduce_to_single_component( nineml_model, componentname=name )

        # synapse ports:
        synapse_ports = []
        for syn in in synapse_components:
            pass
        # TODO: need to be able to get inferred event ports from model Component, i.e.
        # access to the true component below!
    
        
        # New:
        dct["combined_model"] = reduction_process.reducedcomponent
        dct["default_parameters"] = dict( (name, 1.0) for name in reduced_component.parameters )
        dct["default_initial_values"] = dict((name, 0.0) for name in reduced_component.state_variables)
        dct["synapse_types"] = [syn.namespace for syn in synapse_components] 
        dct["standard_receptor_type"] = (dct["synapse_types"] == ('excitatory', 'inhibitory'))
        dct["injectable"] = True # need to determine this. How??
        dct["conductance_based"] = True # how to determine this??
        dct["model_name"] = name
        
        
        # Recording from bindings:
        dct["recordable"] = [port.name for port in reduced_component.analog_ports] + ['spikes', 'regime'] + [binding.name for binding in reduced_component.bindings]
        
        dct["weight_variables"] = dict([ (syn.namespace,syn.namespace+'_'+syn.weight_connector )
                                         for syn in synapse_components ])
        #{'cobaInhib':'cobaInhib_q', 'cobaExcit':'cobaExcit_q',}
        
        
        logger.debug("Creating class '%s' with bases %s and dictionary %s" % (name, bases, dct))
        # generate and compile NMODL code, then load the mechanism into NEUORN
        dct["builder"](reduced_component, dct["weight_variables"], hierarchical_mode=True)
        # TODO: weight variables should really be stored within combined_model
        
        return type.__new__(cls, name, bases, dct)
        
