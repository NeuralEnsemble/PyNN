"""

Export of PyNN models to NeuroML 2

Contact Padraig Gleeson for more details

:copyright: Copyright 2006-2017 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

from itertools import repeat
try:
    from itertools import izip
except ImportError:
    izip = zip  # Python 3 zip returns an iterator already
from pyNN import common
from pyNN.core import ezip
from pyNN.space import Space
from . import simulator
import logging

import neuroml


logger = logging.getLogger("PyNN_NeuroML")


class Connection(common.Connection):
    """
    Store an individual plastic connection and information about it. Provide an
    interface that allows access to the connection's weight, delay and other
    attributes.
    """

    def __init__(self, pre, post, projection, conn_id, pre_pop_comp, post_pop_comp, **attributes):
        #logger.debug("Creating Connection: %s -> %s" % (pre, post))
        
        self.presynaptic_index = pre
        self.postsynaptic_index = post
        projection.connection_wds.append(neuroml.ConnectionWD(id=conn_id,pre_cell_id="../%s/%i/%s"%(projection.presynaptic_population,pre,pre_pop_comp),
                   post_cell_id="../%s/%i/%s"%(projection.postsynaptic_population,post,post_pop_comp), weight=attributes['WEIGHT'], delay='%sms'%attributes['DELAY']))
                   
        for name, value in attributes.items():
            setattr(self, name, value)

    def as_tuple(self, *attribute_names):
        # should return indices, not IDs for source and target
        return tuple([getattr(self, name) for name in attribute_names])


class Projection(common.Projection):
    __doc__ = common.Projection.__doc__
    _simulator = simulator
    
    def __init__(self, presynaptic_population, postsynaptic_population,
                 connector, synapse_type, source=None, receptor_type=None,
                 space=Space(), label=None):
        common.Projection.__init__(self, presynaptic_population, postsynaptic_population,
                                   connector, synapse_type, source, receptor_type,
                                   space, label)
                                   
        nml_doc = simulator._get_nml_doc()
        net = nml_doc.networks[0]
        
        nml_proj_id = self.label.replace(u'\u2192','__TO__')
        syn_id = 'syn__%s'%nml_proj_id
        
        logger.debug("Creating Synapse: %s; %s; %s" % (receptor_type, synapse_type.parameter_space, connector))
        celltype = postsynaptic_population.celltype.__class__.__name__
        logger.debug("Post cell: %s" % (celltype))
        syn = None
        
        if receptor_type == 'inhibitory':
            tau_key = 'tau_syn_I' 
            erev_key = 'e_rev_I'
        else:
            tau_key = 'tau_syn_E'
            erev_key = 'e_rev_E'
        
        if 'cond_exp' in celltype:
            syn = neuroml.ExpCondSynapse(id=syn_id)
            syn.__setattr__('e_rev', postsynaptic_population.celltype.parameter_space[erev_key].base_value)
            nml_doc.exp_cond_synapses.append(syn)
        if 'cond_alpha' in celltype:
            syn = neuroml.AlphaCondSynapse(id=syn_id)
            syn.__setattr__('e_rev', postsynaptic_population.celltype.parameter_space[erev_key].base_value)
            nml_doc.alpha_cond_synapses.append(syn)
        if 'curr_exp' in celltype:
            syn = neuroml.ExpCurrSynapse(id=syn_id)
            nml_doc.exp_curr_synapses.append(syn)
        if 'curr_alpha' in celltype:
            syn = neuroml.AlphaCurrSynapse(id=syn_id)
            nml_doc.alpha_curr_synapses.append(syn)
            
        syn.tau_syn = postsynaptic_population.celltype.parameter_space[tau_key].base_value
            
        self.pre_pop_comp = '%s_%s'%(presynaptic_population.celltype.__class__.__name__, presynaptic_population.label)
        self.post_pop_comp = '%s_%s'%(postsynaptic_population.celltype.__class__.__name__, postsynaptic_population.label)
        
        logger.debug("Creating Projection: %s" % (nml_proj_id))
        self.projection = neuroml.Projection(id=nml_proj_id, presynaptic_population=presynaptic_population.label, 
                        postsynaptic_population=postsynaptic_population.label, synapse=syn_id)
        net.projections.append(self.projection)


        ## Create connections
        self.connections = []
        connector.connect(self)

    def __len__(self):
        return len(self.connections)

    def set(self, **attributes):
        #parameter_space = ParameterSpace
        raise NotImplementedError

    def _convergent_connect(self, presynaptic_indices, postsynaptic_index,
                            **connection_parameters):
        for name, value in connection_parameters.items():
            if isinstance(value, float):
                connection_parameters[name] = repeat(value)
        for pre_idx, other in ezip(presynaptic_indices, *connection_parameters.values()):
            other_attributes = dict(zip(connection_parameters.keys(), other))
            self.connections.append(
                Connection(pre_idx, postsynaptic_index, self.projection, len(self.connections), self.pre_pop_comp, self.post_pop_comp, **other_attributes)
            )