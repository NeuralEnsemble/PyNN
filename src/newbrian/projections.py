"""

"""

from itertools import repeat, izip
from collections import defaultdict
import numpy
import brian
from pyNN import common
from pyNN.core import ezip
from pyNN.parameters import ParameterSpace
from pyNN.space import Space
from . import simulator


class Connection(object):
    """
    Store an individual plastic connection and information about it. Provide an
    interface that allows access to the connection's weight, delay and other
    attributes.
    """

    def __init__(self, pre, post, **attributes):
        self.presynaptic_index = pre
        self.postsynaptic_index = post
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

        ## Create connections
        self.connections = None
        self._n_connections = 0
        if isinstance(self.pre, common.Assembly):
            presynaptic_populations = self.pre.populations
        else:
            presynaptic_populations = [self.pre]
        if isinstance(self.post, common.Assembly):
            postsynaptic_populations = self.post.populations
        else:
            postsynaptic_populations = [self.post]
        self._brian_synapses = defaultdict(dict)
        for i, pre in enumerate(presynaptic_populations):
            for j, post in enumerate(postsynaptic_populations):
                psv = post.celltype.post_synaptic_variables[self.receptor_type]
                pre_eqns = self.synapse_type.pre % psv
                if self.synapse_type.post:
                    post_eqns = self.synapse_type.post % psv
                else:
                    post_eqns = None
                syn_obj = brian.Synapses(pre.brian_group,
                                         post.brian_group,
                                         model=self.synapse_type.eqs,
                                         pre=pre_eqns,
                                         post=post_eqns)
                self._brian_synapses[i][j] = syn_obj
                simulator.state.network.add(syn_obj)
        connector.connect(self)

    def __len__(self):
        return self._n_connections

    def _partition(self, indices):
        """
        partition indices, in case of Assemblies
        """
        if isinstance(self.pre, common.Assembly):
            boundaries = numpy.cumsum([0] + [p.size for p in self.pre.populations])
            assert indices.max() < boundaries[-1]
            partitions = numpy.split(indices, numpy.searchsorted(indices, boundaries[1:-1])) - boundaries[:-1]
        else:
            partitions = [indices]
        return partitions
    
    def _localize_index(self, index):
        """determine which group the postsynaptic index belongs to """
        if isinstance(self.post, common.Assembly):
            boundaries = numpy.cumsum([0] + [p.size for p in self.post.populations])
            j = numpy.searchsorted(boundaries, index, side='right') - 1
            local_index = index - boundaries[j]
            return j, local_index
        else:
            return 0, index

    def _invert(self, indices, mask):
        """
        Given a set of indices into a PopulationView, return the indices into
        the (grand)-parent Population.
        """
        raise NotImplementedError("indices=%s, mask=%s") % (indices, mask)

    def _convergent_connect(self, presynaptic_indices, postsynaptic_index,
                            **connection_parameters):
        connection_parameters.pop("dendritic_delay_fraction", None)  # should try to handle this
        presynaptic_index_partitions = self._partition(presynaptic_indices)
        j, local_index = self._localize_index(postsynaptic_index)
        for i, partition in enumerate(presynaptic_index_partitions):
            if partition.size > 0:
                if isinstance(self.post, common.Assembly) and isinstance(self.post.populations[i], common.PopulationView):
                    partition = self._invert(partition, self.post.populations[i].mask)
                self._brian_synapses[i][j][partition, local_index] = True
        #print "CONNECTING", presynaptic_indices, postsynaptic_index, connection_parameters, presynaptic_index_partitions
        for name, value in connection_parameters.items():
            for i, partition in enumerate(presynaptic_index_partitions):
                #print i, partition, type(partition), bool(partition)
                if partition.size > 0:
                    brian_var = getattr(self._brian_synapses[i][j], name)
                    brian_var[partition, local_index] = value  # units? don't we need to slice value to the appropriate size?
                    #print "----", i, j, partition, local_index, name, value
                    self._n_connections += partition.size

    def _set_attributes(self, connection_parameters):
        connection_parameters.evaluate()
        pass  # to be completed
    
    def _get_attributes_as_arrays(self, *attribute_names):
        values = []
        for name in attribute_names:
            values.append(getattr(self._brian_synapses[0][0], name).to_matrix())  # temporary hack, will give wrong results with Assemblies
        return values  # should put NaN where there is no connection?
