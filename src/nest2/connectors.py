# ==============================================================================
# Connection method classes for nest
# $Id$
# ==============================================================================

from pyNN import common
from pyNN.nest2.__init__ import nest, WDManager, numpy
from pyNN.random import RandomDistribution, NativeRNG
from math import *

def get_target_ports(pre, target_list):
    # The connection dict returned by NEST contains a list of target ids,
    # so it is possible to obtain the target port by finding the index of
    # the target in this list. For now, we stick with saving the target port
    # in Python (faster, but more memory needed), but PyNEST should soon have
    # a function to do the lookup, at which point we will switch to using that.
    conn_dict = nest.GetConnections([pre], 'static_synapse')[0]
    if conn_dict:
        first_port = len(conn_dict['targets'])
    else:
        first_port = 0
    return range(first_port, first_port+len(target_list))

class AllToAllConnector(common.AllToAllConnector, WDManager):    

    def connect(self, projection):
        weight = self.getWeight(self.weights)
        weight = self.convertWeight(weight, projection.synapse_type)
        delay  = self.getDelay(self.delays)
        postsynaptic_neurons  = projection.post.cell.flatten()
        target_list = postsynaptic_neurons.tolist()
        for pre in projection.pre.cell.flat:
            # if self connections are not allowed, check whether pre and post are the same
            if not self.allow_self_connections:
                target_list = postsynaptic_neurons.tolist()
                if pre in target_list:
                    target_list.remove(pre)
            N = len(target_list)
            if isinstance(weight, RandomDistribution):
                weights = list(weight.next(N))
            else:
                weights = [weight]*N
            if isinstance(delay, RandomDistribution):
                delays = list(delay.next(N))
            else:
                delays = [float(delay)]*N
            projection._targets += target_list
            projection._sources += [pre]*N
            projection._targetPorts += get_target_ports(pre, target_list)
            nest.DivergentConnectWD([pre], target_list, weights, delays)
        return len(projection._targets)

class OneToOneConnector(common.OneToOneConnector, WDManager):
    
    def connect(self, projection):
        weight = self.getWeight(self.weights)
        weight = self.convertWeight(weight, projection.synapse_type)
        delay  = self.getDelay(self.delays)
        if projection.pre.dim == projection.post.dim:
            projection._sources = projection.pre.cell.flatten()
            projection._targets = projection.post.cell.flatten()
            N = len(projection._sources)
            projection._targetPorts = [get_target_ports(pre, [None])[0] for pre in projection._sources]
            if isinstance(weight, RandomDistribution):
                weights = list(weight.next(N))
            else:
                weights = [weight]*N
            if isinstance(delay, RandomDistribution):
                delays = list(delay.next(N))
            else:
                delays = [float(delay)]*N
            nest.ConnectWD(projection._sources, projection._targets, weights, delays)
            return projection.pre.size
        else:
            raise Exception("Connection method not yet implemented for the case where presynaptic and postsynaptic Populations have different sizes.")
    
class FixedProbabilityConnector(common.FixedProbabilityConnector, WDManager):
    
    def connect(self, projection):
        weight = self.getWeight(self.weights)
        weight = self.convertWeight(weight, projection.synapse_type)
        delay  = self.getDelay(self.delays)
        postsynaptic_neurons  = projection.post.cell.flatten()
        npost= projection.post.size
        for pre in projection.pre.cell.flat:
            if projection.rng:
                rarr = projection.rng.uniform(0,1,(npost,)) # what about NativeRNG?
            else:
                rarr = numpy.random.uniform(0,1,(npost,))
            target_list = numpy.compress(numpy.less(rarr,self.p_connect),postsynaptic_neurons).tolist()
            # if self connections are not allowed, check whether pre and post are the same
            if not self.allow_self_connections and pre in target_list:
                target_list.remove(pre)
            N=len(target_list)
            if isinstance(weight, RandomDistribution):
                weights = list(weight.next(N))
            else:
                weights = [weight]*N
            if isinstance(delay, RandomDistribution):
                delays = list(delay.next(N))
            else:
                delays = [float(delay)]*N
            projection._targets += target_list
            projection._sources += [pre]*N
            projection._targetPorts += get_target_ports(pre, target_list)
            nest.DivergentConnectWD([pre], target_list, weights, delays)
        return len(projection._sources)
    
class DistanceDependentProbabilityConnector(common.DistanceDependentProbabilityConnector, WDManager):
    
    def connect(self, projection):
        weight = self.getWeight(self.weights)
        weight = self.convertWeight(weight, projection.synapse_type)
        delay  = self.getDelay(self.delays)
        periodic_boundaries = self.periodic_boundaries
        if periodic_boundaries is not None:
            dimensions = projection.post.dim
            periodic_boundaries = numpy.concatenate((dimensions,numpy.zeros(3-len(dimensions))))
        postsynaptic_neurons = projection.post.cell.flatten() # array
        presynaptic_neurons  = projection.pre.cell.flat # iterator 
        # what about NativeRNG?
        if projection.rng:
            if isinstance(projection.rng, NativeRNG):
                print "Warning: use of NativeRNG not implemented. Using NumpyRNG"
                rarr = numpy.random.uniform(0,1,(projection.pre.size*projection.post.size,))
            else:
                rarr = projection.rng.uniform(0,1,(projection.pre.size*projection.post.size,))
        else:
            rarr = numpy.random.uniform(0,1,(projection.pre.size*projection.post.size,))
        j = 0
        idx_post = 0
        for pre in presynaptic_neurons:
            target_list = []
            idx_post = 0
            distances = common.distances(pre, projection.post, self.mask, self.scale_factor, self.offset, periodic_boundaries)
            for post in postsynaptic_neurons:
                if self.allow_self_connections or pre != post: 
                    # calculate the distance between the two cells :
                    d = distances[0][idx_post]
                    p = eval(self.d_expression)
                    if p >= 1 or (0 < p < 1 and rarr[j] < p):
                        target_list.append(post)
                        #projection._targets.append(post)
                        #projection._targetPorts.append(nest.connect(pre_addr,post_addr))
                        #nest.ConnectWD([pre],[post], [weight], [delay])
                j += 1
                idx_post += 1
            N = len(target_list)
            if isinstance(weight, RandomDistribution):
                weights = list(weight.next(N))
            else:
                weights = [weight]*N
            if isinstance(delay, RandomDistribution):
                delays = list(delay.next(N))
            else:
                delays = [float(delay)]*N
            projection._targets += target_list
            projection._sources += [pre]*N 
            projection._targetPorts += get_target_ports(pre, target_list)
            nest.DivergentConnectWD([pre], target_list, weights, delays)
        return len(projection._sources)


class FixedNumberPreConnector(common.FixedNumberPreConnector):
    
    def connect(self, projection):
        npost = projection.post.size
        postsynaptic_neurons  = projection.post.cell.flatten()
        if projection.rng:
            rng = projection.rng
        else:
            rng = numpy.random
        for pre in projection.pre.cell.flat:
            if hasattr(self, 'rand_distr'):
                n = self.rand_distr.next()
            else:
                n = self.n
            target_list = rng.permutation(postsynaptic_neurons)[0:n]
            # if self connections are not allowed, check whether pre and post are the same
            if not self.allow_self_connections and pre in target_list:
                target_list.remove(pre)

            N = len(target_list)
            weights = 1000.0*self.getWeights(N)
            if projection.synapse_type == 'inhibitory':
                weights *= -1
            delays = self.getDelays(N)

            nest.DivergentConnectWD([pre], target_list.tolist(), weights.tolist(), delays.tolist())

            projection._sources += [pre]*N
            conn_dict = nest.GetConnections([pre], 'static_synapse')[0]
            if isinstance(conn_dict, dict):
                all_targets = conn_dict['targets']
                total_targets = len(all_targets)
                projection._targets += all_targets[-N:]
                projection._targetPorts += range(total_targets-N,total_targets)
        return len(projection._sources)

def _n_connections(population):
    """
    Get a list of the total number of connections made by each neuron in a
    population.
    """
    n = numpy.zeros((len(population),),'int')
    conn_dict_list = nest.GetConnections([id for id in population],'static_synapse')
    for i, conn_dict in enumerate(conn_dict_list):
        assert isinstance(conn_dict, dict)
        n[i] = len(conn_dict['targets'])
    return n

class FixedNumberPostConnector(common.FixedNumberPostConnector):
    
    def connect(self, projection):
        npre = projection.pre.size
        presynaptic_neurons  = projection.pre.cell.flatten()
        if projection.rng:
            rng = projection.rng
        else:
            rng = numpy.random
        start_ports = _n_connections(projection.pre)
        for post in projection.post.cell.flat:
            if hasattr(self, 'rand_distr'):
                n = self.rand_distr.next()
            else:
                n = self.n
            source_list = rng.permutation(presynaptic_neurons)[0:n]
            # if self connections are not allowed, check whether pre and post are the same
            if not self.allow_self_connections and pre in source_list:
                source_list.remove(pre)

            N = len(source_list)
            weights = 1000.0*self.getWeights(N)
            if projection.synapse_type == 'inhibitory':
                weights *= -1
            delays = self.getDelays(N)

            nest.ConvergentConnectWD(source_list.tolist(), [post], weights.tolist(), delays.tolist())

        end_ports = _n_connections(projection.pre)
        for pre, start_port, end_port in zip(presynaptic_neurons, start_ports, end_ports):
            projection._targetPorts += range(start_port, end_port)
            projection._sources += [pre]*(end_port-start_port)
            conn_dict = nest.GetConnections([pre], 'static_synapse')[0]
            if isinstance(conn_dict, dict):
                projection._targets += conn_dict['targets'][start_port:end_port]
        print start_ports
        print end_ports
        return len(projection._sources)
