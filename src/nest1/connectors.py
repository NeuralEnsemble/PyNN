# ==============================================================================
# Connection method classes for nest1
# $Id$
# ==============================================================================

from pyNN import common, connectors
from pyNN.nest1.__init__ import pynest, numpy, get_min_delay, get_max_delay
from pyNN.random import RandomDistribution, NativeRNG
from math import *
from numpy import arccos, arcsin, arctan, arctan2, ceil, cos, cosh, e, exp, \
                  fabs, floor, fmod, hypot, ldexp, log, log10, modf, pi, power, \
                  sin, sinh, sqrt, tan, tanh

common.get_min_delay = get_min_delay
common.get_max_delay = get_max_delay

def _convertWeight(w, synapse_type):
    weight = w*1000.0
    if isinstance(w, numpy.ndarray):
        all_negative = (weight<=0).all()
        all_positive = (weight>=0).all()
        assert all_negative or all_positive, "Weights must be either all positive or all negative"
        if synapse_type == 'inhibitory' and all_positive:
            weight *= -1
    elif is_number(weight):
        if synapse_type == 'inhibitory' and weight > 0:
            weight *= -1
    else:
        raise TypeError("we must be either a number or a numpy array")
    return weight


class AllToAllConnector(connectors.AllToAllConnector):    
    
    def connect(self, projection):
        postsynaptic_neurons = numpy.reshape(projection.post.cell,(projection.post.cell.size,))
        presynaptic_neurons  = numpy.reshape(projection.pre.cell,(projection.pre.cell.size,))
        for post in postsynaptic_neurons:
            source_list = presynaptic_neurons.tolist()
            # if self connections are not allowed, check whether pre and post are the same
            if not self.allow_self_connections and post in source_list:
                source_list.remove(post)
            N = len(source_list)
            weights = self.getWeights(N)
            weights = _convertWeight(weights, projection.synapse_type).tolist()
            delays = self.getDelays(N).tolist()
            projection._targets += [post]*N
            projection._sources += source_list
            projection._targetPorts +=  pynest.convergentConnect(source_list, post, weights, delays)
        return len(projection._targets)

class OneToOneConnector(connectors.OneToOneConnector):
    
    def connect(self, projection):
        if projection.pre.dim == projection.post.dim:
            projection._sources = numpy.reshape(projection.pre.cell,(projection.pre.cell.size,))
            projection._targets = numpy.reshape(projection.post.cell,(projection.post.cell.size,))
            N = len(projection._sources)
            weights = self.getWeights(N)
            weights = _convertWeight(weights, projection.synapse_type).tolist()
            delays  = self.getDelays(N).tolist()
            for pre, post, w, d in zip(projection._sources, projection._targets, weights, delays):
                projection._targetPorts.append(pynest.connectWD([pre], [post], w, d))
            return projection.pre.size
        else:
            raise Exception("Connection method not yet implemented for the case where presynaptic and postsynaptic Populations have different sizes.")
    
class FixedProbabilityConnector(connectors.FixedProbabilityConnector):
    
    def connect(self, projection):
        postsynaptic_neurons = numpy.reshape(projection.post.cell,(projection.post.cell.size,))
        presynaptic_neurons  = numpy.reshape(projection.pre.cell,(projection.pre.cell.size,))
        npre = projection.pre.size
        if projection.rng:
            if isinstance(projection.rng, NativeRNG):
                print "Warning: use of NativeRNG not implemented. Using NumpyRNG"
                rng = numpy.random
            else:
                rng = projection.rng
        else:
            rng = numpy.random
        for post in postsynaptic_neurons:
            rarr = rng.uniform(0, 1, npre) # what about NativeRNG?
            source_list = numpy.compress(numpy.less(rarr, self.p_connect), presynaptic_neurons).tolist()
            # if self connections are not allowed, check whether pre and post are the same
            if not self.allow_self_connections and post in source_list:
                source_list.remove(post)
            N = len(source_list)
            weights = self.getWeights(N)
            weights = _convertWeight(weights, projection.synapse_type).tolist()
            delays  = self.getDelays(N).tolist()
            projection._targets += [post]*N
            projection._sources += source_list
            projection._targetPorts += pynest.convergentConnect(source_list, post, weights, delays)
        return len(projection._sources)
    
class DistanceDependentProbabilityConnector(connectors.DistanceDependentProbabilityConnector):
    
    def connect(self, projection):
        periodic_boundaries = self.periodic_boundaries
        if periodic_boundaries is True:
            dimensions = projection.post.dim
            periodic_boundaries = tuple(numpy.concatenate((dimensions, numpy.zeros(3-len(dimensions)))))
        if periodic_boundaries:
            print "Periodic boundaries activated and set to size ", periodic_boundaries
        postsynaptic_neurons = numpy.reshape(projection.post.cell,(projection.post.cell.size,))
        presynaptic_neurons  = numpy.reshape(projection.pre.cell,(projection.pre.cell.size,))
        # what about NativeRNG?
	npre = len(presynaptic_neurons)
        if projection.rng:
            if isinstance(projection.rng, NativeRNG):
                print "Warning: use of NativeRNG not implemented. Using NumpyRNG"
                rng = numpy.random
            else:
                rng = projection.rng
        else:
            rng = numpy.random
        
        get_proba   = lambda d: eval(self.d_expression)
        get_weights = lambda d: eval(self.weights)
        get_delays  = lambda d: eval(self.delays)
            
        for post in postsynaptic_neurons:
            distances = common.distances(projection.pre, post, self.mask, self.scale_factor, self.offset, periodic_boundaries)[:,0]
            # We evaluate the probabilities of connections for those distances
            proba = get_proba(distances)
            rarr = rng.uniform(0, 1, (npre,))
            # We get the list of cells that will established a connecti
            idx = numpy.where((proba >= 1) | ((0 < proba) & (proba < 1) & (rarr <= proba)))[0]
            source_list = presynaptic_neurons[idx].tolist()
            # We remove the post cell if we don't allow self connections
            if not self.allow_self_connections and post in source_list:
                idx.remove(source_list.index(post))
                source_list.remove(post)
            N = len(source_list)
            if isinstance(self.weights,str):
                weights = get_weights(distances[idx])
            else:
                weights = self.getWeights(N)
            weights = _convertWeight(weights, projection.synapse_type).tolist()
            # We deal with the fact that the user could have given a delays distance dependent
            if isinstance(self.delays,str):
                delays = get_delays(distances[idx]).tolist()
            else:
                delays = self.getDelays(N).tolist()
            projection._targets += [post]*N
            projection._sources += source_list
            projection._targetPorts += pynest.convergentConnect(source_list, post, weights, delays)
        return len(projection._sources)
    

class FixedNumberPreConnector(connectors.FixedNumberPreConnector):
    
    def connect(self, projection):
        raise Exception("Not implemented yet !")

class FixedNumberPostConnector(connectors.FixedNumberPostConnector):
    
    def connect(self, projection):
        raise Exception("Not implemented yet !")
    

class FromListConnector(connectors.FromListConnector):
    
    def connect(self, projection):
        for i in xrange(len(self.conn_list)):
            src, tgt, weight, delay = self.conn_list[i][:]
            src = projection.pre[tuple(src)]
            tgt = projection.post[tuple(tgt)]
            projection._sources.append(src)
            projection._targets.append(tgt)
            projection._targetPorts.append(pynest.connectWD([src], [tgt], 1000*weight, delay))
            

class FromFileConnector(connectors.FromFileConnector):
    
    def connect(self, projection):
        # now open the file...
        f = open(self.filename,'r',10000)
        lines = f.readlines()
        f.close()
        # We read the file and gather all the data in a list of tuples (one per line)
        input_tuples = []
        for line in lines:
            single_line = line.rstrip()
            src, tgt, w, d = single_line.split("\t", 4)
            src = "[%s" % src.split("[",1)[1]
            tgt = "[%s" % tgt.split("[",1)[1]
            src, tgt, weight, delay = (eval(src), eval(tgt), float(w), float(d))
            src = projection.pre[tuple(src)]
            tgt = projection.post[tuple(tgt)]
            projection._sources.append(src)
            projection._targets.append(tgt)
            projection._targetPorts.append(pynest.connectWD([src], [tgt], 1000*weight, delay))