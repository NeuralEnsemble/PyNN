# ==============================================================================
# Connection method classes for nest1
# $Id$
# ==============================================================================

from pyNN import common
from pyNN.nest1.__init__ import pynest, numpy
from pyNN.random import RandomDistribution, NativeRNG
from math import *

def _convertWeight(w, synapse_type):
    weight = w*1000.0
    if isinstance(w, numpy.ndarray):
        all_negative = (weight<=0).all()
        all_positive = (weight>=0).all()
        assert all_negative or all_positive, "Weights must be either all positive or all negative"
        if synapse_type == 'inhibitory':
            if all_positive:
                weight *= -1
    elif is_number(weight):
        if synapse_type == 'inhibitory' and weight > 0:
            weight *= -1
    else:
        raise TypeError("we must be either a number or a numpy array")
    return weight


class AllToAllConnector(common.AllToAllConnector):    
    
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

class OneToOneConnector(common.OneToOneConnector):
    
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
    
class FixedProbabilityConnector(common.FixedProbabilityConnector):
    
    def connect(self, projection):
        postsynaptic_neurons = numpy.reshape(projection.post.cell,(projection.post.cell.size,))
        presynaptic_neurons  = numpy.reshape(projection.pre.cell,(projection.pre.cell.size,))
        npre = projection.pre.size
        for post in postsynaptic_neurons:
            if projection.rng:
                rarr = projection.rng.uniform(0,1,(npre,)) # what about NativeRNG?
            else:
                rarr = numpy.random.uniform(0,1,(npre,))
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
    
class DistanceDependentProbabilityConnector(common.DistanceDependentProbabilityConnector):
    
    def connect(self, projection):
        periodic_boundaries = self.periodic_boundaries
        if periodic_boundaries is not None:
            if periodic_boundaries is True: 
                dimensions = projection.post.dim
            else:
                dimensions = [0,0,0]
            periodic_boundaries = numpy.concatenate((dimensions, numpy.zeros(3-len(dimensions))))
        postsynaptic_neurons = numpy.reshape(projection.post.cell,(projection.post.cell.size,))
        presynaptic_neurons  = numpy.reshape(projection.pre.cell,(projection.pre.cell.size,))
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
        for post in postsynaptic_neurons:
            source_list=[]
            idx_pre  = 0
            distances = common.distances(projection.pre, post, self.mask, self.scale_factor, self.offset, periodic_boundaries)
            for pre in presynaptic_neurons:
                if self.allow_self_connections or pre != post: 
                    # calculate the distance between the two cells :
                    d = distances[idx_pre][0]
                    p = eval(self.d_expression)
                    if p >= 1 or (0 < p < 1 and rarr[j] < p):
                        source_list.append(pre)
                        #projection._targets.append(post)
                        #projection._targetPorts.append(pynest.connectWD(pre_addr, post_addr, weight, delay)) 
                j += 1
                idx_pre += 1
            N = len(source_list)
            weights = self.getWeights(N)
            weights = _convertWeight(weights, projection.synapse_type).tolist()
            delays = self.getDelays(N).tolist()
            projection._targets += [post]*N
            projection._sources += source_list
            projection._targetPorts += pynest.convergentConnect(source_list, post, weights, delays)
        return len(projection._sources)
    

class FixedNumberPreConnector(common.FixedNumberPreConnector):
    
    def connect(self, projection):
        raise Exception("Not implemented yet !")

class FixedNumberPostConnector(common.FixedNumberPostConnector):
    
    def connect(self, projection):
        raise Exception("Not implemented yet !")
    

class FromListConnector(common.FromListConnector):
    
    def connect(self, projection):
        for i in xrange(len(self.conn_list)):
            src, tgt, weight, delay = self.conn_list[i][:]
            src = projection.pre[tuple(src)]
            tgt = projection.post[tuple(tgt)]
            projection._sources.append(src)
            projection._targets.append(tgt)
            projection._targetPorts.append(pynest.connectWD([src], [tgt], 1000*weight, delay))
            

class FromFileConnector(common.FromFileConnector):
    
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