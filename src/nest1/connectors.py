# ==============================================================================
# Connection method classes for nest1
# $Id$
# ==============================================================================

from pyNN import common
from pyNN.nest1.__init__ import pynest, WDManager, _min_delay, numpy
from pyNN.random import RandomDistribution, NativeRNG
from math import *


class AllToAllConnector(common.AllToAllConnector, WDManager):    
    
    def connect(self, projection):
        weight = self.getWeight(self.weights)
        weight = self.convertWeight(weight, projection.synapse_type)
        delay  = self.getDelay(self.delays)
        postsynaptic_neurons = numpy.reshape(projection.post.cell,(projection.post.cell.size,))
        presynaptic_neurons  = numpy.reshape(projection.pre.cell,(projection.pre.cell.size,))
        for post in postsynaptic_neurons:
            source_list = presynaptic_neurons.tolist()
            # if self connections are not allowed, check whether pre and post are the same
            if not self.allow_self_connections and post in source_list:
                source_list.remove(post)
            N = len(source_list)
            if isinstance(weight, RandomDistribution):
                weights = list(weight.next(N))
            else:
                weights = [weight]*N
            if isinstance(delay, RandomDistribution):
                delays = list(delay.next(N))
            else:
                delays = [float(delay)]*N
            projection._targets += [post]*N
            projection._sources += source_list
            projection._targetPorts +=  pynest.convergentConnect(source_list,post,weights,delays)
        return len(projection._targets)

class OneToOneConnector(common.OneToOneConnector, WDManager):
    
    def connect(self, projection):
        weight = self.getWeight(self.weights)
        weight = self.convertWeight(weight, projection.synapse_type)
        delay  = self.getDelay(self.delays)
        if projection.pre.dim == projection.post.dim:
            projection._sources = numpy.reshape(projection.pre.cell,(projection.pre.cell.size,))
            projection._targets = numpy.reshape(projection.post.cell,(projection.post.cell.size,))
            N = len(projection._sources)
            if isinstance(weight, RandomDistribution):
                weights = list(weight.next(N))
            else:
                weights = [weight]*N
            if isinstance(delay, RandomDistribution):
                delays = list(delay.next(N))
            else:
                delays = [float(delay)]*N
            for pre,post,w,d in zip(projection._sources,projection._targets,weights, delays):
                pre_addr = pynest.getAddress(pre)
                post_addr = pynest.getAddress(post)
                projection._targetPorts.append(pynest.connectWD(pre_addr,post_addr,w,d))
            return projection.pre.size
        else:
            raise Exception("Connection method not yet implemented for the case where presynaptic and postsynaptic Populations have different sizes.")
    
class FixedProbabilityConnector(common.FixedProbabilityConnector, WDManager):
    
    def connect(self, projection):
        weight = self.getWeight(self.weights)
        weight = self.convertWeight(weight, projection.synapse_type)
        delay  = self.getDelay(self.delays)
        postsynaptic_neurons = numpy.reshape(projection.post.cell,(projection.post.cell.size,))
        presynaptic_neurons  = numpy.reshape(projection.pre.cell,(projection.pre.cell.size,))
        npre = projection.pre.size
        for post in postsynaptic_neurons:
            if projection.rng:
                rarr = projection.rng.uniform(0,1,(npre,)) # what about NativeRNG?
            else:
                rarr = numpy.random.uniform(0,1,(npre,))
            source_list = numpy.compress(numpy.less(rarr,self.p_connect),presynaptic_neurons).tolist()
            # if self connections are not allowed, check whether pre and post are the same
            if not self.allow_self_connections and post in source_list:
                source_list.remove(post)
            N = len(source_list)
            if isinstance(weight, RandomDistribution):
                weights = list(weight.next(N))
            else:
                weights = [weight]*N
            if isinstance(delay, RandomDistribution):
                delays = list(delay.next(N))
            else:
                delays = [float(delay)]*N
            projection._targets += [post]*N
            projection._sources += source_list
            projection._targetPorts += pynest.convergentConnect(source_list,post,weights,delays)
        return len(projection._sources)
    
class DistanceDependentProbabilityConnector(common.DistanceDependentProbabilityConnector, WDManager):
    
    def connect(self, projection):
        weight = self.getWeight(self.weights)
        weight = self.convertWeight(weight, projection.synapse_type)
        delay  = self.getDelay(self.delays)
        periodic_boundaries = self.periodic_boundaries
        if periodic_boundaries is not None:
            if periodic_boundaries is True: 
                dimensions = projection.post.dim
            else:
                dimensions = periodic_boundaries
            periodic_boundaries = numpy.concatenate((dimensions,numpy.zeros(3-len(dimensions))))
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
                    # calculate the addresses of cells
                    #pre_addr  = pynest.getAddress(pre)
                    #post_addr = pynest.getAddress(post)
                    if p >= 1 or (0 < p < 1 and rarr[j] < p):
                        source_list.append(pre)
                        #projection._targets.append(post)
                        #projection._targetPorts.append(pynest.connectWD(pre_addr,post_addr,weight,delay)) 
                j += 1
                idx_pre += 1
            N = len(source_list)
            if isinstance(weight, RandomDistribution):
                weights = list(weight.next(N))
            else:
                weights = [weight]*N
            if isinstance(delay, RandomDistribution):
                delays = list(delay.next(N))
            else:
                delays = [float(delay)]*N
            projection._targets += [post]*N
            projection._sources += source_list
            projection._targetPorts += pynest.convergentConnect(source_list,post,weights,delays)
        return len(projection._sources)


class FixedNumberPreConnector(common.FixedNumberPreConnector):
    
    def connect(self, projection):
        raise Exception("Not implemented yet !")


class FixedNumberPostConnector(common.FixedNumberPostConnector):
    
    def connect(self, projection):
        raise Exception("Not implemented yet !")