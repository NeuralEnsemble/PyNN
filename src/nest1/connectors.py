# ==============================================================================
# Connection method classes for nest1
# $Id$
# ==============================================================================

from pyNN import common
from pyNN.nest1.__init__ import pynest, _min_delay, numpy
from pyNN.random import RandomDistribution, NativeRNG
from math import *

class WDManager(object):
    
    def getWeight(self, w=None):
        if w is not None:
            weight = w
        else:
            weight = 1.
        return weight
        
    def getDelay(self, d=None):
        if d is not None:
            delay = d
        else:
            delay = _min_delay
        return delay
    
    def convertWeight(self, w, synapse_type):
        weight = w*1000.0

        if synapse_type == 'inhibitory':
            # We have to deal with the distribution, and anticipate the
            # fact that we will need to multiply by a factor 1000 the weights
            # in nest...
            if isinstance(weight, RandomDistribution):
                if weight.name == "uniform":
                    print weight.name, weight.parameters
                    (w_min,w_max) = weight.parameters
                    if w_min >= 0 and w_max >= 0:
                        weight.parameters = (-w_max, -w_min)
                elif weight.name ==  "normal":
                    (w_mean,w_std) = weight.parameters
                    if w_mean > 0:
                        weight.parameters = (-w_mean, w_std)
                else:
                    print "WARNING: no conversion of the weights for this particular distribution"
            elif weight > 0:
                weight *= -1
        return weight

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
                dimensions = [0,0,0]
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
    

class FromListConnector(common.FromListConnector):
    
    def connect(self, projection):
        for i in xrange(len(self.conn_list)):
            src, tgt, weight, delay = self.conn_list[i][:]
            src = projection.pre[tuple(src)]
            tgt = projection.post[tuple(tgt)]
            pre_addr = pynest.getAddress(src)
            post_addr = pynest.getAddress(tgt)
            projection._sources.append(src)
            projection._targets.append(tgt)
            projection._targetPorts.append(pynest.connectWD(pre_addr,post_addr, 1000*weight, delay))
            

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
            src, tgt, weight, delay = (eval(src),eval(tgt),float(w),float(d))
            src = projection.pre[tuple(src)]
            tgt = projection.post[tuple(tgt)]
            pre_addr = pynest.getAddress(src)
            post_addr = pynest.getAddress(tgt)
            projection._sources.append(src)
            projection._targets.append(tgt)
            projection._targetPorts.append(pynest.connectWD(pre_addr,post_addr, 1000*weight, delay))