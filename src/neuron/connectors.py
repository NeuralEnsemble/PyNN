# ==============================================================================
# Connection method classes for neuron
# $Id$
# ==============================================================================

from pyNN import common
from pyNN.random import RandomDistribution, NativeRNG
from pyNN.neuron.__init__ import hoc_execute, h, get_min_delay, num_processes, get_rank
import numpy
from math import *


# ==============================================================================
#   Connection method classes
# ==============================================================================

class HocConnector(object):
    
    def singleConnect(self, projection, src, tgt, weight, delay):
        """
        Write hoc commands to connect a single pair of neurons.
        """
        cmdlist = ['nc = pc.gid_connect(%d,%s.object(%d).%s)' % (src,
                                                                 projection.post.hoc_label,
                                                                 projection.post.gidlist.index(tgt),
                                                                 projection._syn_objref),
                'nc.weight = %f' % weight,
                'nc.delay  = %f' % delay,
                'tmp = %s.append(nc)' % projection.hoc_label]
        projection.connections.append((src, tgt))
        return cmdlist
    
    def getWeight(self, w):
        if w is not None: 
            weight = w
        else: 
            weight = 1.
        return weight
    
    def getDelay(self, d):
        if d is not None:
            delay = max((d, get_min_delay()))
        else:
            delay = get_min_delay()
        return delay

    def _process_conn_list(self, conn_list, projection):
        """Extract fields from list of tuples and construct the hoc commands."""
        hoc_commands = []
        for i in xrange(len(conn_list)):
            src, tgt, weight, delay = conn_list[i][:]
            src = projection.pre[tuple(src)]
            tgt = projection.post[tuple(tgt)]
            hoc_commands += self.singleConnect(projection, src, tgt, weight, delay)
        return hoc_commands


class AllToAllConnector(common.AllToAllConnector, HocConnector):    
    
    def connect(self, projection):
        weight = self.getWeight(self.weights)
        delay = self.getDelay(self.delays)
        hoc_commands = []
        for tgt in projection.post.gidlist:
            for src in projection.pre.fullgidlist:
                if self.allow_self_connections or projection.pre != projection.post or tgt != src:
                    if isinstance(weight, RandomDistribution):
                        w = weight.next()
                    else:
                        w = weight
                    if isinstance(delay, RandomDistribution):
                        d = delay.next()
                    else:
                        d = delay
                    #print "Setting connection delay from %s to %s in %s to %g" % (src, tgt, projection.label, d)
                    hoc_commands += self.singleConnect(projection, src, tgt, w, d)
        return hoc_commands


class OneToOneConnector(common.OneToOneConnector, HocConnector):
    
    def connect(self, projection):
        weight = self.getWeight(self.weights)
        delay = self.getDelay(self.delays)
        if projection.pre.dim == projection.post.dim:
            hoc_commands = []
            for tgt in projection.post.gidlist:
                src = tgt - projection.post.gid_start + projection.pre.gid_start
                if isinstance(weight, RandomDistribution): w = weight.next()
                else: w = weight
                if isinstance(delay, RandomDistribution): d = delay.next()
                else: d = delay
                hoc_commands += self.singleConnect(projection, src, tgt, w, d)
        else:
            raise Exception("Method '%s' not yet implemented for the case where presynaptic and postsynaptic Populations have different sizes." % sys._getframe().f_code.co_name)
        return hoc_commands


class FixedProbabilityConnector(common.FixedProbabilityConnector, HocConnector):
    
    def connect(self, projection):
        weight = self.getWeight(self.weights)
        delay = self.getDelay(self.delays)
        if isinstance(projection.rng, NativeRNG):
            hoc_commands = ['rng = new Random(%d)' % 0 or distribution.rng.seed,
                            'tmp = rng.uniform(0,1)']
            # Here we are forced to execute the commands on line to be able to
            # catch the connections from NEURON
            hoc_execute(hoc_commands)
            #rarr = [HocToPy.get('rng.repick()','float') for j in xrange(projection.pre.size*projection.post.size)]
            rarr = [h.rng.repick() for j in xrange(projection.pre.size*projection.post.size)]
        else:
            # We use concatenate, rather than just creating
            # n=projection.pre.size*projection.post.size random numbers,
            # in case of uneven distribution of neurons between MPI nodes
            rarr = numpy.concatenate([projection.rng.next(projection.post.size, 'uniform', (0,1)) \
                                      for src in projection.pre.fullgidlist])
        hoc_commands = []
        j = 0
        required_rarr_length = len(projection.pre.fullgidlist) * len(projection.post.gidlist)
        assert len(rarr) >= required_rarr_length, \
               "Random array is too short (%d elements, needs %d)" % (len(rarr), required_rarr_length)
        for src in projection.pre.fullgidlist:
            for tgt in projection.post.gidlist:
                if self.allow_self_connections or projection.pre != projection.post or tgt != src:
                    if rarr[j] < self.p_connect:  
                        if isinstance(weight, RandomDistribution): w = weight.next()
                        else: w = weight
                        if isinstance(delay, RandomDistribution): d = delay.next()
                        else: d = delay
                        hoc_commands += self.singleConnect(projection, src, tgt, w, d) 
                j += 1
        return hoc_commands


class DistanceDependentProbabilityConnector(common.DistanceDependentProbabilityConnector, HocConnector):
    
    def connect(self, projection):
        weight = self.getWeight(self.weights)
        delay = self.getDelay(self.delays)
        periodic_boundaries = self.periodic_boundaries
        if periodic_boundaries is not None:
            dimensions = projection.post.dim
            periodic_boundaries = numpy.concatenate((dimensions, numpy.zeros(3-len(dimensions))))
        if isinstance(projection.rng, NativeRNG):
            hoc_commands = ['rng = new Random(%d)' % 0 or distribution.rng.seed,
                            'tmp = rng.uniform(0,1)']
            # Here we are forced to execute the commands on line to be able to
            # catch the connections from NEURON
            hoc_execute(hoc_commands)
            #rarr = [HocToPy.get('rng.repick()','float') for j in xrange(projection.pre.size*projection.post.size)]
            rarr = [h.rng.repick() for j in xrange(projection.pre.size*projection.post.size)]
        else:
            rarr = projection.rng.uniform(0,1, projection.pre.size*projection.post.size)
        # We need to use the gid stored as ID, so we should modify the loop to scan the global gidlist (containing ID)
        hoc_commands = []
        j = 0
        for tgt in projection.post.gidlist:
            idx_pre  = 0
            distances = common.distances(projection.pre, tgt, self.mask, self.scale_factor, self.offset, periodic_boundaries)
            for src in projection.pre.fullgidlist:
                if self.allow_self_connections or projection.pre != projection.post or tgt != src: 
                    # calculate the distance between the two cells :
                    d = distances[idx_pre][0]
                    p = eval(self.d_expression)
                    if p >= 1 or (0 < p < 1 and rarr[j] < p):
                        if isinstance(weight, RandomDistribution): w = weight.next()
                        else: w = weight
                        if isinstance(delay, RandomDistribution): d = delay.next()
                        else: d = delay
                        hoc_commands += self.singleConnect(projection, src, tgt, w, d)
                j += 1
                idx_pre += 1
        return hoc_commands


class FixedNumberPreConnector(common.FixedNumberPreConnector, HocConnector):
    
    def connect(self, projection):
        raise Exception("Not implemented yet !")


class FixedNumberPostConnector(common.FixedNumberPostConnector, HocConnector):
    
    def connect(self, projection):
        raise Exception("Not implemented yet !")


class FromListConnector(common.FromListConnector, HocConnector):
    
    def connect(self, projection):
        return self._process_conn_list(self.conn_list, projection)

    
class FromFileConnector(common.FromFileConnector, HocConnector):
    
    def connect(self, projection):
        if self.distributed:
            myid = int(h.pc.id())
            self.filename += ".%d" % myid
        # open the file...
        f = open(self.filename, 'r', 10000)
        lines = f.readlines()
        f.close()
        # gather all the data in a list of tuples (one per line)
        input_tuples = []
        for line in lines:
            single_line = line.rstrip()
            src, tgt, w, d = single_line.split("\t", 4)
            src = "[%s" % src.split("[",1)[1]
            tgt = "[%s" % tgt.split("[",1)[1]
            input_tuples.append((eval(src), eval(tgt), float(w), float(d)))
        return self._process_conn_list(input_tuples, projection)