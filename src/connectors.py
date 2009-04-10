# encoding: utf-8
"""
Base Connector classes

"""

import logging
import numpy
from numpy import arccos, arcsin, arctan, arctan2, ceil, cos, cosh, e, exp, \
                  fabs, floor, fmod, hypot, ldexp, log, log10, modf, pi, power, \
                  sin, sinh, sqrt, tan, tanh
from common import Connector, simulator, distances
from pyNN import random


class AllToAllConnector(Connector):
    """
    Connects all cells in the presynaptic population to all cells in the
    postsynaptic population.
    """
    
    def __init__(self, allow_self_connections=True, weights=0.0, delays=None):
        Connector.__init__(self, weights, delays)
        assert isinstance(allow_self_connections, bool)
        self.allow_self_connections = allow_self_connections

    def connect(self, projection):
        simulator.probabilistic_connect(self, projection, 1.0)


class FromListConnector(Connector):
    """
    Make connections according to a list.
    """
    
    def __init__(self, conn_list):
        Connector.__init__(self, 0., 0.)
        self.conn_list = conn_list        


class FromFileConnector(Connector):
    """
    Make connections according to a list read from a file.
    """
    
    def __init__(self, filename, distributed=False):
        Connector.__init__(self, 0., 0.)
        self.filename = filename
        self.distributed = distributed
       
        
class FixedNumberPostConnector(Connector):
    """
    Each pre-synaptic neuron is connected to exactly n post-synaptic neurons
    chosen at random.
    """
    
    def __init__(self, n, allow_self_connections=True, weights=0.0, delays=None):
        Connector.__init__(self, weights, delays)
        assert isinstance(allow_self_connections, bool)
        self.allow_self_connections = allow_self_connections
        if isinstance(n, int):
            self.n = n
            assert n >= 0
        elif isinstance(n, random.RandomDistribution):
            self.rand_distr = n
            # weak check that the random distribution is ok
            assert numpy.all(numpy.array(n.next(100)) >= 0), "the random distribution produces negative numbers"
        else:
            raise Exception("n must be an integer or a RandomDistribution object")


class FixedNumberPreConnector(Connector):
    """
    Each post-synaptic neuron is connected to exactly n pre-synaptic neurons
    chosen at random.
    """
    def __init__(self, n, allow_self_connections=True, weights=0.0, delays=None):
        Connector.__init__(self, weights, delays)
        assert isinstance(allow_self_connections, bool)
        self.allow_self_connections = allow_self_connections
        if isinstance(n, int):
            self.n = n
            assert n >= 0
        elif isinstance(n, random.RandomDistribution):
            self.rand_distr = n
            # weak check that the random distribution is ok
            assert numpy.all(numpy.array(n.next(100)) >= 0), "the random distribution produces negative numbers"
        else:
            raise Exception("n must be an integer or a RandomDistribution object")


class OneToOneConnector(Connector):
    """
    Where the pre- and postsynaptic populations have the same size, connect
    cell i in the presynaptic population to cell i in the postsynaptic
    population for all i.
    In fact, despite the name, this should probably be generalised to the
    case where the pre and post populations have different dimensions, e.g.,
    cell i in a 1D pre population of size n should connect to all cells
    in row i of a 2D post population of size (n,m).
    """
    
    def __init__(self, weights=0.0, delays=None):
        Connector.__init__(self, weights, delays)
    
    
class FixedProbabilityConnector(Connector):
    """
    For each pair of pre-post cells, the connection probability is constant.
    """
    
    def __init__(self, p_connect, allow_self_connections=True, weights=0.0, delays=None):
        Connector.__init__(self, weights, delays)
        assert isinstance(allow_self_connections, bool)
        self.allow_self_connections = allow_self_connections
        self.p_connect = float(p_connect)
        assert 0 <= self.p_connect
    
    def connect(self, projection):
        logging.info("Connecting %s to %s with probability %s" % (projection.pre.label,
                                                                  projection.post.label,
                                                                  self.p_connect))
        simulator.probabilistic_connect(self, projection, self.p_connect)    

        
class DistanceDependentProbabilityConnector(Connector):
    """
    For each pair of pre-post cells, the connection probability depends on distance.
    d_expression should be the right-hand side of a valid python expression
    for probability, involving 'd', e.g. "exp(-abs(d))", or "float(d<3)"
    If axes is not supplied, then the 3D distance is calculated. If supplied,
    axes should be a string containing the axes to be used, e.g. 'x', or 'yz'
    axes='xyz' is the same as axes=None.
    It may be that the pre and post populations use different units for position, e.g.
    degrees and µm. In this case, `scale_factor` can be specified, which is applied
    to the positions in the post-synaptic population. An offset can also be included.
    """
    
    AXES = {'x' : [0],    'y': [1],    'z': [2],
            'xy': [0,1], 'yz': [1,2], 'xz': [0,2], 'xyz': None, None: None}
    
    def __init__(self, d_expression, axes=None, scale_factor=1.0, offset=0.0,
                 periodic_boundaries=False, allow_self_connections=True,
                 weights=0.0, delays=None):
        Connector.__init__(self, weights, delays)
        assert isinstance(allow_self_connections, bool)
        assert isinstance(d_expression, str)
        try:
            d = 0; assert 0 <= eval(d_expression), eval(d_expression)
            d = 1e12; assert 0 <= eval(d_expression), eval(d_expression)
        except ZeroDivisionError, err:
            raise ZeroDivisionError("Error in the distance expression %s. %s" % (d_expression, err))
        self.d_expression = d_expression
        # We will use the numpy functions, so we need to parse the function
        # given by the user to look for some key function and add numpy
        # in front of them (or add from numpy import *)
        #func = ['exp','log','sin','cos','cosh','sinh','tan','tanh']
        #for item in func:
            #self.d_expression = self.d_expression.replace(item,"numpy.%s" %item)
        self.allow_self_connections = allow_self_connections
        self.mask = DistanceDependentProbabilityConnector.AXES[axes]
        self.periodic_boundaries = periodic_boundaries
        if self.mask is not None:
            self.mask = numpy.array(self.mask)
        self.scale_factor = scale_factor
        self.offset = offset
        
    def connect(self, projection):
        periodic_boundaries = self.periodic_boundaries
        if periodic_boundaries is True:
            dimensions = projection.post.dim
            periodic_boundaries = tuple(numpy.concatenate((dimensions, numpy.zeros(3-len(dimensions)))))
        if periodic_boundaries:
            print "Periodic boundaries activated and set to size ", periodic_boundaries
        # this is not going to work for parallel sims
        # either build up d gradually by iterating over local target cells,
        # or use the local_mask somehow to pick out only part of the distance matrix
        d = distances(projection.pre, projection.post, self.mask,
                      self.scale_factor, self.offset,
                      periodic_boundaries)
        p_array = eval(self.d_expression)
        if isinstance(self.weights,str):
            raise Exception("The weights distance dependent are not implemented yet")
        if isinstance(self.delays,str):
            raise Exception("The delays distance dependent are not implemented yet")
        simulator.probabilistic_connect(self, projection, p_array.flatten())