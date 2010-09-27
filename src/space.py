# encoding: utf-8
"""
Tools for performing spatial/topographical calculations.

"""

# There must be some Python package out there that provides most of this stuff.

import numpy
import math
from operator import and_
from pyNN.random import NumpyRNG

def distance(src, tgt, mask=None, scale_factor=1.0, offset=0.0,
             periodic_boundaries=None): # may need to add an offset parameter
    """
    Return the Euclidian distance between two cells.
    `mask` allows only certain dimensions to be considered, e.g.::
      * to ignore the z-dimension, use `mask=array([0,1])`
      * to ignore y, `mask=array([0,2])`
      * to just consider z-distance, `mask=array([2])`
    `scale_factor` allows for different units in the pre- and post- position
    (the post-synaptic position is multipied by this quantity).
    """
    d = src.position - scale_factor*(tgt.position + offset)
    
    if not periodic_boundaries == None:
        d = numpy.minimum(abs(d), periodic_boundaries-abs(d))
    if mask is not None:
        d = d[mask]
    return numpy.sqrt(numpy.dot(d, d))


class Space(object):
    
    AXES = {'x' : [0],    'y': [1],    'z': [2],
            'xy': [0,1], 'yz': [1,2], 'xz': [0,2], 'xyz': range(3), None: range(3)}
    
    def __init__(self, axes=None, scale_factor=1.0, offset=0.0,
                 periodic_boundaries=None):
        """
        axes -- if not supplied, then the 3D distance is calculated. If supplied,
            axes should be a string containing the axes to be used, e.g. 'x', or
            'yz'. axes='xyz' is the same as axes=None.
        scale_factor -- it may be that the pre and post populations use
            different units for position, e.g. degrees and µm. In this case,
            `scale_factor` can be specified, which is applied to the positions
            in the post-synaptic population.
        offset -- if the origins of the coordinate systems of the pre- and post-
            synaptic populations are different, `offset` can be used to adjust
            for this difference. The offset is applied before any scaling.
        periodic_boundaries -- either `None`, or a tuple giving the boundaries
            for each dimension, e.g. `((x_min, x_max), None, (z_min, z_max))`.
        """
        self.periodic_boundaries = periodic_boundaries
        self.axes = numpy.array(Space.AXES[axes])
        self.scale_factor = scale_factor
        self.offset = offset
        
    def distances(self, A, B, expand=False):
        """
        Calculate the distance matrix between two sets of coordinates, given
        the topology of the current space.
        From http://projects.scipy.org/pipermail/numpy-discussion/2007-April/027203.html
        """
        if len(A.shape) == 1:
            A = A.reshape(3, 1)
        if len(B.shape) == 1:
            B = B.reshape(3, 1)
        # I'm not sure the following line should be here. Operations may be redundant and not very 
        # transparent from the user point of view. I moved it into the DistanceDependentProbability Connector
        #B = self.scale_factor*(B + self.offset)
        d = numpy.zeros((len(self.axes), A.shape[1], B.shape[1]), dtype=A.dtype)
        for axis in self.axes:
            diff2 = A[axis,:,None] - B[axis, :]
            if self.periodic_boundaries is not None:
                boundaries = self.periodic_boundaries[axis]
                if boundaries is not None:
                    range = boundaries[1]-boundaries[0]
                    ad2   = abs(diff2)
                    diff2 = numpy.minimum(ad2, range-ad2)
            diff2 **= 2
            d[axis] = diff2
        if not expand:
            d = numpy.sum(d, 0)
        numpy.sqrt(d, d)
        return d



class PositionsGenerator(object):

        def __init__(self, dimensions, axes=None):
            """
            dimensions -- either `None`, or a tuple/list giving the dimensions
                          for each dimension, e.g. `((x_min, x_max), None, (z_min, z_max))`.            
            axes -- if not supplied, then the 3D distance is calculated. If supplied,
                    axes should be a string containing the axes to be used, e.g. 'x', or
                    'yz'. axes='xyz' is the same as axes=None.
            """
            self.dimensions = list(dimensions)
            self.axes = numpy.array(Space.AXES[axes]) 
            assert len(self.dimensions) == 3, "Dimensions should be of size 3, and axes should specify orientation!"
            for item in self.dimensions:
                if item is not None:
                    assert len(item) == 2, "dimensions should be a list of tuples (min, max), not %s" %item            
                    assert item[0] <= item[0], "items elements should be (min, max), with min <= max"
            
        def get(self, dims):
            self.M         = numpy.prod(numpy.array(dims))
            self.positions = numpy.zeros((3, self.M))            
            pass


class RandomPositions(PositionsGenerator):

        def __init__(self, dimensions, seed=None):
            PositionsGenerator.__init__(self, dimensions)
            self.seed = seed
                        
        def get(self, dims):
            PositionsGenerator.get(self, dims)
            numpy.random.seed(self.seed)
            for axis in self.axes:
                item = self.dimensions[axis]            
                if item is not None:
                    bmin, bmax              = item
                    self.positions[axis, :] = bmin + bmax * numpy.random.rand(self.M)
            return self.positions


class GridPositions(PositionsGenerator):

        def get(self, dims):
            PositionsGenerator.get(self, dims)            
            res = ""
            for d in dims:
                res += "0:%d," %d
            grid = eval("numpy.mgrid[%s]" %res[0:-1])
            for axis in self.axes:
                item = self.dimensions[axis]                
                if item is not None:
                    bmin, bmax = item
                    padding    = (bmax-bmin)/float(dims[axis])
                    data       = numpy.linspace(bmin+padding, bmax, dims[axis])
                    self.positions[axis, :] = data[grid[axis].flatten()]
            return self.positions


class BaseStructure(object):
    
    def __eq__(self, other):
        return reduce(and_, (getattr(self, attr) == getattr(other, attr)
                             for attr in self.parameter_names))

    def get_parameters(self):
        P = {}
        for name in self.parameter_names:
            P[name] = getattr(self, name)
        return P


class Line(BaseStructure):
    parameter_names = ("dx", "x0", "y0", "z0")
    
    def __init__(self, dx=1.0, x0=0.0, y0=0.0, z0=0.0):
        self.dx = dx
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
    
    def generate_positions(self, n):
        x = self.dx*numpy.arange(n, dtype=float) + self.x0
        y = numpy.zeros(n) + self.y0
        z = numpy.zeros(n) + self.z0
        return numpy.array((x,y,z))
    
    def describe(self, n):
        return "line with %d positions" % n


class Grid2D(BaseStructure):
    parameter_names = ("aspect_ratio", "dx", "dy", "x0", "y0", "fill_order")
    
    def __init__(self, aspect_ratio=1.0, dx=1.0, dy=1.0, x0=0.0, y0=0.0, z=0, fill_order="sequential"):
        """
        aspect_ratio - ratio of the number of grid points per side (not the ratio
                       of the side lengths, unless dx == dy)
        """
        self.aspect_ratio = aspect_ratio
        assert fill_order in ('sequential', 'random')
        self.fill_order = fill_order
        self.dx = dx; self.dy = dy; self.x0 = x0; self.y0 = y0; self.z = z
    
    def calculate_size(self, n):
        nx = math.sqrt(n*self.aspect_ratio)
        ny = n/nx
        return nx, ny
    
    def generate_positions(self, n):
        nx, ny = self.calculate_size(n)
        x,y,z = numpy.indices((nx,ny,1), dtype=float)
        x = self.x0 + self.dx*x.flatten()
        y = self.y0 + self.dy*y.flatten()
        z = self.z + z.flatten()
        if self.fill_order == 'sequential':
            return numpy.array((x,y,z)) # use column_stack, if we decide to switch from (3,n) to (n,3)
        else:
            raise NotImplementedError
    
    def describe(self, n):
        return "2D grid of size (%d, %d)" % self.calculate_size(n)
        

class Grid3D(BaseStructure):
    parameter_names = ("aspect_ratios", "dx", "dy", "dz", "x0", "y0", "z0", "fill_order")
    
    def __init__(self, aspect_ratioXY, aspect_ratioXZ, dx=1.0, dy=1.0, dz=1.0, x0=0.0, y0=0.0, z0=0,
                 fill_order="sequential"):
        """
        If fill_order is 'sequential', the z-index will be filled first, then y then x, i.e.
        the first cell will be at (0,0,0) (given default values for the other arguments),
        the second at (0,0,1), etc.
        """
        self.aspect_ratios = (aspect_ratioXY, aspect_ratioXZ)
        assert fill_order in ('sequential', 'random')
        self.fill_order = fill_order
        self.dx = dx; self.dy = dy; self.dz = dz
        self.x0 = x0; self.y0 = y0; self.z0 = z0
    
    def calculate_size(self, n):
        a,b = self.aspect_ratios
        nx = int(round(math.pow(n*a*b, 1/3.0)))
        ny = int(round(nx/a))
        nz = int(round(nx/b))
        assert nx*ny*nz == n, str((nx, ny, nz, nx*ny*nz, n, a, b))
        return nx, ny, nz
    
    def generate_positions(self, n):
        nx, ny, nz = self.calculate_size(n)
        x,y,z = numpy.indices((nx,ny,nz), dtype=float)
        x = self.x0 + self.dx*x.flatten()
        y = self.y0 + self.dy*y.flatten()
        z = self.z0 + self.dz*z.flatten()
        if self.fill_order == 'sequential':
            return numpy.array((x,y,z))
        else:
            raise NotImplementedError

    def describe(self, n):
        return "3D grid of size (%d, %d, %d)" % self.calculate_size(n)

class Shape(object):
    pass

class Cuboid(Shape):
    
    def __init__(self, height, width, depth):
        """Not sure how h,w,d should be related to x,y,z."""
        self.height = height
        self.width = width
        self.depth = depth
        
    def sample(self, n, rng):
        return rng.uniform(size=(n,3)) * (self.height, self.width, self.depth)


class Sphere(Shape):
    
    def __init__(self, radius):
        Shape.__init__(self)
        self.radius = radius
        
    def sample(self, n, rng):
        raise NotImplementedError
        #theta_phi = rng.uniform(size=(n,2))
        #r = rng.???
        # now transform from r,theta,phi to x,y,z


class RandomStructure(BaseStructure):
    parameter_names = ('boundary', 'origin', 'rotation', 'rng')
    
    def __init__(self, boundary, origin=(0.0,0.0,0.0), rotation=(0,0), rng=None):
        """
        `boundary` - a subclass of Shape
        `origin` - a coordinate tuple (x,y,z)
        `rotation` - (theta, phi) tuple giving the rotation to be applied (in degrees)
        
        Note that the rotation is applied before the origin shift
        """
        self.boundary = boundary
        self.origin = origin
        self.rng = rng or NumpyRNG()
        
    def generate_positions(self, n):
        return numpy.array(self.origin) + rotate(self.boundary.sample(n, rng), *self.origin)
    