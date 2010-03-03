# encoding: utf-8
"""
Tools for performing spatial/topographical calculations.

"""

import numpy

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
        
    def distances(self, A, B, post_mask=None):
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
        d = numpy.zeros((A.shape[1], B.shape[1]), dtype=A.dtype)
        for axis in self.axes:
            diff2 = A[axis,:,None] - B[axis, post_mask]
            if self.periodic_boundaries is not None:
                boundaries = self.periodic_boundaries[axis]
                if boundaries is not None:
                    range = boundaries[1]-boundaries[0]
                    ad2   = abs(diff2)
                    diff2 = numpy.minimum(ad2, range-ad2)
            diff2 **= 2
            d[:,post_mask] += diff2
        numpy.sqrt(d[:,post_mask], d[:,post_mask])
        return d
