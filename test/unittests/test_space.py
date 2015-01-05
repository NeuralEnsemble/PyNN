"""
Tests of the `space` module.

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

from pyNN import space
import unittest
import numpy
try:
    from unittest.mock import Mock
except ImportError:
    from mock import Mock
from nose.tools import assert_equal, assert_raises
from pyNN.utility import assert_arrays_equal
from math import sqrt

def assert_arrays_almost_equal(a, b, threshold, msg=''):
    if a.shape != b.shape:
        raise unittest.TestCase.failureException("Shape mismatch: a.shape=%s, b.shape=%s" % (a.shape, b.shape))
    if not (abs(a-b) < threshold).all():
        err_msg = "%s != %s" % (a, b)
        err_msg += "\nlargest difference = %g" % abs(a-b).max()
        if msg:
            err_msg += "\nOther information: %s" % msg
        raise unittest.TestCase.failureException(err_msg)

def test_distance():
    cell1 = Mock()
    cell2 = Mock()
    A = lambda *x: numpy.array(x)
    cell1.position = A(2.3, 4.5, 6.7)
    cell2.position = A(2.3, 4.5, 6.7)
    assert_equal(space.distance(cell1, cell2), 0.0)
    cell2.position = A(5.3, 4.5, 6.7)
    assert_equal(space.distance(cell1, cell2), 3.0)
    cell2.position = A(5.3, 8.5, 6.7)
    assert_equal(space.distance(cell1, cell2), 5.0)
    cell2.position = A(5.3, 8.5, -5.3)
    assert_equal(space.distance(cell1, cell2), 13.0)
    assert_equal(space.distance(cell1, cell2, mask=A(0,1)), 5.0)
    assert_equal(space.distance(cell1, cell2, mask=A(2)), 12.0)
    assert_equal(space.distance(cell1, cell2, offset=A(-3.0, -4.0, 12.0)), 0.0)
    cell2.position = A(10.6, 17.0, -10.6)
    assert_equal(space.distance(cell1, cell2, scale_factor=0.5), 13.0)
    cell2.position = A(-1.7, 8.5, -5.3)
    assert_equal(space.distance(cell1, cell2, periodic_boundaries=A(7.0, 1e12, 1e12)), 13.0)



class SpaceTest(unittest.TestCase):

    def setUp(self):
        N = numpy.array
        self.A = N([0.0, 0.0, 0.0])
        self.B = N([1.0, 1.0, 1.0])
        self.C = N([-1.0, -1.0, -1.0])
        self.D = N([2.0, 3.0, 4.0])
        self.ABCD = N([[0.0, 0.0, 0.0],
                       [1.0, 1.0, 1.0],
                       [-1.0, -1.0, -1.0],
                       [2.0, 3.0, 4.0]])

    def assertArraysEqual(self, A, B):
        self.assert_((A==B).all(), "%s != %s" % (A,B))

    def test_infinite_space_with_3D_distances(self):
        s = space.Space()
        self.assertEqual(s.distances(self.A, self.B), sqrt(3))
        self.assertEqual(s.distances(self.C, self.B), sqrt(12))
        self.assertArraysEqual(s.distances(self.A, self.ABCD),
                               numpy.array([0.0, sqrt(3), sqrt(3), sqrt(29)]))
        self.assertArraysEqual(s.distances(self.A, self.ABCD),
                               s.distances(self.ABCD, self.A).T)
        assert_arrays_equal(s.distances(self.ABCD, self.ABCD),
                            numpy.array([0.0, sqrt(3), sqrt(3), sqrt(29),
                                         sqrt(3), 0.0, sqrt(12), sqrt(14),
                                         sqrt(3), sqrt(12), 0.0, sqrt(50.0),
                                         sqrt(29), sqrt(14), sqrt(50.0), 0.0]))
        self.assertArraysEqual(s.distances(self.ABCD, self.A),
                               numpy.array([0.0, sqrt(3), sqrt(3), sqrt(29)]))

    def test_generator_for_infinite_space_with_3D_distances(self):
        s = space.Space()
        f = lambda i: self.ABCD[i]
        g = lambda j: self.ABCD[j]
        self.assertArraysEqual(s.distance_generator(f, g)(0, numpy.arange(4)),
                               numpy.array([0.0, sqrt(3), sqrt(3), sqrt(29)]))
        assert_arrays_equal(numpy.fromfunction(s.distance_generator(f, g), shape=(4, 4), dtype=int),
                            numpy.array([(0.0, sqrt(3), sqrt(3), sqrt(29)),
                                         (sqrt(3), 0.0, sqrt(12), sqrt(14)),
                                         (sqrt(3), sqrt(12), 0.0, sqrt(50.0)),
                                         (sqrt(29), sqrt(14), sqrt(50.0), 0.0)]))

    def test_infinite_space_with_collapsed_axes(self):
        s_x = space.Space(axes='x')
        s_xy = space.Space(axes='xy')
        s_yz = space.Space(axes='yz')
        self.assertEqual(s_x.distances(self.A, self.B), 1.0)
        self.assertEqual(s_xy.distances(self.A, self.B), sqrt(2))
        self.assertEqual(s_x.distances(self.A, self.D), 2.0)
        self.assertEqual(s_xy.distances(self.A, self.D), sqrt(13))
        self.assertEqual(s_yz.distances(self.A, self.D), sqrt(25))
        self.assertArraysEqual(s_yz.distances(self.D, self.ABCD),
                               numpy.array([sqrt(25), sqrt(13), sqrt(41), sqrt(0)]))

    def test_infinite_space_with_scale_and_offset(self):
        s = space.Space(scale_factor=2.0, offset=1.0)
        self.assertEqual(s.distances(self.A, self.B), sqrt(48))
        self.assertEqual(s.distances(self.B, self.A), sqrt(3))
        self.assertEqual(s.distances(self.C, self.B), sqrt(75))
        self.assertEqual(s.distances(self.B, self.C), sqrt(3))
        self.assertArraysEqual(s.distances(self.A, self.ABCD),
                               numpy.array([sqrt(12), sqrt(48), sqrt(0), sqrt(200)]))

    def test_cylindrical_space(self):
        s = space.Space(periodic_boundaries=((-1.0, 4.0), (-1.0, 4.0), (-1.0, 4.0)))
        self.assertEqual(s.distances(self.A, self.B), sqrt(3))
        self.assertEqual(s.distances(self.A, self.D), sqrt(4+4+1))
        self.assertEqual(s.distances(self.C, self.D), sqrt(4+1+0))
        self.assertArraysEqual(s.distances(self.A, self.ABCD),
                               numpy.array([0.0, sqrt(3), sqrt(3), sqrt(4+4+1)]))
        self.assertArraysEqual(s.distances(self.A, self.ABCD),
                               s.distances(self.ABCD, self.A).T)
        self.assertArraysEqual(s.distances(self.C, self.ABCD),
                               numpy.array([sqrt(3), sqrt(4+4+4), 0.0, sqrt(4+1+0)]))


class LineTest(unittest.TestCase):

    def test_generate_positions_default_parameters(self):
        line = space.Line()
        n = 4
        positions = line.generate_positions(n)
        assert_equal(positions.shape, (3,n))
        assert_arrays_almost_equal(
            positions,
            numpy.array([[0,0,0], [1,0,0], [2,0,0], [3,0,0]], float).T,
            threshold=1e-15
        )

    def test_generate_positions(self):
        line = space.Line(dx=100.0, x0=-100.0, y=444.0, z=987.0)
        n = 2
        positions = line.generate_positions(n)
        assert_equal(positions.shape, (3,n))
        assert_arrays_almost_equal(
            positions,
            numpy.array([[-100,444,987], [0,444,987]], float).T,
            threshold=1e-15
        )

    def test__eq__(self):
        line1 = space.Line()
        line2 = space.Line(1.0, 0.0, 0.0, 0.0)
        line3 = space.Line(dx=2.0)
        assert_equal(line1, line2)
        assert line1 != line3

    def test_get_parameters(self):
        params = dict(dx=100.0, x0=-100.0, y=444.0, z=987.0)
        line = space.Line(**params)
        assert_equal(line.get_parameters(), params)


class Grid2D_Test(object):

    def setup(self):
        self.grid1 = space.Grid2D()
        self.grid2 = space.Grid2D(aspect_ratio=3.0, dx=11.1, dy=9.9, x0=123, y0=456, z=789)

    def test_calculate_size(self):
        assert_equal(self.grid1.calculate_size(n=1), (1,1))
        assert_equal(self.grid1.calculate_size(n=4), (2,2))
        assert_equal(self.grid1.calculate_size(n=9), (3,3))
        assert_raises(Exception, self.grid1.calculate_size, n=10)
        assert_equal(self.grid2.calculate_size(n=3), (3,1))
        assert_equal(self.grid2.calculate_size(n=12), (6,2))
        assert_equal(self.grid2.calculate_size(n=27), (9,3))
        assert_raises(Exception, self.grid2.calculate_size, n=4)

    def test_generate_positions(self):
        n = 4
        positions = self.grid1.generate_positions(n)
        assert_equal(positions.shape, (3,n))
        assert_arrays_almost_equal(
            positions,
            numpy.array([
                [0,0,0], [0,1,0],
                [1,0,0], [1,1,0]
                ]).T,
            1e-15)
        assert_arrays_almost_equal(
            self.grid2.generate_positions(12),
            numpy.array([
                [123,456,789], [123,465.9,789],
                [123+11.1,456,789], [123+11.1,465.9,789],
                [123+22.2,456,789], [123+22.2,465.9,789],
                [123+33.3,456,789], [123+33.3,465.9,789],
                [123+44.4,456,789], [123+44.4,465.9,789],
                [123+55.5,456,789], [123+55.5,465.9,789],
            ]).T,
            1e-15)


class Grid3D_Test(object):

    def setup(self):
        self.grid1 = space.Grid3D()
        self.grid2 = space.Grid3D(aspect_ratioXY=3.0,
                                  aspect_ratioXZ=2.0,
                                  dx=11, dy=9, dz=7,
                                  x0=123, y0=456, z0=789)

    def test_calculate_size(self):
        assert_equal(self.grid1.calculate_size(n=1), (1,1,1))
        assert_equal(self.grid1.calculate_size(n=8), (2,2,2))
        assert_equal(self.grid1.calculate_size(n=27), (3,3,3))
        assert_raises(Exception, self.grid1.calculate_size, n=10)
        assert_equal(self.grid2.calculate_size(n=36), (6,2,3))
        assert_equal(self.grid2.calculate_size(n=288), (12,4,6))
        assert_raises(Exception, self.grid2.calculate_size, n=100)

    def test_generate_positions(self):
        n = 8
        positions = self.grid1.generate_positions(n)
        assert_equal(positions.shape, (3,n))
        assert_arrays_almost_equal(
            positions,
            numpy.array([
                [0,0,0], [0,0,1], [0,1,0], [0,1,1],
                [1,0,0], [1,0,1], [1,1,0], [1,1,1]
                ]).T,
            1e-15)


class TestSphere(object):

    def test__create(self):
        s = space.Sphere(2.5)
        assert_equal(s.radius, 2.5)

    def test_sample(self):
        n = 1000
        s = space.Sphere(2.5)
        positions = s.sample(n, numpy.random)
        assert_equal(positions.shape, (n,3))
        for axis in range(2):
            assert 1 < max(positions[:,axis]) < 2.5
            assert -1 > min(positions[:,axis]) > -2.5
        s2 = numpy.sum(positions**2, axis=1)
        assert max(s2) < 6.25


class TestCuboid(object):

    def test_sample(self):
        n = 1000
        c = space.Cuboid(3, 4, 5)
        positions = c.sample(n, numpy.random)
        assert_equal(positions.shape, (n,3))
        assert 1 < max(positions[:,0]) < 1.5, max(positions[:,0])
        assert -1 > min(positions[:,0]) > -1.5
        assert -1.5 > min(positions[:,1]) > -2.0
        assert -2 > min(positions[:,2]) > -2.5


class TestRandomStructure(object):

    def test_generate_positions(self):
        n = 1000
        s = space.Sphere(2.5)
        rs = space.RandomStructure(boundary=s, origin=(1.0, 1.0, 1.0))
        positions = rs.generate_positions(n)
        assert_equal(positions.shape, (3,n))
        for axis in range(2):
            assert 3 < max(positions[axis,:]) < 3.5
            assert -1 > min(positions[axis,:]) > -1.5


