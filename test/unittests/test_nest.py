import nest
import numpy
from mock import Mock
from nose.tools import assert_equal
from pyNN.standardmodels import StandardCellType, build_translations
from pyNN.nest import Population
from pyNN.nest.cells import NativeCellType
from pyNN.common import IDMixin

class MockStandardCellType(StandardCellType):
    default_parameters = {
        "foo": 99.9,
        "hoo": 100.0,
        "woo": 5.0,
    }
    default_initial_values = {
        "v": 0.0,
    }
    translations = build_translations(
        ('foo', 'FOO'),
        ('hoo', 'HOO', 3.0),
        ('woo', 'WOO', '2*woo + hoo', '(WOO - HOO)/2'),
    )


class MockNativeCellType(NativeCellType):
    default_parameters = {
        "FOO": 99.9,
        "HOO": 300.0,
        "WOO": 112.0,
    }
    default_initial_values = {
        "v": 0.0,
    }
    nest_model = "mock_neuron"


class MockID(IDMixin):
    set_parameters = Mock()


class TestPopulation(object):

    def setup(self):
        self.orig_cc = Population._create_cells
        self.orig_init = Population.initialize
        self.orig_ss = nest.SetStatus
        Population._create_cells = Mock()
        Population.initialize = Mock()
        nest.SetStatus = Mock()
        
    def teardown(self):
        Population._create_cells = self.orig_cc
        Population.initialize = self.orig_init
        nest.SetStatus = self.orig_ss

    def test_set_with_standard_celltype(self):
        p = Population(10, MockStandardCellType)
        p.all_cells = numpy.array([MockID()]*10, dtype=object) #numpy.arange(10)
        p._mask_local = numpy.ones((10,), bool)
        p.set("foo", 32)
        assert_equal(nest.SetStatus.call_args[0][1], {"FOO": 32.0})
        p.set("hoo", 33.0)
        assert_equal(nest.SetStatus.call_args[0][1], {"HOO": 99.0})
        p.set("woo", 6.0)
        assert_equal(nest.SetStatus.call_args[0][1], {})
        p.all_cells[0].set_parameters.assert_called_with(woo=6.0)
        
    def test_set_with_native_celltype(self):
        gd_orig = nest.GetDefaults
        nest.GetDefaults = Mock(return_value={"FOO": 1.2, "HOO": 3.4, "WOO": 5.6})
        p = Population(10, MockNativeCellType)
        p.all_cells = numpy.array([MockID()]*10, dtype=object) #numpy.arange(10)
        p._mask_local = numpy.ones((10,), bool)
        p.set("FOO", 32)
        assert_equal(nest.SetStatus.call_args[0][1], {"FOO": 32.0})
        p.set("HOO", 33.0)
        assert_equal(nest.SetStatus.call_args[0][1], {"HOO": 33.0})
        p.set("WOO", 6.0)
        assert_equal(nest.SetStatus.call_args[0][1], {"WOO": 6.0})
        nest.GetDefaults = gd_orig