from pyNN import common
from pyNN.common.populations import BasePopulation
try:
    from unittest.mock import Mock
except ImportError:
    from mock import Mock
from inspect import isfunction
from nose.tools import assert_equal


def test_build_create():
    population_class = Mock()
    create_function = common.build_create(population_class)
    assert isfunction(create_function)

    p = create_function("cell class", "cell params", n=999)
    population_class.assert_called_with(999, "cell class", cellparams="cell params")

def test_build_connect():
    projection_class = Mock()
    connector_class = Mock(return_value="connector")
    syn_class = Mock(return_value="syn")
    connect_function = common.build_connect(projection_class, connector_class, syn_class)
    assert isfunction(connect_function)

    prj = connect_function("source", "target", "weight", "delay", "receptor_type", "p", "rng")
    syn_class.assert_called_with(weight="weight", delay="delay")
    connector_class.assert_called_with(p_connect="p", rng="rng")
    projection_class.assert_called_with("source", "target", "connector", synapse_type="syn", receptor_type="receptor_type")

    class MockID(common.IDMixin):
       def as_view(self):
            return "view"

    prj = connect_function(MockID(), MockID(), "weight", "delay", "receptor_type", "p", "rng")
    projection_class.assert_called_with("view", "view", "connector", synapse_type="syn", receptor_type="receptor_type")

def test_set():
    cells = BasePopulation()
    cells.set = Mock()
    common.set(cells, param="val")
    cells.set.assert_called_with(param="val")

def test_build_record():
    simulator = Mock()
    simulator.state.write_on_end = []
    record_function = common.build_record(simulator)
    assert isfunction(record_function)

    source = BasePopulation()
    source.record = Mock()
    record_function(('v', 'spikes'), source, "filename")
    source.record.assert_called_with(('v', 'spikes'), to_file="filename", sampling_interval=None)
    assert_equal(simulator.state.write_on_end, [(source, ('v', 'spikes'), "filename")])

def test_build_record_with_assembly():
    simulator = Mock()
    simulator.state.write_on_end = []
    record_function = common.build_record(simulator)
    assert isfunction(record_function)

    p1 = BasePopulation()
    p2 = BasePopulation()
    source = common.Assembly(p1, p2)
    source.record = Mock()
    record_function('foo', source, "filename")
    source.record.assert_called_with('foo', to_file="filename", sampling_interval=None)
    assert_equal(simulator.state.write_on_end, [(source, 'foo', "filename")]) # not sure this is what we want - won't file get over-written?
