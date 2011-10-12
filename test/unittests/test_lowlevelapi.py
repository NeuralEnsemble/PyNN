from pyNN import common
from pyNN.common.populations import BasePopulation
from mock import Mock
from inspect import isfunction
from nose.tools import assert_equal


def test_build_create():
    population_class = Mock()
    create_function = common.build_create(population_class)
    assert isfunction(create_function)
    
    p = create_function("cell class", "cell params", n=999)
    population_class.assert_called_with(999, "cell class", "cell params")
    
def test_build_connect():
    projection_class = Mock()
    connector_class = Mock(return_value="connector")
    connect_function = common.build_connect(projection_class, connector_class)
    assert isfunction(connect_function)
    
    prj = connect_function("source", "target", "weight", "delay", "synapse_type", "p", "rng")
    connector_class.assert_called_with(p_connect="p", weights="weight", delays="delay")
    projection_class.assert_called_with("source", "target", "connector", target="synapse_type", rng="rng")
    
    class MockID(common.IDMixin):
       def as_view(self):
            return "view"
 
    prj = connect_function(MockID(), MockID(), "weight", "delay", "synapse_type", "p", "rng")
    projection_class.assert_called_with("view", "view", "connector", target="synapse_type", rng="rng")
    
def test_set():
    cells = BasePopulation()
    cells.set = Mock()
    common.set(cells, "param", "val")
    cells.set.assert_called_with("param", "val")
    
def test_build_record():
    simulator = Mock()
    simulator.recorder_list = []
    record_function = common.build_record("foo", simulator)
    assert isfunction(record_function)
    
    source = BasePopulation()
    source._record = Mock()
    source.recorders = {'foo': Mock()}
    record_function(source, "filename")
    source._record.assert_called_with("foo", to_file="filename")
    assert_equal(simulator.recorder_list, [source.recorders['foo']])

def test_build_record_with_assembly():
    simulator = Mock()
    simulator.recorder_list = []
    record_function = common.build_record("foo", simulator)
    assert isfunction(record_function)
    
    p1 = BasePopulation()
    p2 = BasePopulation()
    source = common.Assembly(p1, p2)
    source._record = Mock()
    for p in p1, p2:
        p.recorders = {'foo': Mock()}
    record_function(source, "filename")
    source._record.assert_called_with("foo", to_file="filename")
    assert_equal(simulator.recorder_list, [p1.recorders['foo'], p2.recorders['foo']])
