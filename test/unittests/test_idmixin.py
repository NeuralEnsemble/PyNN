from pyNN import common, errors, standardmodels
from nose.tools import assert_equal, assert_raises

class MockStandardCell(standardmodels.StandardCellType):
    default_parameters = {
        'a': 20.0,
        'b': -34.9,
        'c': 2.2,
    }
    translations = standardmodels.build_translations(('a', 'A'), ('b', 'B'), ('c', 'C', 'c + a', 'C - A'))

class MockNativeCell(object):
    @classmethod
    def has_parameter(cls, name):
        return False
    @classmethod
    def get_parameter_names(cls):
        return []

class MockPopulation(object):
    def __init__(self, standard):
        if standard:
            self.celltype = MockStandardCell()
        else:
            self.celltype = MockNativeCell()
        self._is_local_called = False
        self._positions = {}
        self._initial_values = {}
    def id_to_index(self, id):
        return 1234
    def is_local(self, id):
        self._is_local_called = True
        return True
    def _set_cell_position(self, id, pos):
        self._positions[id] = pos
    def _get_cell_position(self, id):
        return (1.2, 3.4, 5.6)
    def _get_cell_initial_value(self, id, variable):
        return -65.0
    def _set_cell_initial_value(self, id, variable, value):
        self._initial_values[id] = (variable, value)

class MockID(common.IDMixin):
    def __init__(self, standard_cell):
        self.parent = MockPopulation(standard=standard_cell)
        self.foo = "bar"
        self._parameters = {'A': 76.5, 'B': 23.4, 'C': 100.0}


class MockCurrentSource(object):
    def __init__(self):
        self._inject_into = []
    def inject_into(self, objs):
        self._inject_into.extend(objs)

class Test_IDMixin():

    def setup(self):
        self.id = MockID(standard_cell=True)
        self.id_ns = MockID(standard_cell=False)

    #def test_getattr_with_parameter_attr(self):
    #    assert_equal(self.id.a, 76.5)
    #    assert_equal(self.id_ns.A, 76.5)
    #    assert_raises(errors.NonExistentParameterError, self.id.__getattr__, "tau_m")
    #    assert_raises(errors.NonExistentParameterError, self.id_ns.__getattr__, "tau_m")

    def test_getattr_with_nonparameter_attr(self):
        assert_equal(self.id.foo, "bar")
        assert_equal(self.id_ns.foo, "bar")

    def test_getattr_with_parent_not_set(self):
        del(self.id.parent)
        assert_raises(Exception, self.id.__getattr__, "parent")

    #def test_setattr_with_parameter_attr(self):
    #    self.id.a = 87.6
    #    self.id_ns.A = 98.7
    #    assert_equal(self.id.a, 87.6)
    #    assert_equal(self.id_ns.A, 98.7)

    #def test_set_parameters(self):
    #    assert_raises(errors.NonExistentParameterError, self.id.set_parameters, hello='world')
    #    ##assert_raises(errors.NonExistentParameterError, self.id_ns.set_parameters, hello='world')
    #    self.id.set_parameters(a=12.3, c=77.7)
    #    assert_equal(self.id._parameters, {'A': 12.3, 'B': 23.4, 'C': 90.0})

    #def test_get_parameters(self):
    #    assert_equal(self.id.get_parameters(), {'a': 76.5, 'b': 23.4, 'c': 23.5})

    def test_celltype_property(self):
        assert_equal(self.id.celltype.__class__, MockStandardCell)
        assert_equal(self.id_ns.celltype.__class__, MockNativeCell)

    def test_is_standard_cell(self):
        assert self.id.is_standard_cell
        assert not self.id_ns.is_standard_cell

    def test_position_property(self):
        for id in (self.id, self.id_ns):
            assert_equal(id.position, (1.2, 3.4, 5.6))
            id.position = (9,8,7)
            assert_equal(id.parent._positions[id], (9,8,7))

    def test_local_property(self):
        for id in (self.id, self.id_ns):
            assert id.parent._is_local_called is False
            assert id.local
            assert id.parent._is_local_called is True

    def test_inject(self):
        for id in (self.id, self.id_ns):
            cs = MockCurrentSource()
            id.inject(cs)
            assert_equal(cs._inject_into, [id])

    def test_get_initial_value(self):
        self.id.get_initial_value('v')

    def test_set_initial_value(self):
        self.id.set_initial_value('v', -77.7)
        assert_equal(self.id.parent._initial_values[self.id], ('v', -77.7))
