from pyNN.standardmodels import build_translations, StandardModelType, \
                                SynapseDynamics, STDPMechanism, \
                                STDPWeightDependence, STDPTimingDependence
from pyNN import errors
from nose.tools import assert_equal, assert_raises
from mock import Mock
import numpy

def test_build_translations():
    t = build_translations(
            ('a', 'A'),
            ('b', 'B', 1000.0),
            ('c', 'C', 'c + a', 'C - A')
        )
    assert_equal(set(t.keys()), set(['a', 'b', 'c']))
    assert_equal(set(t['a'].keys()),
                 set(['translated_name', 'forward_transform', 'reverse_transform']))
    assert_equal(t['a']['translated_name'], 'A')
    assert_equal(t['a']['forward_transform'], 'a')
    assert_equal(t['a']['reverse_transform'], 'A')
    assert_equal(t['b']['translated_name'], 'B')
    assert_equal(t['b']['forward_transform'], 'float(1000)*b')
    assert_equal(t['b']['reverse_transform'], 'B/float(1000)')
    assert_equal(t['c']['translated_name'], 'C')
    assert_equal(t['c']['forward_transform'], 'c + a')
    assert_equal(t['c']['reverse_transform'], 'C - A')


    
##test StandardModelType

def test_has_parameter():
    M = StandardModelType
    M.default_parameters = {'a': 22.2, 'b': 33.3}
    assert M.has_parameter('a')
    assert M.has_parameter('b')
    assert not M.has_parameter('z')

def test_get_parameter_names():
    M = StandardModelType
    M.default_parameters = {'a': 22.2, 'b': 33.3}
    assert_equal(set(M.get_parameter_names()), set(['a', 'b']))

def test_instantiate():
    """
    Instantiating a StandardModelType should set self.parameters to the value
    of translate(check_parameters(parameters)).
    """
    M = StandardModelType
    M.default_parameters = {'a': 0.0, 'b': 0.0}
    M.translations = {'a': None, 'b': None}
    P1 = {'a': 22.2, 'b': 33.3}
    P2 = {'A': 22.2, 'B': 333}
    orig_check_parameters = M.check_parameters
    orig_translate = M.translate
    M.check_parameters = Mock(return_value=P2)
    M.translate = Mock()
    m = M(P1)
    assert isinstance(m.parameters, Mock)
    M.check_parameters.assert_called_with(P1, with_defaults=True)
    M.translate.assert_called_with(P2)
    
    M.check_parameters = orig_check_parameters
    M.translate = orig_translate

def test_check_parameters_without_defaults():
    M = StandardModelType
    M.default_parameters = {'a': 22.2, 'b': 33.3, 'c': [1, 2, 3], 'd': 'hello'}
    assert_equal(M.check_parameters({'a': 11, 'c': [4, 5, 6], 'd': 'goodbye'}),
                 {'a': 11.0,'c': [4, 5, 6], 'd': 'goodbye'})

def test_check_parameters_with_defaults():
    M = StandardModelType
    M.default_parameters = {'a': 22.2, 'b': 33.3, 'c': [1, 2, 3]}
    assert_equal(M.check_parameters({'a': 11, 'c': [4, 5, 6]}, with_defaults=True),
                 {'a': 11.0, 'b': 33.3, 'c': [4, 5, 6]})

def test_check_parameters_with_nonexistent_parameter():
    M = StandardModelType
    M.default_parameters = {'a': 22.2, 'b': 33.3, 'c': [1, 2, 3]}
    assert_raises(errors.NonExistentParameterError,
                  M.check_parameters, {'a': 11.1, 'z': 99.9})

def test_check_parameters_with_invalid_value():
    M = StandardModelType
    M.default_parameters = {'a': 22.2, 'b': 33.3, 'c': [1, 2, 3], 'd': 'hello'}
    assert_raises(errors.InvalidParameterValueError,
                  M.check_parameters, {'a': 11.1, 'b': [4,3,2]})
    assert_raises(errors.InvalidParameterValueError,
                  M.check_parameters, {'a': 11.1, 'c': 12.3})
    assert_raises(errors.InvalidParameterValueError,
                  M.check_parameters, {'a': 11.1, 'd': 12.3})

def test_translate():
    M = StandardModelType
    M.default_parameters = {'a': 22.2, 'b': 33.3, 'c': 44.4}
    M.translations = build_translations(
            ('a', 'A'),
            ('b', 'B', 1000.0),
            ('c', 'C', 'c + a', 'C - A'),
        )
    assert_equal(M.translate({'a': 23.4, 'b': 34.5, 'c': 45.6}),
                 {'A': 23.4, 'B': 34500.0, 'C': 69.0})

def test_translate_with_invalid_transformation():
    M = StandardModelType
    M.translations = build_translations(
            ('a', 'A'),
            ('b', 'B', 'b + z', 'B-Z'),
    )
    M.default_parameters = {'a': 22.2, 'b': 33.3}
    #really we should trap such errors in build_translations(), not in translate()
    assert_raises(NameError,
                  M.translate,
                  {'a': 23.4, 'b': 34.5})

def test_translate_with_divide_by_zero_error():
    M = StandardModelType
    M.default_parameters = {'a': 22.2, 'b': 33.3}
    M.translations = build_translations(
            ('a', 'A'),
            ('b', 'B', 'b/0', 'B*0'),
    )
    assert_raises(ZeroDivisionError,
                  M.translate, {'a': 23.4, 'b': 34.5})

def test_reverse_translate():
    M = StandardModelType
    M.default_parameters = {'a': 22.2, 'b': 33.3, 'c': 44.4}
    M.translations = build_translations(
            ('a', 'A'),
            ('b', 'B', 1000.0),
            ('c', 'C', 'c + a', 'C - A'),
        )
    assert_equal(M.reverse_translate({'A': 23.4, 'B': 34500.0, 'C': 69.0}),
                 {'a': 23.4, 'b': 34.5, 'c': 45.6})

def test_reverse_translate_with_invalid_transformation():
    M = StandardModelType
    M.translations = build_translations(
            ('a', 'A'),
            ('b', 'B', 'b + z', 'B-Z'),
    )
    M.default_parameters = {'a': 22.2, 'b': 33.3}
    #really we should trap such errors in build_translations(), not in reverse_translate()
    assert_raises(NameError,
                  M.reverse_translate,
                  {'A': 23.4, 'B': 34.5})

def test_simple_parameters():
    M = StandardModelType
    M.default_parameters = {'a': 22.2, 'b': 33.3, 'c': 44.4}
    M.translations = build_translations(
            ('a', 'A'),
            ('b', 'B', 1000.0),
            ('c', 'C', 'c + a', 'C - A'),
        )
    assert_equal(M.simple_parameters(), ['a'])

def test_scaled_parameters():
    M = StandardModelType
    M.default_parameters = {'a': 22.2, 'b': 33.3, 'c': 44.4}
    M.translations = build_translations(
            ('a', 'A'),
            ('b', 'B', 1000.0),
            ('c', 'C', 'c + a', 'C - A'),
        )
    assert_equal(M.scaled_parameters(), ['b'])

def test_computed_parameters():
    M = StandardModelType
    M.default_parameters = {'a': 22.2, 'b': 33.3, 'c': 44.4}
    M.translations = build_translations(
            ('a', 'A'),
            ('b', 'B', 1000.0),
            ('c', 'C', 'c + a', 'C - A'),
        )
    assert_equal(M.computed_parameters(), ['c'])

def test_update_parameters():
    """
    update_parameters(P) should update self.parameters with the value of
    translate(P)
    """
    M = StandardModelType
    orig_check_parameters = M.check_parameters
    orig_translate = M.translate
    M.check_parameters = Mock()
    P = {'a': 3}
    M.translate = Mock(return_value=P)
    m = M({})
    m.parameters = Mock()
    m.update_parameters({})
    m.parameters.update.assert_called_with(P)
    M.check_parameters = orig_check_parameters
    M.translate = orig_translate

def test_describe():
    M = StandardModelType
    M.default_parameters = {'a': 22.2, 'b': 33.3, 'c': 44.4}
    M.translations = build_translations(
            ('a', 'A'),
            ('b', 'B', 1000.0),
            ('c', 'C', 'c + a', 'C - A'),
        )
    m = M({})
    assert isinstance(m.describe(), basestring)

# test StandardCellType

## test SynapseDynamics

# test create

def test_describe_SD():
    sd = SynapseDynamics()
    assert isinstance(sd.describe(), basestring)
    assert isinstance(sd.describe(template=None), dict)

## test ShortTermPlasticityMechanism

def test_STDPMechanism_create():
    STDPTimingDependence.__init__ = Mock(return_value=None)
    STDPWeightDependence.__init__ = Mock(return_value=None)
    td = STDPTimingDependence()
    wd = STDPWeightDependence()
    stdp = STDPMechanism(td, wd, None, 0.5)
    assert_equal(stdp.timing_dependence, td)
    assert_equal(stdp.weight_dependence, wd)
    assert_equal(stdp.voltage_dependence, None)
    assert_equal(stdp.dendritic_delay_fraction, 0.5)

def test_STDPMechanism_create_invalid_types():
    assert_raises(AssertionError, # probably want a more informative error
                  STDPMechanism, timing_dependence="abc")
    assert_raises(AssertionError, # probably want a more informative error
                  STDPMechanism, weight_dependence="abc")
    assert_raises(AssertionError, # probably want a more informative error
                  STDPMechanism, dendritic_delay_fraction = "abc")
    assert_raises(AssertionError, # probably want a more informative error
                  STDPMechanism, dendritic_delay_fraction = "1.1")


## test STDPWeightDependence

## test STDPTimingDependence

