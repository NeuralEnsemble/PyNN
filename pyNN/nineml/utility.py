# encoding: utf-8
"""

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

from lazyarray import larray
from pyNN import random
from pyNN.parameters import Sequence
import nineml.user as nineml

# catalog_url = "http://svn.incf.org/svn/nineml/catalog"
# catalog_url = join(dirname(__file__), "catalog")
catalog_url = "catalog"


units_map = {
    # arguably we should do the units mapping with the PyNN names, i.e. before translation.
    "time": "ms",
    "potential": "mV",
    "threshold": "mV",
    "capacitance": "nF",
    "frequency": "Hz",
    "duration": "ms",
    "onset": "ms",
    "amplitude": "nA",  # dodgy. Probably better to include units with class definitions
    "weight": "dimensionless",
    "delay": "ms",
    "dx": u"µm", "dy": u"µm", "dz": u"µm",
    "x0": u"µm", "y0": u"µm", "z0": u"µm",
    "aspectratio": "dimensionless",
}


def reverse_map(D):
    """
    Return a dict having D.values() as its keys and D.keys() as its values.
    """
    E = {}
    for k, v in D.items():
        if v in E:
            raise KeyError(
                "Cannot reverse this mapping, as it is not one-to-one "
                f"('{v}' would map to both '{E[v]}' and '{k}')"
            )
        E[v] = k
    return E


def infer_units(parameter_name):
    unit = "unknown"
    for fragment, u in units_map.items():
        if fragment in parameter_name.lower():
            unit = u
            break
    return unit


random_distribution_url_map = {
    'uniform': 'http://www.uncertml.org/distributions/uniform',
    'normal': 'http://www.uncertml.org/distributions/normal',
    'exponential': 'http://www.uncertml.org/distributions/exponential',
}


random_distribution_parameter_map = {
    'normal': ('mean', 'variance'),
    'uniform': ('minimum', 'maximum'),
    'exponential': ('rate',),    # need to translate - see hack below
}


def build_random_distribution(random_distribution_component):
    """
    Given a NineML random distribution component, return a PyNN RandomDistribution object.
    """
    rd = random_distribution_component
    rd_name = reverse_map(random_distribution_url_map)[rd.standard_library]
    rd_param_names = random_distribution_parameter_map[rd_name]
    rd_params = [rd.property(rdp_name).value for rdp_name in rd_param_names]
    if rd_name == 'exponential':
        # temporary hack - need to implement a proper translation mechanism
        # UncertML uses rate in Hz, PyNN uses beta (1/rate) in ms
        rd_params[0] = 1000.0 / rd_params[0]
    rand_distr = random.RandomDistribution(rd_name, rd_params)
    return rand_distr


def build_parameter_set(parameters, shape=None, dimensionless=False):
    parameter_list = []
    for name, value in parameters.items():
        if isinstance(value, larray):
            value.shape = shape
            if value.is_homogeneous:
                value = value.evaluate(simplify=True)
                if isinstance(value, Sequence):
                    value = value.value.tolist()
                elif isinstance(value, bool):
                    value = int(value)
            elif isinstance(value, random.RandomDistribution):
                rand_distr = value
                value = nineml.RandomDistributionComponent(
                    name="%s(%s)" % (rand_distr.name, ",".join(str(p)
                                                               for p in rand_distr.parameters)),
                    definition=nineml.Definition(random_distribution_url_map[rand_distr.name],
                                                 "random"),
                    parameters=build_parameter_set(
                        map_random_distribution_parameters(rand_distr.name, rand_distr.parameters),
                        dimensionless=True))
            else:
                raise Exception("not supported")
        else:
            if isinstance(value, bool):
                value = int(value)
        if dimensionless:
            unit = "dimensionless"
        elif isinstance(value, str):
            unit = None
        else:
            unit = infer_units(name)
        parameter_list.append(nineml.Property(name, value, unit))
    return nineml.PropertySet(*parameter_list)


def map_random_distribution_parameters(name, parameters):
    parameter_map = random_distribution_parameter_map
    P = {}
    for name, val in zip(parameter_map[name], parameters):
        P[name] = val
    return P
