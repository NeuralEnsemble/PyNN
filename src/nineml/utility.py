# encoding: utf-8
"""

:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

from pyNN import random
import nineml.user_layer as nineml

#catalog_url = "http://svn.incf.org/svn/nineml/catalog"
catalog_url = "https://neuralensemble.org/svn/PyNN/trunk/src/nineml/catalog"

units_map = { # arguably we should do the units mapping with the PyNN names, i.e. before translation.
    "time": "ms",
    "potential": "mV",
    "threshold": "mV",
    "capacitance": "nF",
    "frequency": "Hz",
    "duration": "ms",
    "onset": "ms",
    "amplitude": "nA", # dodgy. Probably better to include units with class definitions
    "weight": "dimensionless",
    "delay": "ms",
    "dx": u"µm", "dy": u"µm", "dz": u"µm",
    "x0": u"µm", "y0": u"µm", "z0": u"µm",
    "aspectratio": "dimensionless",
}

def infer_units(parameter_name):
    unit = "unknown"
    for fragment, u in units_map.items():
        if fragment in parameter_name.lower():
            unit = u
            break
    return unit

random_distribution_url_map = {
    'uniform': "%s/randomdistributions/uniform_distribution.xml" % catalog_url,
    'normal': "%s/randomdistributions/normal_distribution.xml" % catalog_url,
}

random_distribution_parameter_map = {
    'normal': ('mean', 'standardDeviation'),
    'uniform': ('lowerBound', 'upperBound'),
}

def build_parameter_set(parameters, dimensionless=False):
    parameter_list = []
    for name, value in parameters.items():
        if isinstance(value, random.RandomDistribution):
            rand_distr = value
            value = nineml.RandomDistribution(
                name="%s(%s)" % (rand_distr.name, ",".join(str(p) for p in rand_distr.parameters)),
                definition=nineml.Definition(random_distribution_url_map[rand_distr.name]),
                parameters=build_parameter_set(map_random_distribution_parameters(rand_distr.name, rand_distr.parameters),
                                               dimensionless=True))
        if dimensionless:
            unit = "dimensionless"
        elif isinstance(value, basestring):
            unit = None
        else:
            unit = infer_units(name)
        parameter_list.append(nineml.Parameter(name, value, unit))
    return nineml.ParameterSet(*parameter_list)

def map_random_distribution_parameters(name, parameters):
    parameter_map = random_distribution_parameter_map
    P = {}
    for name,val in zip(parameter_map[name], parameters):
        P[name] = val
    return P
