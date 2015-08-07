# encoding: utf-8
"""
Defines a backend-independent, partial implementation of the PyNN API

Backend simulator modules are not required to use any of the code herein,
provided they provide the correct interface, but it is suggested that they use
as much as is consistent with good performance (optimisations may require
overriding some of the default definitions given here).

Utility functions and classes:
    is_conductance()
    check_weight()

Base classes to be sub-classed by individual backends:
    IDMixin
    Population
    PopulationView
    Assembly
    Projection
    
Function-factories to generate backend-specific API functions:
    build_reset()
    build_state_queries()
    build_create()
    build_connect()
    build_record()
    
Common implementation of API functions:
    set()
    initialize()

Function skeletons to be extended by backends:
    setup()
    end()
    run()

Global constants:
    DEFAULT_MAX_DELAY
    DEFAULT_TIMESTEP
    DEFAULT_MIN_DELAY

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

from .populations import IDMixin, BasePopulation, Population, PopulationView, Assembly, is_conductance
from .projections import Projection, Connection
from .procedural_api import build_create, build_connect, set, build_record, initialize
from .control import setup, end, build_run, build_reset, build_state_queries
