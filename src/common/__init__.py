# encoding: utf-8
"""
Defines a common implementation of the PyNN API.

Simulator modules are not required to use any of the code herein, provided they
provide the correct interface, but it is suggested that they use as much as is
consistent with good performance (optimisations may require overriding some of
the default definitions given here).

Utility functions and classes:
    is_conductance()
    check_weight()
    check_delay()

Accessing individual neurons:
    IDMixin

Common API implementation/base classes:
  1. Simulation set-up and control:
    setup()
    end()
    run()
    reset()
    get_time_step()
    get_current_time()
    get_min_delay()
    get_max_delay()
    rank()
    num_processes()

  2. Creating, connecting and recording from individual neurons:
    create()
    connect()
    set()
    initialize()
    build_record()

  3. Creating, connecting and recording from populations of neurons:
    Population
    PopulationView
    Assembly
    Projection

:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

$Id: $
"""

from populations import IDMixin, BasePopulation, Population, PopulationView, Assembly, is_conductance
from projections import Projection, check_weight, DEFAULT_WEIGHT
from procedural_api import build_create, build_connect, set, build_record, initialize
from control import setup, end, run, build_reset, build_state_queries
