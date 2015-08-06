# encoding: utf-8
"""
Alternative, procedural API for creating, connecting and recording from individual neurons

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

from .populations import IDMixin, BasePopulation, Assembly


def build_create(population_class):
    def create(cellclass, cellparams=None, n=1):
        """
        Create `n` cells all of the same type.

        Returns a Population object.
        """
        return population_class(n, cellclass, cellparams=cellparams)
    return create


def build_connect(projection_class, connector_class, static_synapse_class):
    def connect(pre, post, weight=0.0, delay=None, receptor_type=None,
                p=1, rng=None):
        """
        Connect a source of spikes to a synaptic target.

        `source` and `target` can both be individual cells or populations/
        assemblies of cells, in which case all possible connections are made
        with probability `p`, using either the random number generator supplied,
        or the default RNG otherwise. Weights should be in nA or µS.
        """
        if isinstance(pre, IDMixin):
            pre = pre.as_view()
        if isinstance(post, IDMixin):
            post = post.as_view()
        connector = connector_class(p_connect=p, rng=rng)
        synapse = static_synapse_class(weight=weight, delay=delay)
        return projection_class(pre, post, connector, receptor_type=receptor_type,
                                synapse_type=synapse)
    return connect


def set(cells, **parameters):
    """
    Set one or more parameters for every cell in a population, view or assembly.

    Values passed to set() may be:
        (1) single values
        (2) RandomDistribution objects
        (3) mapping functions, where a mapping function accepts a single
            argument (the cell index) and returns a single value.

    Here, a "single value" may be either a single number or a list/array of
    numbers (e.g. for spike times). Values should be expressed in the standard
    PyNN units (i.e. millivolts, nanoamps, milliseconds, microsiemens,
    nanofarads, event per second).
    """
    assert isinstance(cells, (BasePopulation, Assembly))
    cells.set(**parameters)


def build_record(simulator):
    def record(variables, source, filename, sampling_interval=None, annotations=None):
        """
        Record variables to a file. source can be an individual cell, a
        Population, PopulationView or Assembly.
        """
        # would actually like to be able to record to an array and choose later
        # whether to write to a file.
        if not isinstance(source, (BasePopulation, Assembly)):
            source = source.parent
        source.record(variables, to_file=filename, sampling_interval=sampling_interval)
        if annotations:
            source.annotate(**annotations)
        simulator.state.write_on_end.append((source, variables, filename))
    return record


def initialize(cells, **initial_values):
    """
    Set initial values of state variables, e.g. the membrane potential.

    Values passed to initialize() may be:
        (1) single numeric values (all neurons set to the same value)
        (2) RandomDistribution objects
        (3) lists/arrays of numbers of the same size as the population
        (4) mapping functions, where a mapping function accepts a single
            argument (the cell index) and returns a single number.

    Values should be expressed in the standard PyNN units (i.e. millivolts,
    nanoamps, milliseconds, microsiemens, nanofarads, event per second).

    Examples::

        initialize(cells, v=-70.0)
        initialize(cells, v=rand_distr, gsyn_exc=0.0)
        initialize(cells, v=lambda i: -65 + i/10.0)
    """
    assert isinstance(cells, (BasePopulation, Assembly)), type(cells)
    cells.initialize(**initial_values)
