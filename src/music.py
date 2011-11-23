"""
This module supports running simulations with multiple, interacting simulators
at once, provided those simulators support the MUSIC communication interface.


:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""


def setup(configurations):
    """
    Specify PyNN module ranks and external MUSIC programs to be launched.
    """
    # Need Ctypes wrapper to libmusic pre-MPI-init rank assigner
    raise NotImplementedError


def get_simulator(simulator_name):
    """
    Return either the real PyNN module for the requested backend simulator (if
    that simulator is running on the current MPI node) or a `ProxySimulator`
    object (if the requested simulator is not running on the current node).
    """
    raise NotImplementedError


def run(simtime):
    """Run the simulation for simtime ms."""
    #  finally setup delayed MUSIC port setup,
    # work through back-end specific eventport setup,
    # call run for backend assigned to this rank.
    raise NotImplementedError


class Projection(object): # may wish to inherit from common.projections.Projection
    """
    docstring goes here
    """
    #queues projections and port configs to be evaluated in music.run
    pass


class Port(object): # aka Pipe aka DataLink # other name suggestions welcome
    """
    Representation of a data connection from a PyNN Population to an external
    MUSIC program (data consumer) such as a visualization tool.
    """

    def __init__(self, population, variable, external_tool):
        raise NotImplementedError


class ProxySimulator(object):
    """
    A proxy for a real simulator backend, which has the same API but doesn't
    actually run simulations.
    """
    # this may be of use outside of pyNN.music, but for simplicity I suggest
    # we develop it here for now and then think about moving/generalizing it
    # later
    pass
