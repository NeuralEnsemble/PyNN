"""
Definition of default parameters (and hence, standard parameter names) for
standard current source models.

:copyright: Copyright 2006-2015 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

from pyNN.standardmodels import StandardCurrentSource
from pyNN.parameters import Sequence

class DCSource(StandardCurrentSource):
    """Source producing a single pulse of current of constant amplitude.

    Arguments:
        `start`:
            onset time of pulse in ms
        `stop`:
            end of pulse in ms
        `amplitude`:
            pulse amplitude in nA
    """

    default_parameters = {
        'amplitude'     : 1.0,       #
        'start'         : 0.0,      #
        'stop'          : 1e12,  #
    }


class ACSource(StandardCurrentSource):
    """Source producing a single pulse of current of constant amplitude.

    Arguments:
        `start`:
            onset time of pulse in ms
        `stop`:
            end of pulse in ms
        `amplitude`:
            sine amplitude in nA
        `offset`:
            sine offset in nA
        `frequency`:
            frequency in Hz
        `phase`:
            phase in degrees
    """

    default_parameters = {
        'amplitude'     : 1.0,      #
        'start'         : 0.0,     #
        'stop'          : 1e12, #
        'frequency'     : 10.,
        'offset'        : 0.,
        'phase'         : 0.
    }


class StepCurrentSource(StandardCurrentSource):
    """A step-wise time-varying current source.

    Arguments:
        `times`:
            list/array of times at which the injected current changes.
        `amplitudes`:
            list/array of current amplitudes to be injected at the times
            specified in `times`.

    The injected current will be zero up until the first time in `times`. The
    current will continue at the final value in `amplitudes` until the end
    of the simulation.
    """

    default_parameters = {
        'amplitudes': Sequence([]),
        'times'     : Sequence([])
    }

class NoisyCurrentSource(StandardCurrentSource):
    """A Gaussian "white" noise current source. The current amplitude changes at fixed
    intervals, with the new value drawn from a Gaussian distribution.

    Required arguments:
        `mean`:
            mean current amplitude in nA
        `stdev`:
            standard deviation of the current amplitude in nA

    Optional arguments:
        `dt`:
            interval between updates of the current amplitude. Must be a
            multiple of the simulation time step. If not specified, the
            simulation time step will be used.
        `start`:
            onset of the current injection in ms. If not specified, the current
            will begin at the start of the simulation.
        `stop`:
            end of the current injection in ms. If not specified, the current
            will continue until the end of the simulation.
        `rng`:
            an RNG object from the `pyNN.random` module. For speed, this should
            be a `NativeRNG` instance (uses the simulator's internal random
            number generator). For reproducibility across simulators, use one of
            the other RNG types. If not specified, a NumpyRNG is used.
     """

    default_parameters = {
        'mean'           : 0.,
        'stdev'          : 1.,
        'start'          : 0.,
        'stop'           : 1e12,
        'dt'             : 0.1
    }
