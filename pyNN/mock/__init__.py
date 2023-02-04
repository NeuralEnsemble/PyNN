"""
Mock implementation of the PyNN API, for testing and documentation purposes.

This simulator implements the PyNN API, but generates random data rather than
really running simulations.

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""


from .. import errors, random, space                            # noqa: F401
from ..network import Network                                   # noqa: F401
from ..space import Space                                       # noqa: F401
from ..random import NumpyRNG, GSLRNG, RandomDistribution       # noqa: F401
from ..connectors import *                                      # noqa: F403, F401
from ..recording import *                                       # noqa: F403, F401
from ..standardmodels import StandardCellType
from .standardmodels import *                                   # noqa: F403, F401
from .populations import Population, PopulationView, Assembly   # noqa: F401
from .projections import Projection                             # noqa: F401
from .control import (                                          # noqa: F401
    setup,
    end,
    run,
    run_until,
    run_for,
    reset,
    initialize,
    get_current_time,
    get_time_step,
    get_min_delay,
    get_max_delay,
    num_processes,
    rank,
)
from .procedural_api import create, connect, record, record_v, record_gsyn  # noqa: F401


def list_standard_models():
    """Return a list of all the StandardCellType classes available for this simulator."""
    return [obj.__name__ for obj in globals().values()
            if isinstance(obj, type) and issubclass(obj, StandardCellType)]
