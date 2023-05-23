# -*- coding: utf-8 -*-
"""
NEST v3 implementation of the PyNN API.

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

try:
    import tables  # noqa: F401 - due to freeze when importing nest before tables
except ImportError:
    pass

from .. import errors, random, space                                # noqa: F401
from ..network import Network                                       # noqa: F401
from ..space import Space                                           # noqa: F401
from ..standardmodels import StandardCellType
from ..random import NumpyRNG, GSLRNG, RandomDistribution           # noqa: F401
from .cells import NativeCellType, native_cell_type                 # noqa: F401
from .electrodes import NativeElectrodeType, native_electrode_type  # noqa: F401
from .synapses import NativeSynapseType, native_synapse_type        # noqa: F401
from .standardmodels.cells import *                                 # noqa: F403, F401
from .connectors import *                                           # noqa: F403, F401
from .standardmodels.synapses import *                              # noqa: F403, F401
from .standardmodels.electrodes import *                            # noqa: F403, F401
from .standardmodels.receptors import *                             # noqa: F403, F401
from .recording import *                                            # noqa: F403, F401
from .random import NativeRNG                                       # noqa: F401
from .populations import Population, PopulationView, Assembly       # noqa: F401
from .projections import Projection                                 # noqa: F401
from .control import (                                              # noqa: F401
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
from .procedural_api import create, connect, record, record_v, record_gsyn, set  # noqa: F401


# ==============================================================================
#   Utility functions
# ==============================================================================


def list_standard_models():
    """Return a list of all the StandardCellType classes available for this simulator."""
    standard_cell_types = [obj for obj in globals().values() if isinstance(
        obj, type) and issubclass(obj, StandardCellType) and obj is not StandardCellType]
    for cell_class in standard_cell_types:
        try:
            create(cell_class())
        except Exception as e:
            print("Warning: %s is defined, but produces the following error: %s" %
                  (cell_class.__name__, e))
            standard_cell_types.remove(cell_class)
    return [obj.__name__ for obj in standard_cell_types]
