"""
Export of PyNN scripts as NineML.

:copyright: Copyright 2006-2013 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import numpy
from pyNN import recording
from . import simulator


class Recorder(recording.Recorder):
    _simulator = simulator

    def _record(self, variable, new_ids):
        pass

    def get(self, variables, gather=False, filter_ids=None, clear=False,
            annotations=None):
        pass
    
    def write(self, variables, file=None, gather=False, filter_ids=None,
              clear=False, annotations=None):
        pass

    def _local_count(self, variable, filter_ids=None):
        return {}
