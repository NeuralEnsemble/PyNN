# -*- coding: utf-8 -*-
"""
hardware implementation of the PyNN API.
It includes the submodules that stand on another directory.
This solution is a clean way to make the submodules (brainscales, etc...)
be indeed submodules of hardware, even if they don't stand on the same directory

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

from auxiliary import get_path_to_analog_hardware_backend, import_all_submodules

__path__.append(get_path_to_analog_hardware_backend())
import_all_submodules(__path__)
