# -*- coding: utf-8 -*-
"""
auxiliary functions to look for the hardware backend.

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

from pkgutil import iter_modules
from os import environ, path

# ==============================================================================
# getters
# ==============================================================================


def get_symap2ic_path():
    if 'SYMAP2IC_PATH' not in environ:
        raise ImportError(
            """
            symap2ic software is not available!
            - is symap2ic installed?
            - is environment variable SYMAP2IC_PATH set?""")
    symap2ic_path = environ['SYMAP2IC_PATH']
    if not path.exists(symap2ic_path):
        raise ImportError(
            """
            SYMAP2IC_PATH = %s
            SYMAP2IC_PATH points to a non existing directory"""
            % symap2ic_path)
    return symap2ic_path


def get_pynn_hw_path():
    hardware_path = environ['PYNN_HW_PATH']
    if not path.exists(hardware_path):
        raise Exception(
            """
            PYNN_HW_PATH = %s
            You are using PYNN_HW_PATH to point to the PyNN hardware backend.
            But, PYNN_HW_PATH points to a non existing directory"""
            % hardware_path)
    return hardware_path


def get_hardware_path(symap2ic_path):
    hardware_path = path.join(symap2ic_path, "components/pynnhw/src/hardware")
    if not path.exists(hardware_path):
        raise Exception(
            """
            hardware_path = %s
            It should point to the PyNN hardware backend
            But, hardware_path points to a non existing directory"""
            % hardware_path)
    return hardware_path

# ==============================================================================
# Utility functions
# ==============================================================================


def import_module(version="brainscales"):
    __import__("brainscales", globals(), locals(), [], -1)

# ==============================================================================
# Functions called by __init__.py
# ==============================================================================


def get_path_to_analog_hardware_backend():
    symap2ic_path = get_symap2ic_path()
    if 'PYNN_HW_PATH' in environ:
        hardware_path = get_pynn_hw_path()
    else:
        hardware_path = get_hardware_path(symap2ic_path)

    return hardware_path


def import_all_submodules(module_path):
    for importer, module_name, ispkg in iter_modules(module_path):
        if ispkg is True:
            import_module(version=module_name)
            print("Linked: submodule hardware.%s" % module_name)
