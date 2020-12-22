# encoding: utf-8
from testconfig import config


if 'testFile' in config:
    file_name = config['testFile']
    exec("from . import ( %s )" % file_name)
else:
    from . import (scenario1,
                   scenario2,
                   scenario3,
                   ticket166,
                   test_simulation_control,
                   test_recording,
                   test_cell_types,
                   test_electrodes,
                   scenario4,
                   test_parameter_handling,
                   test_procedural_api,
                   issue274,
                   test_connectors,
                   issue231,
                   test_connection_handling,
                   test_synapse_types)
