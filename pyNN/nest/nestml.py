# -*- coding: utf-8 -*-
"""
Support cell types defined in NESTML.


Classes:
    NESTMLCellType   - base class for cell types, not used directly

Functions:
    nestml_cell_type - return a new NESTMLCellType subclass


:copyright: Copyright 2006-2024 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import logging
import os.path
import re
import shutil
import tempfile

from pyNN.nest.cells import native_cell_type
import pyNN

logger = logging.getLogger("PyNN")



def nestml_cell_type(name, nestml_description):
    """
    Return a new NESTMLCellType subclass from a NESTML description.
    """

    from pynestml.frontend.pynestml_frontend import generate_nest_target
    import nest

    module_name = "pynnnestmlmodule"  # todo: customize this
    if os.path.exists(nestml_description):
        # description is a file path
        input_path = nestml_description
        have_tmpdir = False
    else:
        # assume description is a string containing nestml code
        input_path = tempfile.mkdtemp()
        with open(os.path.join(input_path, "tmp.nestml"), "w") as fp:
            fp.write(nestml_description)
        have_tmpdir = True
    generate_nest_target(
        input_path=input_path,
        target_path=None,  # "/tmp/nestml_target",
        install_path=None,
        module_name=module_name,
    )
    if have_tmpdir:
        shutil.rmtree(input_path)

    nest.Install(module_name)

    # todo: get units information from nestml_description, provide to "native_cell_type()"
    return native_cell_type(name)

def _get_model_name(filename: str) -> str:
    """Get the model filename from a NESTML model file"""
    with open(filename, "r") as fp:
        nestml_model = fp.read()
        model_name = re.findall(r"model [^:\s]*:", nestml_model)[0][6:-1]

    return model_name

def nestml_synapse_type(synapse_name, nestml_description, postsynaptic_neuron_nestml_description=None, weight_variable="w", delay_variable="d"):
    """
    Return a new NESTMLCellType subclass from a NESTML description.

    For plastic synapses, the synapse code needs to be generated in close combination with the postsynaptic neuron code. For this reason, for plastic synapses, it is necessary to specify the ``postsynaptic_neuron_nestml_description``.
    """

    from pynestml.frontend.pynestml_frontend import generate_nest_target
    import nest

    module_name = "pynnnestmlmodule"  # todo: customize this

    input_path = []
    codegen_opts = {}

    # write synapse model to file if necessary
    if os.path.exists(nestml_description):
        # description is a file path
        input_path.append(nestml_description)
        synapse_filename = nestml_description
        have_synapse_tmpdir = False
    else:
        # assume description is a string containing nestml code
        synapse_input_path = tempfile.mkdtemp()
        synapse_filename = os.path.join(synapse_input_path, "tmp.nestml")
        input_path.append(synapse_input_path)
        with open(synapse_filename, "w") as fp:
            fp.write(nestml_description)
        have_synapse_tmpdir = True

    codegen_opts["weight_variable"] = {}
    codegen_opts["weight_variable"][_get_model_name(synapse_filename)] = weight_variable
    codegen_opts["delay_variable"] = {}
    codegen_opts["delay_variable"][_get_model_name(synapse_filename)] = delay_variable

    if postsynaptic_neuron_nestml_description:
        # write neuron model to file if necessary
        if os.path.exists(postsynaptic_neuron_nestml_description):
            # description is a file path
            input_path.append(postsynaptic_neuron_nestml_description)
            neuron_filename = postsynaptic_neuron_nestml_description
            have_neuron_tmpdir = False
        else:
            # assume description is a string containing nestml code
            neuron_input_path = tempfile.mkdtemp()
            neuron_filename = os.path.join(neuron_input_path, "tmp.nestml")
            input_path.append(neuron_input_path)
            with open(neuron_filename, "w") as fp:
                fp.write(postsynaptic_neuron_nestml_description)
            have_neuron_tmpdir = True

        # specify co-generation of synapse and postsynaptic neuron code in the codegen_opts dictionary
        codegen_opts["neuron_synapse_pairs"] = [{"neuron": _get_model_name(neuron_filename),
                                                 "synapse": _get_model_name(synapse_filename),
                                                 "post_ports": ["post_spikes"]}]

        neuron_name = _get_model_name(neuron_filename) + "__with_" + _get_model_name(synapse_filename)
        synapse_name = _get_model_name(synapse_filename) + "__with_" + _get_model_name(neuron_filename)

    generate_nest_target(
        input_path=input_path,
        target_path=None,  # "/tmp/nestml_target",
        install_path=None,
        module_name=module_name,
        codegen_opts=codegen_opts
    )

    if have_synapse_tmpdir:
        shutil.rmtree(synapse_input_path)

    if postsynaptic_neuron_nestml_description and have_neuron_tmpdir:
        shutil.rmtree(neuron_input_path)

    nest.Install(module_name)
    import pdb;pdb.set_trace()

    # todo: get units information from nestml_description, provide to "native_cell_type()"
    if postsynaptic_neuron_nestml_description:
        synapse_instance = pyNN.nest.native_synapse_type(synapse_name)
        synapse_instance.delay_variable = delay_variable
        synapse_instance.weight_variable = weight_variable
        return synapse_instance, pyNN.nest.native_cell_type(neuron_name)

    return pyNN.nest.native_synapse_type(synapse_name)
