# -*- coding: utf-8 -*-
"""
Support for neuron and synapse models defined in NESTML.

NESTML models must be registered before calling sim.setup(). setup() triggers a
single compilation pass (via PyNESTML) covering all registered models, then installs
the resulting NEST module. After setup() the returned classes behave identically to
those returned by native_cell_type() / native_synapse_type().

Functions:
    nestml_cell_type    - register a NESTML neuron description; return a cell type class
    nestml_synapse_type - register a NESTML synapse description; return a synapse type class

:copyright: Copyright 2006-2024 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import logging
import os.path
import re
import shutil
import tempfile
import uuid

from pyNN.nest.cells import native_cell_type
from pyNN.models import BaseCellType, BaseSynapseType
from pyNN.nest.synapses import NESTSynapseMixin
import pyNN

logger = logging.getLogger("PyNN")

_MODULE_NAME = "pynnnestmlmodule"

# Module-level registry — survives state.clear() so definitions persist across setup() calls
_pending = []    # list of entry dicts, one per registered model
_compiled = False  # True after the first successful compile + install


def _check_not_compiled():
    if _compiled:
        raise RuntimeError(
            "NESTML models must be registered before sim.setup() is called. "
            "Cannot add new NESTML models after compilation."
        )


def _make_pending_cell_type_class(name):
    """Return a stub cell type class that will be resolved at setup() time."""

    class NESTMLCellType(BaseCellType):
        _pending = True
        _nestml_name = name
        default_parameters = {}
        default_initial_values = {}
        nest_name = {"on_grid": name, "off_grid": name}
        recordable = []
        injectable = True
        uses_parrot = False

        def __init__(self, **parameters):
            self._pending_parameters = parameters

        def get_schema(self):
            return {n: type(v) for n, v in self.default_parameters.items()}

        @classmethod
        def _resolve(cls, nest_name=None):
            real = native_cell_type(nest_name or cls._nestml_name)
            for attr in ("nest_name", "nest_model", "default_parameters", "default_initial_values",
                         "recordable", "injectable", "uses_parrot", "receptor_types",
                         "standard_receptor_type", "conductance_based", "always_local", "units"):
                if hasattr(real, attr):
                    setattr(cls, attr, getattr(real, attr))
            cls._pending = False
            cls.__init__ = BaseCellType.__init__

    NESTMLCellType.__name__ = name
    NESTMLCellType.__qualname__ = name
    return NESTMLCellType


def nestml_cell_type(name, nestml_description):
    """
    Register a NESTML neuron description and return a cell type class.

    ``nestml_description`` may be a path to a .nestml file or a string containing
    NESTML source code.

    The returned class is a stub until sim.setup() is called. setup() compiles all
    registered NESTML models together in a single PyNESTML invocation and installs
    the resulting NEST module. After setup() the class behaves identically to one
    returned by native_cell_type().
    """
    _check_not_compiled()
    stub = _make_pending_cell_type_class(name)
    _pending.append({"type": "neuron", "name": name, "description": nestml_description, "stub": stub})
    return stub


def _get_model_name(filename: str) -> str:
    """Extract the model name from a NESTML source file."""
    with open(filename, "r") as fp:
        nestml_model = fp.read()
    return re.findall(r"model [^:\s]*:", nestml_model)[0][6:-1]


def nestml_synapse_type(
        synapse_name,
        nestml_description,
        postsynaptic_neuron_nestml_description=None,
        weight_variable="w",
        delay_variable=None
):
    """
    Register a NESTML synapse description and return a synapse type class.

    ``nestml_description`` and (optionally) ``postsynaptic_neuron_nestml_description``
    may each be a path to a .nestml file or a string containing NESTML source code.

    For plastic synapses that require co-generation with a postsynaptic neuron model,
    provide ``postsynaptic_neuron_nestml_description``. The co-generated neuron class
    is then accessible as ``synapse_class.postsynaptic_cell_type`` both before and
    after sim.setup().

    The returned class is a stub until sim.setup() triggers compilation (see
    nestml_cell_type() for details).
    """
    _check_not_compiled()

    # For co-generation, create the neuron stub now so callers can hold a reference to it
    # before setup() is called. It will be resolved during _compile_and_resolve().
    neuron_stub = None
    if postsynaptic_neuron_nestml_description:
        neuron_stub = _make_pending_cell_type_class(synapse_name + "_postsynaptic_neuron")

    # Capture in local names to avoid same-name shadowing in the class body below.
    _delay_var = delay_variable
    _weight_var = weight_variable

    class NESTMLSynapseType(NESTSynapseMixin, BaseSynapseType):
        _pending = True
        _nestml_synapse_name = synapse_name
        default_parameters = {}
        nest_name = synapse_name
        delay_variable = _delay_var
        weight_variable = _weight_var

        def __init__(self, **parameters):
            self._pending_parameters = parameters

        def get_schema(self):
            return {n: type(v) for n, v in self.default_parameters.items()}

        @property
        def native_parameters(self):
            return self.parameter_space

        def get_native_names(self, *names):
            return names

        @classmethod
        def _resolve(cls, resolved_synapse_name, resolved_neuron_name=None):
            real = pyNN.nest.native_synapse_type(resolved_synapse_name)
            for attr in ("nest_name", "default_parameters"):
                setattr(cls, attr, getattr(real, attr))
            if resolved_neuron_name and hasattr(cls, 'postsynaptic_cell_type'):
                cls.postsynaptic_cell_type._resolve(resolved_neuron_name)
            cls._pending = False
            cls.__init__ = BaseSynapseType.__init__

    NESTMLSynapseType.__name__ = synapse_name
    NESTMLSynapseType.__qualname__ = synapse_name

    if neuron_stub is not None:
        NESTMLSynapseType.postsynaptic_cell_type = neuron_stub

    _pending.append({
        "type": "synapse",
        "name": synapse_name,
        "description": nestml_description,
        "neuron_description": postsynaptic_neuron_nestml_description,
        "weight_variable": weight_variable,
        "delay_variable": delay_variable,
        "stub": NESTMLSynapseType,
        "neuron_stub": neuron_stub,
    })
    return NESTMLSynapseType


def _ensure_file(description, tmpdirs):
    """
    Return a path to a .nestml file for the given description.

    If ``description`` is already a file path, return it unchanged. If it is inline
    NESTML source code, write it to a temporary file and return that path (the temp
    directory is appended to ``tmpdirs`` for later cleanup).
    """
    if os.path.isfile(description):
        return description
    tmpdir = tempfile.mkdtemp()
    tmpdirs.append(tmpdir)
    filepath = os.path.join(tmpdir, "model.nestml")
    with open(filepath, "w") as fp:
        fp.write(description)
    return filepath


def _compile_and_resolve():
    """
    Compile all pending NESTML models together and resolve their stub classes.

    Called by sim.setup() after the NEST kernel is initialised. If no models have
    been registered this is a no-op.
    """
    global _compiled, _pending

    if not _pending:
        _compiled = True
        return

    from pynestml.frontend.pynestml_frontend import generate_nest_target
    import nest

    tmpdirs = []
    input_paths = []
    codegen_opts = {}
    synapse_pairs = []  # tracks co-generation relationships for name resolution

    for entry in _pending:
        filepath = _ensure_file(entry["description"], tmpdirs)
        input_paths.append(filepath)

        if entry["type"] == "synapse":
            syn_model = _get_model_name(filepath)
            entry["_syn_model"] = syn_model  # actual NESTML model name, used for resolution
            codegen_opts.setdefault("synapse_models", []).append(syn_model)
            codegen_opts.setdefault("weight_variable", {})[syn_model] = entry["weight_variable"]
            if entry["delay_variable"] is not None:
                codegen_opts.setdefault("delay_variable", {})[syn_model] = entry["delay_variable"]

            if entry["neuron_description"]:
                neuron_filepath = _ensure_file(entry["neuron_description"], tmpdirs)
                input_paths.append(neuron_filepath)
                neu_model = _get_model_name(neuron_filepath)
                synapse_pairs.append({
                    "neuron": neu_model,
                    "synapse": syn_model,
                    "post_ports": ["post_spikes"],
                    "entry": entry,
                })

    if synapse_pairs:
        codegen_opts["neuron_synapse_pairs"] = [
            {"neuron": p["neuron"], "synapse": p["synapse"], "post_ports": p["post_ports"]}
            for p in synapse_pairs
        ]

    # Use a unique module name to prevent filename conflicts when multiple compilations
    # run concurrently (e.g. pytest-xdist parallel workers).
    module_name = "pynnnestml_" + uuid.uuid4().hex[:8] + "module"

    build_dir = tempfile.mkdtemp()
    tmpdirs.append(build_dir)
    try:
        generate_nest_target(
            input_path=input_paths,
            target_path=build_dir,
            install_path=None,
            module_name=module_name,
            codegen_opts=codegen_opts,
        )
        nest.Install(module_name)
    finally:
        for tmpdir in tmpdirs:
            shutil.rmtree(tmpdir, ignore_errors=True)

    for entry in _pending:
        stub = entry["stub"]
        if entry["type"] == "neuron":
            stub._resolve()
        else:
            pair = next((p for p in synapse_pairs if p["entry"] is entry), None)
            if pair:
                resolved_syn = pair["synapse"] + "__with_" + pair["neuron"]
                resolved_neu = pair["neuron"] + "__with_" + pair["synapse"]
                stub._resolve(resolved_syn, resolved_neu)
            else:
                stub._resolve(entry["_syn_model"])

    _compiled = True
    _pending.clear()
