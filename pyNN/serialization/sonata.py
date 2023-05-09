# encoding: utf-8
"""
Support for the SONATA format (https://github.com/AllenInstitute/sonata/)

Public functions
----------------

- export_to_sonata()
- import_from_sonata()
- load_sonata_simulation_plan()

Usage
-----

This module would typically be used as follows::

    from pyNN.serialization import import_from_sonata, load_sonata_simulation_plan
    import pyNN.neuron as sim

    simulation_plan = load_sonata_simulation_plan("simulation_config.json")
    simulation_plan.setup(sim)
    net = import_from_sonata("circuit_config.json", sim)
    simulation_plan.execute(net)


:copyright: Copyright 2018 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import os
from os.path import join, isdir, exists
from collections import defaultdict
import shutil
from string import Template
import csv
from warnings import warn
import json
import logging

try:
    import h5py
    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False
import numpy as np

from ..network import Network
from ..parameters import Sequence


# Note: The SonataIO class will be moved to Neo once fully implemented

# from neo.io import SonataIO
import neo
from neo.io.baseio import BaseIO


class SonataIO(BaseIO):
    """
    Neo IO module for simulation input and output in the SONATA format.

    See https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md

    Currently only supports spike files.
    Support for current inputs and for reports will be implemented soon.
    """

    def __init__(self, base_dir,
                 spikes_file="spikes.h5",
                 spikes_sort_order=None,
                 report_config=None,
                 node_sets=None):
        if not HAVE_H5PY:
            raise Exception("You need to install h5py to use SonataIO")
        self.base_dir = base_dir
        self.spike_file = spikes_file
        self.spikes_sort_order = spikes_sort_order
        self.report_config = report_config
        self.node_sets = node_sets

    def read(self):
        """
        Read all data* from a SONATA dataset directory.

        Returns a list of Blocks.

        (*Currently only spike data supported)
        """
        file_path = join(self.base_dir, self.spike_file)
        block = neo.Block(file_origin=file_path)
        segment = neo.Segment(file_origin=file_path)
        spikes_file = h5py.File(file_path, 'r')
        for gid in np.unique(spikes_file['spikes']['gids']):
            index = spikes_file['spikes']['gids'][()] == gid
            spike_times = spikes_file['spikes']['timestamps'][index]
            segment.spiketrains.append(
                neo.SpikeTrain(spike_times,
                               t_stop=spike_times.max() + 1.0,
                               t_start=0.0,
                               units='ms',
                               source_id=gid)
            )
        block.segments.append(segment)
        return [block]

    def write(self, blocks):
        """
        Write a list of Blocks to SONATA HDF5 files.

        """
        if not os.path.isdir(self.base_dir):
            os.makedirs(self.base_dir)
        # Write spikes
        spike_file_path = join(self.base_dir, self.spike_file)
        spikes_file = h5py.File(spike_file_path, 'w')
        spike_trains = []
        for block in blocks:
            for segment in block.segments:
                spike_trains.extend(segment.spiketrains)

        spikes_group = spikes_file.create_group("spikes")
        all_spike_times = np.hstack(st.rescale('ms').magnitude
                                    for st in spike_trains).astype(np.float64)
        gids = np.hstack(st.annotations["source_index"] * np.ones(st.shape, dtype=np.uint64)
                         for st in spike_trains)
        # todo: handle sorting
        spikes_group.create_dataset("timestamps", data=all_spike_times, dtype=np.float64)
        spikes_group.create_dataset("gids", data=gids, dtype=np.uint64)
        spikes_file.close()
        logger.info("Wrote spike output to {}".format(spike_file_path))

        # Write signals
        for report_name, report_metadata in self.report_config.items():
            file_name = report_metadata.get("file_name", report_name + ".h5")
            file_path = join(self.base_dir, file_name)

            signal_file = h5py.File(file_path, 'w')
            targets = self.node_sets[report_metadata["cells"]]
            for block in blocks:
                for (assembly, mask) in targets:
                    if block.name == assembly.label:
                        if len(block.segments) > 1:
                            raise NotImplementedError()
                        signal = block.segments[0].filter(name=report_metadata["variable_name"])
                        if len(signal) != 1:
                            raise NotImplementedError()

                        node_ids = np.arange(assembly.size)[mask]

                        report_group = signal_file.create_group("report")
                        population_group = report_group.create_group(assembly.label)
                        dataset = population_group.create_dataset("data", data=signal[0].magnitude)
                        dataset.attrs["units"] = signal[0].units.dimensionality.string
                        dataset.attrs["variable_name"] = report_metadata["variable_name"]
                        n = dataset.shape[1]
                        mapping_group = population_group.create_group("mapping")
                        mapping_group.create_dataset("node_ids", data=node_ids)
                        # "gids" not in the spec, but expected by some bmtk utils
                        mapping_group.create_dataset("gids", data=node_ids)
                        # mapping_group.create_dataset("index_pointers", data=np.zeros((n,)))
                        mapping_group.create_dataset(
                            "index_pointer", data=np.arange(0, n+1))  # ??spec unclear
                        mapping_group.create_dataset("element_ids", data=np.zeros((n,)))
                        mapping_group.create_dataset("element_pos", data=np.zeros((n,)))
                        time_ds = mapping_group.create_dataset(
                            "time",
                            data=(
                                float(signal[0].t_start.rescale('ms')),
                                float(signal[0].t_stop.rescale('ms')),
                                float(signal[0].sampling_period.rescale('ms'))
                            ))
                        time_ds.attrs["units"] = "ms"
                        logger.info("Wrote block {} to {}".format(block.name, file_path))
            signal_file.close()


MAGIC = 0x0a7a
logger = logging.getLogger("pyNN.serialization.sonata")

# ----- utility functions, not intended for use outside this module ----------


def asciify(label):
    """To be safe, we will use only ascii for strings inside HDF5"""
    return label.replace(u"→", "-").encode('ascii')


def cast(value):
    """Try to cast strings to numeric values"""
    try:
        value = int(value)
    except ValueError:
        try:
            value = float(value)
        except ValueError:
            pass
    return value


def to_string(s):
    if isinstance(s, bytes):
        s = s.decode('utf-8')
    return s


def read_types_file(file_path, node_or_edge):
    """Read node type or edge type parameters from a SONATA CSV file.

    Returns a dict representing the parameters row-wise, indexed by the type id.
    """
    with open(file_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=' ', quotechar='"')
        types_table = list(csv_reader)
    types_map = {}
    id_label = "{}_type_id".format(node_or_edge)
    for row in types_table:
        types_map[int(row[id_label])] = row  # NodeType(**row)
    logger.info(types_map)
    return types_map


def sonata_id_to_index(population, id):
    # this relies on SONATA ids being sequential
    if "first_sonata_id" in population.annotations:
        offset = population.annotations["first_sonata_id"]
    else:
        raise Exception("Population not annotated with SONATA ids")
    return id - offset


def condense(value, types_array):
    """Transform parameters taken from SONATA CSV and/or HDF5 files
    into a suitable form for PyNN.

    Arguments
    ---------

    value - NumPy array or dict
        NumPy arrays are returned directly.
        Dicts should have node type ids as keys and the parameter values
        for the different types as values. Where all node/edge types have the same
        value, this single value is returned. Where different node/edge types have
        different values for a given parameter, a NumPy array of size equal
        to the number of nodes/edges in the SONATA group (cells in the PyNN Population
        or connections in the PyNN Projection) is constructed to contain the different
        parameter values.
    types_array - NumPy array
        Subset of the data from "/nodes/<population_name>/node_type_id" or
        from "/edges/<population_name>/edge_type_id" that applies to this group.
        Needed to construct parameter arrays.
    """
    # todo: use lazyarray
    if isinstance(value, np.ndarray):
        return value
    elif isinstance(value, dict):
        assert len(value) > 0
        value_array = np.array(list(value.values()))
        if np.all(value_array == value_array[0]):
            return value_array[0]
        else:
            if np.issubdtype(value_array.dtype, np.number):
                new_value = np.ones_like(types_array) * np.nan
            elif np.issubdtype(value_array.dtype, np.str_):
                new_value = np.array(["UNDEFINED"] * types_array.size)
            else:
                raise TypeError("Cannot handle annotations that are neither numbers or strings")
            for node_type_id, val in value.items():
                new_value[types_array == node_type_id] = val
            return new_value
    else:
        raise TypeError(
            "Unexpected type. Expected Numpy array or dict, got {}".format(type(value)))


def load_config(config_file):
    """Load a SONATA circuit or simulation config file

    This performs substitutions for the variables in the "manifest"
    section, then returns the config as a dict.
    """
    with open(config_file) as fp:
        config = json.load(fp)

    # Build substitutions from manifest
    substitutions = {}
    for key, value in config["manifest"].items():
        if key.startswith('$'):
            key = key[1:]
        substitutions[key] = value
    for key, value in substitutions.items():
        if '$' in value:
            substitutions[key] = Template(value).substitute(**substitutions)

    # Perform substitutions
    def traverse(obj):
        if isinstance(obj, dict):
            return {k: traverse(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [traverse(elem) for elem in obj]
        else:
            if isinstance(obj, str) and obj.startswith('$'):
                return Template(obj).substitute(**substitutions)
            else:
                return obj

    return traverse(config)


# ----- public functions -----------------------------------------------------

def export_to_sonata(network, output_path, target="PyNN", overwrite=False):
    """Export a PyNN network in SONATA format

    If `target` is "neuron" or "NEURON", then PyNN cell types are exported as
    single compartment neurons with NEURON .mod files, and only StaticSynapses are supported.

    If `target` is "PyNN", then the names of PyNN standard cells and synapse models
    are used in the exported files. This allows simulations with any PyNN-supported
    simulator.

    If `target` is "nest" or "NEST", then the names of NEST cell and synapse models
    are used in the exported files.


    A "node group" in SONATA corresponds approximately to a PyNN Population or PopulationView.
    A "node population" in SONATA corresponds approximately to a PyNN Assembly, except that
    node populations must be disjoint, whereas a given neuron may be in more than one
    PyNN Assembly (it is PyNN Populations that are disjoint).

    In this first version of the export, we simplify this by exporting each Population
    as an implicit Assembly with a single member, at the cost of losing some information about
    the PyNN network structure. This approach could be improved in future.
    """
    if not HAVE_H5PY:
        raise Exception("You need to install h5py to use SONATA")

    # --- define directory layout ---
    config = {
        "target_simulator": target,
        "manifest": {
            "$BASE_DIR": "{}".format(output_path),
            "$NETWORK_DIR": "$BASE_DIR/networks",
            "$COMPONENT_DIR": "$BASE_DIR/components"
        },
        "components": {
            "morphologies_dir": "$COMPONENT_DIR/morphologies",
            "synaptic_models_dir": "$COMPONENT_DIR/synapse_dynamics",
            "point_neuron_models_dir": "$COMPONENT_DIR/point_neuron_dynamics",
            "mechanisms_dir": "$COMPONENT_DIR/mechanisms",
            "biophysical_neuron_models_dir": "$COMPONENT_DIR/biophysical_neuron_dynamics",
            "templates": "$COMPONENT_DIR/hoc_templates",
        },
        "networks": {
            "nodes": [],
            "edges": []
        }
    }

    base_dir = config["manifest"]["$BASE_DIR"]
    network_dir = Template(config["manifest"]["$NETWORK_DIR"]).substitute(BASE_DIR=base_dir)
    component_dir = Template(config["manifest"]["$COMPONENT_DIR"]).substitute(BASE_DIR=base_dir)

    for directory in (base_dir, network_dir, component_dir):
        if exists(directory) and isdir(directory) and overwrite:
            shutil.rmtree(directory)
        os.makedirs(directory)

    # --- export neuronal morphologies ---
    # - not necessary for the current version of PyNN

    # --- export NMODL files for neuron and synapse models ---
    # - note that this assumes that plasticity mechanisms are
    #   implemented as part of the synapse model rather than as
    #   separate weight-adjuster mechanisms
    # - also note that future versions of this format may
    #   allow exporting mechanisms in LEMS format
    if target.lower() == "neuron":
        raise NotImplementedError

    # --- export biophysical neuron channel distribution ---
    # - not necessary for the current version of PyNN

    # --- export nodes ---
    # - we define a separate SONATA node population for each PyNN Population
    # - we may in future use node groups and/or node types to support PopulationViews
    # - Assemblies are not explicitly represented in the SONATA structure,
    #   rather their constituent Populations are exported individually.

    for i, population in enumerate(network.populations):

        config["networks"]["nodes"].append({
            "nodes_file": "$NETWORK_DIR/nodes_{}.h5".format(population.label),
            "node_types_file": "$NETWORK_DIR/node_types_{}.csv".format(population.label)
        })
        node_type_path = Template(config["networks"]["nodes"][i]
                                  ["node_types_file"]).substitute(NETWORK_DIR=network_dir)
        nodes_path = Template(config["networks"]["nodes"][i]
                              ["nodes_file"]).substitute(NETWORK_DIR=network_dir)

        n = population.size
        population_label = asciify(population.label)
        csv_rows = []
        csv_columns = set()
        node_type_info = {
            "node_type_id": i,
        }
        if "SpikeSource" in population.celltype.__class__.__name__:
            node_type_info["model_type"] = "virtual"
        else:
            if target.lower() == "neuron":
                node_type_info["model_type"] = "single_compartment"
            elif target.lower() in ("pynn", "nest"):
                node_type_info["model_type"] = "point_neuron"
                node_type_info["model_template"] = (
                    f"{target.lower()}:{population.celltype.__class__.__name__}"
                )
            else:
                raise NotImplementedError
        group_label = 0  # "default"
        # todo: add "population" column

        # write HDF5 file
        nodes_file = h5py.File(nodes_path, 'w')
        # ??? unclear what is the required format or the current version!
        nodes_file.attrs["version"] = (0, 1)
        nodes_file.attrs["magic"] = MAGIC  # needs to be uint32
        root = nodes_file.create_group("nodes")  # todo: add attribute with network name
        # we use a single node group for the full Population
        default = root.create_group(population_label)
        # todo: check and fix the dtypes in the following
        default.create_dataset("node_id", data=population.all_cells.astype('i4'), dtype='i4')
        default.create_dataset("node_type_id", data=i * np.ones((n,)), dtype='i2')
        default.create_dataset("node_group_id", data=np.array(
            [group_label] * n), dtype='i2')  # todo: calculate the max label size
        # required data type not specified. Optional?
        default.create_dataset("node_group_index", data=np.arange(n, dtype=int), dtype='i2')

        # parameters
        node_group = default.create_group(str(group_label))
        node_params_group = node_group.create_group("dynamics_params")
        for parameter_name in population.celltype.default_parameters:
            if parameter_name == "spike_times":
                warn("spike times should be added manually to simulation_config")
            else:
                # we could make a single get call to get all params at once, at the expense
                # of higher memory usage. To profile...
                values = population.get(parameter_name, gather=True, simplify=True)
                if isinstance(values, np.ndarray):
                    # array, put into the HDF5 file and put 'NONE' in the CSV file
                    node_params_group.create_dataset(parameter_name, data=values)
                    node_type_info[parameter_name] = "NONE"
                else:
                    # scalar, put into the CSV file
                    node_type_info[parameter_name] = values

        # positions in space
        x, y, z = population.positions
        node_group.create_dataset('x', data=x)
        node_group.create_dataset('y', data=y)
        node_group.create_dataset('z', data=z)

        csv_rows.append(node_type_info)
        csv_columns.update(node_type_info.keys())

        nodes_file.close()

        # now write csv file
        field_names = sorted(set.union(*(set(row) for row in csv_rows))
                             )  # todo: `node_type_id` must be first
        with open(node_type_path, 'w') as csv_file:
            csv_writer = csv.DictWriter(csv_file,
                                        fieldnames=field_names,
                                        delimiter=' ',
                                        quotechar='"')
            csv_writer.writeheader()
            for row in csv_rows:
                logger.info(row)
                for column_name in csv_columns:
                    if column_name not in row:
                        row[column_name] = "NONE"
                csv_writer.writerow(row)

    # --- export edges ---
    # - we define a separate group and edge-type for each PyNN Projection

    for i, projection in enumerate(network.projections):
        projection_label = asciify(projection.label)
        config["networks"]["edges"].append({
            "edges_file": "$NETWORK_DIR/edges_{}.h5".format(projection_label),
            "edge_types_file": "$NETWORK_DIR/edge_types_{}.csv".format(projection_label)
        })
        edge_type_path = Template(config["networks"]["edges"][i]
                                  ["edge_types_file"]).substitute(NETWORK_DIR=network_dir)
        edges_path = Template(config["networks"]["edges"][i]
                              ["edges_file"]).substitute(NETWORK_DIR=network_dir)

        n = projection.size()
        csv_rows = []
        edge_type_info = {
            "edge_type_id": i,
            "model_template": "{}:{}".format(target.lower(),
                                             projection.synapse_type.__class__.__name__),
            "receptor_type": projection.receptor_type
        }
        parameter_names = list(projection.synapse_type.default_parameters)
        values = np.array(
            projection.get(parameter_names, format="list", gather=True, with_address=True)
        )
        source_index = values[:, 0].astype(int)
        target_index = values[:, 1].astype(int)
        source_gids = projection.pre.all_cells[source_index].astype('i4')
        target_gids = projection.post.all_cells[target_index].astype('i4')
        group_label = 0  # "default"

        # Write HDF5 file
        edges_file = h5py.File(edges_path, 'w')
        root = edges_file.create_group("edges")  # todo: add attribute with network name

        default_edge_pop = root.create_group(projection_label)
        default_edge_pop.create_dataset("source_node_id", data=source_gids, dtype='i4')
        default_edge_pop.create_dataset("target_node_id", data=target_gids, dtype='i4')
        default_edge_pop["source_node_id"].attrs["node_population"] = asciify(
            projection.pre.label)  # todo: handle PopualtionViews
        default_edge_pop["target_node_id"].attrs["node_population"] = asciify(
            projection.post.label)
        default_edge_pop.create_dataset("edge_type_id", data=i * np.ones((n,)), dtype='i2')
        # S32')  # todo: calculate the max label size
        default_edge_pop.create_dataset(
            "edge_group_id", data=np.array([group_label] * n), dtype='i2')
        default_edge_pop.create_dataset(
            "edge_group_index", data=np.arange(n, dtype=int), dtype='i2')

        edge_group = default_edge_pop.create_group(str(group_label))
        edge_params = edge_group.create_group("dynamics_params")
        for j, parameter_name in zip(range(2, values.shape[1]), parameter_names):
            if isinstance(values[:, j], np.ndarray):
                # array, put into the HDF5 file
                edge_params.create_dataset(parameter_name, data=values[:, j])
            else:
                # scalar, put into the CSV file
                # for now, this will never be the case - need to detect homogeneous case
                edge_type_info[parameter_name] = values[:, j]

        csv_rows.append(edge_type_info)

        # todo: add receptor_type to csv_rows

        edges_file.close()

        # now write csv file
        field_names = sorted(set.union(*(set(row) for row in csv_rows)))
        with open(edge_type_path, 'w') as csv_file:
            csv_writer = csv.DictWriter(csv_file,
                                        fieldnames=field_names,
                                        delimiter=' ',
                                        quotechar='"')
            csv_writer.writeheader()
            for row in csv_rows:
                csv_writer.writerow(row)

    # --- write the config file ---

    with open(join(output_path, "circuit_config.json"), "w") as fp:
        json.dump(config, fp, indent=2)

    # --- export recording configuration ---
    # todo


def import_from_sonata(config_file, sim):
    """
    We map a SONATA population to a PyNN Assembly, since both allow heterogeneous cell types.
    We map a SONATA node group to a PyNN Population, since both have homogeneous parameter
    namespaces.
    SONATA node types are used to give different parameters to different subsets of nodes
    in a group.
    This can be handled in PyNN by indexing and, equivalently, by defining PopulationViews.
    We map a SONATA edge group to a PyNN Projection, i.e. a SONATA edge population may
    result in multiple PyNN Projections.
    """
    if not HAVE_H5PY:
        raise Exception("You need to install h5py to use SONATA")

    config = load_config(config_file)

    if config.get("target_simulator", None) not in ("PyNN", "NEST"):
        warn("`target_simulator` is not set to 'PyNN' or 'NEST'. Proceeding with caution...")

    sonata_node_populations = []
    for nodes_config in config["networks"]["nodes"]:

        # Load node types into node_types_map
        node_types_map = read_types_file(nodes_config["node_types_file"], 'node')

        # Open nodes file, check it is valid
        nodes_file = h5py.File(nodes_config["nodes_file"], 'r')
        version = nodes_file.attrs.get("version", None)
        magic = nodes_file.attrs.get("magic", None)
        if magic is not None and magic != MAGIC:
            # for now we assume that not all SONATA files will have the magic attribute set
            raise Exception("Invalid SONATA file")

        # Read data about node populations and groups
        sonata_node_populations.extend([
            NodePopulation.from_data(np_label, np_data, node_types_map, config)
            for np_label, np_data in nodes_file["nodes"].items()
        ])

    sonata_edge_populations = []

    if "edges" in config["networks"]:
        for edges_config in config["networks"]["edges"]:

            # Load edge types into edge_types_map
            edge_types_map = read_types_file(edges_config["edge_types_file"], 'edge')

            # Open edges file, check it is valid
            edges_file = h5py.File(edges_config["edges_file"], 'r')
            version = edges_file.attrs.get("version", None)  # noqa: F841
            magic = edges_file.attrs.get("magic", None)
            if magic is not None and magic != MAGIC:
                # for now we assume that not all SONATA files will have the magic attribute set
                raise Exception("Invalid SONATA file")

            # Read data about edge populations and groups

            sonata_edge_populations.extend([
                EdgePopulation.from_data(ep_label, ep_data, edge_types_map, config)
                for ep_label, ep_data in edges_file["edges"].items()
            ])

    # Now map the SONATA data structures to PyNN ones

    net = Network()
    for node_population in sonata_node_populations:
        assembly = node_population.to_assembly(sim)
        net.add(assembly)
    for edge_population in sonata_edge_populations:
        projections = edge_population.to_projections(net, sim)
        net.add(*projections)

    return net


def load_sonata_simulation_plan(config_file):
    """Create a simulation plan (what to record, etc.) from a simulation config file."""
    config = load_config(config_file)
    plan = SimulationPlan(**config)
    return plan


# ----- internal API ---------------------------------------------------------

# The following classes are not intended to be instantiated directly, but
# are used internally by the public functions.


class NodePopulation(object):
    """Representation of a SONATA node population"""

    @classmethod
    def from_data(cls, name, h5_data, node_types_map, config):
        """Create a NodePopulation instance, containing one or more NodeGroups, from data.

        Arguments
        ---------

        name : string
            Taken from the SONATA nodes HDF5 file.
        h5_data : HDF5 Group
            The "/nodes/<population_name>" group.
        node_types_map : dict
            Data loaded from node types CSV file. Top-level keys are node type ids.
        config : dict
            Circuit config loaded from JSON.
        """
        obj = cls()
        obj.name = name
        obj.node_groups = []
        obj.node_ids = h5_data['node_id']

        node_group_ids = h5_data['node_group_id'][()]
        for ng_label in np.unique(node_group_ids):
            mask = node_group_ids == ng_label
            logger.info("NODE GROUP {}, size {}".format(ng_label, mask.sum()))

            node_type_array = h5_data['node_type_id'][mask]
            node_group_index = h5_data['node_group_index'][mask].tolist()
            obj.node_groups.append(
                NodeGroup.from_data(ng_label,
                                    node_type_array,
                                    node_group_index,
                                    h5_data[str(ng_label)],
                                    node_types_map,
                                    config)
            )
            # todo: handle spatial structure - h5_data['x'], etc.
        return obj

    def __repr__(self):
        return "NodePopulation(name='{}', node_groups={})".format(self.name, self.node_groups)

    def to_assembly(self, sim):
        """Create a PyNN Assembly from this NodePopulation.

        The Assembly will contain one Population for each NodeGroup.
        """
        assembly = sim.Assembly(label=self.name)
        assembly.annotations["first_sonata_id"] = self.node_ids[()].min()
        for node_group in self.node_groups:
            pop = node_group.to_population(sim)
            assembly += pop
        return assembly


class NodeGroup(object):
    """Representation of a SONATA node population"""

    def __len__(self):
        return self.node_types_array.size

    @property
    def size(self):
        return len(self)

    @classmethod
    def from_data(cls, id, node_types_array, index, h5_data, node_types_map, config):
        """Create a NodeGroup instance from data.

        Arguments
        ---------

        id : integer
            Taken from the SONATA nodes HDF5 file.
        node_types_array : NumPy array
            Subset of the data from "/nodes/<population_name>/node_type_id"
            that applies to this group.
        index : list
            Subset of the data from "/nodes/<population_name>/node_group_index"
            that applies to this group.
        h5_data : HDF5 Group
            The "/nodes/<population_name>/<group_id>" group.
        node_types_map : dict
            Data loaded from node types CSV file. Top-level keys are node type ids.
        config : dict
            Circuit config loaded from JSON.
        """
        obj = cls()
        obj.id = id
        obj.node_types_array = node_types_array

        parameters = defaultdict(dict)
        node_type_ids = np.unique(node_types_array)

        # parameters defined directly in node_types csv file
        for node_type_id in node_type_ids:
            for name, value in node_types_map[node_type_id].items():
                parameters[name][node_type_id] = cast(value)

        # parameters defined in json files referenced from node_types.csv
        if "dynamics_params" in parameters:
            for node_type_id in node_type_ids:
                parameter_file_name = parameters["dynamics_params"][node_type_id]
                parameter_file_path = join(config["components"]["point_neuron_models_dir"],
                                           parameter_file_name)
                with open(parameter_file_path) as fp:
                    dynamics_params = json.load(fp)

                for name, value in dynamics_params.items():
                    parameters[name][node_type_id] = value

        # parameters defined in .h5 files
        if 'dynamics_params' in h5_data:
            dynamics_params_group = h5_data['dynamics_params']
            # not sure the next bit is using `index` correctly
            for key in dynamics_params_group.keys():
                parameters[key] = dynamics_params_group[key][index]

        obj.parameters = parameters
        obj.config = config
        logger.info(parameters)

        return obj

    def __repr__(self):
        return "NodeGroup(id='{}', parameters={})".format(self.id, self.parameters)

    def get_cell_type(self, sim):
        """Determine which PyNN cell type to use, and return its class."""
        cell_types = set()
        model_types = self.parameters["model_type"]
        for node_type_id, model_type in model_types.items():
            if model_type not in ("point_neuron", "point_process", "virtual"):
                raise NotImplementedError("Only point neurons currently supported.")

            if model_type == "virtual":
                if self.config.get("target_simulator") == "NEST":
                    cell_types.add("nest:spike_generator")
                else:
                    cell_types.add("pyNN:SpikeSourceArray")
            else:
                cell_types.add(self.parameters["model_template"][node_type_id])

        if len(cell_types) != 1:
            raise Exception("Heterogeneous group, not currently supported.")

        cell_type = cell_types.pop()
        prefix, cell_type_name = cell_type.split(":")
        if prefix.lower() not in ("pynn", "nrn", "nest"):
            raise NotImplementedError(
                f"Only PyNN, NEST and NEURON-native networks currently supported, not: {prefix}"
                f" (from {self.parameters['model_template'][node_type_id]})"
            )
        if prefix.lower() == "nest":
            cell_type_cls = sim.native_cell_type(cell_type_name)
            if cell_type_name == "spike_generator":
                cell_type_cls.uses_parrot = False
        else:
            cell_type_cls = getattr(sim, cell_type_name)
        logger.info("  cell_type: {}".format(cell_type_cls))
        return cell_type_cls

    def to_population(self, sim):
        """Create a PyNN Population from this NodeGroup."""
        cell_type_cls = self.get_cell_type(sim)
        parameters = {}
        annotations = {}

        for name, value in self.parameters.items():
            if name in cell_type_cls.default_parameters:
                parameters[name] = condense(value, self.node_types_array)
            else:
                annotations[name] = condense(value, self.node_types_array)
        # todo: handle spatial structure - nodes_file["nodes"][np_label][ng_label]['x'], etc.

        # temporary hack to work around problem with 300 Intfire cell example
        if cell_type_cls.__name__ == 'IntFire1':
            parameters['tau'] *= 1000.0
            parameters['refrac'] *= 1000.0
        # end hack

        cell_type = cell_type_cls(**parameters)
        pop = sim.Population(self.size,
                             cell_type,
                             label=str(self.id))
        pop.annotate(**annotations)
        logger.info("--------> {}".format(pop))
        # todo: create PopulationViews if multiple node_types
        return pop


class EdgePopulation(object):
    """Representation of a SONATA edge population"""

    @classmethod
    def from_data(cls, name, h5_data, edge_types_map, config):
        """Create an EdgePopulation instance, containing one or more EdgeGroups, from data.

        Arguments
        ---------

        name : string
            Taken from the SONATA edges HDF5 file.
        h5_data : HDF5 Group
            The "/edges/<population_name>" group.
        edge_types_map : dict
            Data loaded from edge types CSV file. Top-level keys are edge type ids.
        config : dict
            Circuit config loaded from JSON.
        """

        obj = cls()
        obj.name = name
        obj.source_node_ids = h5_data["source_node_id"][()]
        obj.source_node_population = to_string(h5_data["source_node_id"].attrs["node_population"])
        obj.target_node_ids = h5_data["target_node_id"][()]
        obj.target_node_population = to_string(h5_data["target_node_id"].attrs["node_population"])

        obj.edge_groups = []
        for eg_label in np.unique(h5_data['edge_group_id'][()]):
            mask = h5_data['edge_group_id'][()] == eg_label
            logger.info("EDGE GROUP {}, size {}".format(eg_label, mask.sum()))

            edge_type_array = h5_data['edge_type_id'][mask]
            edge_group_index = h5_data['edge_group_index'][mask].tolist()
            source_ids = obj.source_node_ids[mask]
            target_ids = obj.target_node_ids[mask]
            # note: it may be more efficient in an MPI context
            #       to defer the extraction of source_ids, etc.

            obj.edge_groups.append(
                EdgeGroup.from_data(eg_label,
                                    edge_type_array,
                                    edge_group_index,
                                    source_ids,
                                    target_ids,
                                    h5_data[str(eg_label)],
                                    edge_types_map,
                                    config)
            )

        return obj

    def __repr__(self):
        return "EdgePopulation(name='{}', edge_groups={})".format(self.name, self.edge_groups)

    def to_projections(self, net, sim):
        """Create a list of PyNN Projections from this EdgePopulation."""
        pre = net.get_component(self.source_node_population)
        post = net.get_component(self.target_node_population)
        projections = []
        for edge_group in self.edge_groups:
            projection = edge_group.to_projection(pre, post, self.name, sim)
            projections.append(projection)
        return projections


class EdgeGroup(object):
    """Representation of a SONATA edge group."""

    @classmethod
    def from_data(cls, id, edge_types_array, index, source_ids, target_ids,
                  h5_data, edge_types_map, config):
        """Create an EdgeGroup instance from data.

        Arguments
        ---------

        id : integer
            Taken from the SONATA edges HDF5 file.
        node_types_array : NumPy array
            Subset of the data from "/edges/<population_name>/edge_type_id"
            that applies to this group.
        index : list
            Subset of the data from "/edges/<population_name>/edge_group_index"
            that applies to this group.
        h5_data : HDF5 Group
            The "/edges/<population_name>/<group_id>" group.
        edge_types_map : dict
            Data loaded from edge types CSV file. Top-level keys are edge type ids.
        config : dict
            Circuit config loaded from JSON.
        """
        obj = cls()
        obj.id = id
        obj.edge_types_array = edge_types_array
        obj.source_ids = source_ids
        obj.target_ids = target_ids

        parameters = defaultdict(dict)
        edge_type_ids = np.unique(edge_types_array)

        # parameters defined directly in edge_types csv file
        for edge_type_id in edge_type_ids:
            for name, value in edge_types_map[edge_type_id].items():
                parameters[name][edge_type_id] = cast(value)

        # parameters defined in json files referenced from edge_types.csv
        if "dynamics_params" in parameters:
            for edge_type_id in edge_type_ids:
                parameter_file_name = parameters["dynamics_params"][edge_type_id]
                parameter_file_path = join(config["components"]["synaptic_models_dir"],
                                           parameter_file_name)
                with open(parameter_file_path) as fp:
                    dynamics_params = json.load(fp)

                for name, value in dynamics_params.items():
                    parameters[name][edge_type_id] = value

        # parameters defined in .h5 files
        if 'dynamics_params' in h5_data:
            dynamics_params_group = h5_data['dynamics_params']
            # not sure the next bit is using `index` correctly
            for key in dynamics_params_group.keys():
                parameters[key] = dynamics_params_group[key][index]
        if 'nsyns' in h5_data:
            parameters['nsyns'] = h5_data['nsyns'][index]
        if 'syn_weight' in h5_data:
            parameters['syn_weight'] = h5_data['syn_weight'][index]

        obj.parameters = parameters
        obj.config = config
        logger.info(parameters)
        return obj

    def __repr__(self):
        return "EdgeGroup(id='{}', parameters={})".format(self.id, self.parameters)

    def get_synapse_and_receptor_type(self, sim):
        """Determine which PyNN synapse and receptor type to use.

        Returns the synapse type class, and the receptor type label."""
        synapse_types = set()

        model_templates = self.parameters.get("model_template", None)
        if model_templates:
            for edge_type_id, model_template in model_templates.items():
                synapse_types.add(model_template)

            if len(synapse_types) != 1:
                raise Exception("Heterogeneous group, not currently supported.")

            # synapse_type = synapse_types.pop()
            prefix, synapse_type_name = model_template.split(":")
            if prefix.lower() not in ("pynn", "nrn", "nest"):
                raise NotImplementedError(
                    "Only PyNN, NEST and NEURON-native networks currently supported.")
        else:
            prefix = "pyNN"
            synapse_type_name = "StaticSynapse"

        if prefix == "nest":
            synapse_type_cls = sim.native_synapse_type(synapse_type_name)
        else:
            synapse_type_cls = getattr(sim, synapse_type_name)

        receptor_types = self.parameters.get("receptor_type", None)
        if receptor_types:
            receptor_types = set(receptor_types.values())

            if len(receptor_types) != 1:
                raise Exception("Heterogeneous receptor types, not currently supported.")
                # but should be, since SONATA egde groups can contain mixed excitatory
                # and inhibitory connections.
                # Would need to split into separate PyNN Projections, in this case.

            receptor_type = receptor_types.pop()
        else:
            # temporary hack to make 300-cell example work, due to PyNN bug #597
            receptor_type = "default"
            # value should really be None.

        logger.info("  synapse_type: {}".format(synapse_type_cls))
        logger.info("  receptor_type: {}".format(receptor_type))
        return synapse_type_cls, receptor_type

    def to_projection(self, pre, post, edge_population_name, sim):
        """Create a PyNN Projection from this EdgeGroup.

        Arguments
        ---------

        pre - PyNN Assembly
            The assembly created from the pre-synaptic node population.
        post - PyNN Assembly
            The assembly created from the post-synaptic node population.
        edge_population_name - string
            Name of the edge population containing this edge group.
        """
        synapse_type_cls, receptor_type = self.get_synapse_and_receptor_type(sim)
        parameters = {}
        annotations = {}

        for name, value in self.parameters.items():
            if name in synapse_type_cls.default_parameters:
                parameters[name] = condense(value, self.edge_types_array)
            elif name == "syn_weight":
                parameters["weight"] = condense(value, self.edge_types_array)
            else:
                annotations[name] = value

        if "sign" in annotations:
            # special case from the 300 IF example, not mentioned in the SONATA spec
            # nor in the .mod file for IntFire1
            sign = condense(annotations["sign"], self.edge_types_array)
            parameters["weight"] *= sign

        if "nsyns" in annotations:
            # special case (?) from the 300 IF example, not mentioned in the SONATA spec
            # nor in the .mod file for IntFire1
            nsyns = condense(annotations["nsyns"], self.edge_types_array)
            parameters["weight"] *= nsyns

        conn_list_args = [
            sonata_id_to_index(pre, self.source_ids),
            sonata_id_to_index(post, self.target_ids)
        ]
        column_names = []
        synapse_type_parameters = {}
        for name, value in parameters.items():
            if isinstance(value, (np.ndarray)):
                column_names.append(name)
                conn_list_args.append(value)
            else:
                synapse_type_parameters[name] = value

        conn_list = np.array(conn_list_args).transpose()
        connector = sim.FromListConnector(conn_list,
                                          column_names=column_names)
        syn = synapse_type_cls(**synapse_type_parameters)
        prj = sim.Projection(pre, post, connector, syn,
                             receptor_type=receptor_type,
                             label="{}-{}".format(edge_population_name, self.id))
        prj.annotate(**annotations)
        logger.info("--------> {}".format(prj))
        return prj


class SimulationPlan(object):
    """ """

    def __init__(self, run, inputs=None, output=None, reports=None,
                 target_simulator=None, node_sets_file=None, conditions=None,
                 manifest=None, **additional_sections):
        self.run_config = run
        self.inputs = inputs
        self.output = output
        self.reports = reports
        self.target_simulator = target_simulator
        self.node_sets_file = node_sets_file
        self.conditions = conditions
        self.manifest = manifest
        self.additional_sections = additional_sections
        if self.inputs is None:
            self.inputs = {}
        if self.reports is None:
            self.reports = {}
        if self.node_sets_file is not None:
            with open(self.node_sets_file) as fp:
                self.node_sets = json.load(fp)
                # make all node set names lower case, needed by 300 IF neuron example
                self.node_sets = {k.lower(): v for k, v in self.node_sets.items()}
                # todo: handle compound node sets

    def setup(self, sim):
        self.sim = sim
        sim.setup(timestep=self.run_config["dt"])

    def _get_target(self, config, net):
        if "node_set" in config:  # input config
            targets = self.node_set_map[config["node_set"]]
        elif "cells" in config:   # recording config
            # inconsistency in SONATA spec? Why not call this "node_set" also?
            targets = self.node_set_map[config["cells"]]
        return targets

    def _set_input_spikes(self, input_config, net):
        # determine which assembly the spikes are for
        targets = self._get_target(input_config, net)
        if len(targets) != 1:
            raise NotImplementedError()
        base_assembly, mask = targets[0]
        assembly = base_assembly[mask]
        assert isinstance(assembly, self.sim.Assembly)

        # load spike data from file
        if input_config["module"] != "h5":
            raise NotImplementedError()
        io = SonataIO(base_dir="",
                      spikes_file=input_config["input_file"])
        data = io.read()
        assert len(data) == 1
        if "trial" in input_config:
            raise NotImplementedError()
            # assuming we can map trials to segments
        assert len(data[0].segments) == 1
        spiketrains = data[0].segments[0].spiketrains
        if len(spiketrains) != assembly.size:
            raise NotImplementedError()
        # todo: map cell ids in spikes file to ids/index in the population
        assembly.set(spike_times=[Sequence(st.times.rescale('ms').magnitude)
                                  for st in spiketrains])

    def _set_input_currents(self, input_config, net):
        # determine which assembly the currents are for
        if "input_file" in input_config:
            raise NotImplementedError("Current clamp from source file not yet supported.")
        targets = self._get_target(input_config, net)
        if len(targets) != 1:
            raise NotImplementedError()
        base_assembly, mask = targets[0]
        assembly = base_assembly[mask]
        assert isinstance(assembly, self.sim.Assembly)
        amplitude = input_config["amp"]  # nA
        if self.target_simulator == "NEST":
            amplitude = input_config["amp"]/1000.0  # pA

        current_source = self.sim.DCSource(amplitude=amplitude,
                                           start=input_config["delay"],
                                           stop=input_config["delay"] + input_config["duration"])
        assembly.inject(current_source)

    def _calculate_node_set_map(self, net):
        # for each "node set" in the config, determine which populations
        # and node_ids it corresponds to
        self.node_set_map = {}

        # first handle implicit node sets - i.e. each node population is an implicit node set
        for assembly in net.assemblies:
            self.node_set_map[assembly.label] = [(assembly, slice(None))]

        # now handle explictly-declared node sets
        #   todo: handle compound node sets
        for node_set_name, node_set_definition in self.node_sets.items():
            if isinstance(node_set_definition, dict):  # basic node set
                filters = node_set_definition
                if "population" in filters:
                    assemblies = [net.get_component(filters["population"])]
                else:
                    assemblies = list(net.assemblies)

                self.node_set_map[node_set_name] = []
                for assembly in assemblies:
                    mask = True
                    for attr_name, attr_value in filters.items():
                        print(attr_name, attr_value, "____")
                        if attr_name == "population":
                            continue
                        elif attr_name == "node_id":
                            # convert integer mask to boolean mask
                            node_mask = np.zeros(assembly.size, dtype=bool)
                            node_mask[attr_value] = True
                            mask = np.logical_and(mask, node_mask)
                        else:
                            values = assembly.get_annotations(attr_name)[attr_name]
                            mask = np.logical_and(mask, values == attr_value)
                    if isinstance(mask, (bool, np.bool_)) and mask is True:
                        mask = slice(None)
                    self.node_set_map[node_set_name].append((assembly, mask))
            elif isinstance(node_set_definition, list):  # compound node set
                raise NotImplementedError("Compound node sets not yet supported")
            else:
                raise TypeError("Expecting node set definition to be a list or dict")

    def execute(self, net):
        self._calculate_node_set_map(net)

        # create/configure inputs
        for input_name, input_config in self.inputs.items():
            if input_config["input_type"] == "spikes":
                self._set_input_spikes(input_config, net)
            elif input_config["input_type"] == "current_clamp":
                self._set_input_currents(input_config, net)
            else:
                raise NotImplementedError("Only 'spikes' and 'current_clamp' supported")

        # configure recording
        # SONATA requires that we record spikes from all non-virtual nodes
        net.record('spikes', include_spike_source=False)
        for report_name, report_config in self.reports.items():
            targets = self._get_target(report_config, net)
            for (base_assembly, mask) in targets:
                assembly = base_assembly[mask]
                assembly.record(report_config["variable_name"])

        # run simulation
        self.sim.run(self.run_config["tstop"])

        # write output
        if "overwrite_output_dir" in self.run_config:
            directory = self.output["output_dir"]
            if exists(directory) and isdir(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)
        io = SonataIO(self.output["output_dir"],
                      spikes_file=self.output.get("spikes_file", "spikes.h5"),
                      spikes_sort_order=self.output["spikes_sort_order"],
                      report_config=self.reports,
                      node_sets=self.node_set_map)
        # todo: handle reports
        net.write_data(io)

    @classmethod
    def from_config(cls, config):
        obj = cls(**config)
        return obj
