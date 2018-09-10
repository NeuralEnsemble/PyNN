"""
Functions for exporting and importing networks to/from files
"""

import os
from os.path import join, isdir, exists
import shutil
from string import Template
import csv
from warnings import warn
import json
import h5py
import numpy as np
from .network import Network


MAGIC = 0x0a7a


def asciify(label):
    """To be safe, we will use only ascii for strings inside HDF5"""
    return label.replace(u"→", "-").encode('ascii')


def export_to_sonata(network, output_path, target="PyNN", overwrite=False):
    """Export a network in SONATA format

    If `target` is "neuron" or "NEURON", then PyNN cell types are exported as
    single compartment neurons with NEURON .mod files, and only StaticSynapses are supported.

    If `target` is "PyNN", then the names of PyNN standard cells and synapse models
    are used in the exported files. This allows simulations with any PyNN-supported
    simulator.

    If `target` is "nest" or "NEST", then the names of NEST cell and synapse models
    are used in the exported files.
    """

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
            "nodes": [  # todo: could have multiple entries here
                {
                    "nodes_file": "$NETWORK_DIR/nodes.h5",
                    "node_types_file": "$NETWORK_DIR/node_types.csv"
                }
            ],
            "edges":[  # todo: could have multiple entries here
                {
                    "edges_file": "$NETWORK_DIR/edges.h5",
                    "edge_types_file": "$NETWORK_DIR/edge_types.csv"
                },
            ]
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
    # - we define a separate group and node-type for each PyNN Population
    # - maybe in future we could exploit node-type to support PopulationViews
    #   and Assemblies, but we keep it simple for now

    node_type_path = Template(config["networks"]["nodes"][0]["node_types_file"]).substitute(NETWORK_DIR=network_dir)
    nodes_path = Template(config["networks"]["nodes"][0]["nodes_file"]).substitute(NETWORK_DIR=network_dir)

    nodes_file = h5py.File(nodes_path, 'w')
    nodes_file.attrs["version"] = (0, 1)  # ??? unclear what is the required format or the current version!
    nodes_file.attrs["magic"] = MAGIC  # needs to be uint32

    n = network.count_neurons()
    root = nodes_file.create_group("nodes")  # todo: add attribute with network name
    default = root.create_group("default")  # for now, put the entire Network into a single SONATA "population"
                                            # if future, we could have a SONATA population for each PyNN Assembly
    default.create_dataset("node_id", shape=(n,), dtype='i4')
    default.create_dataset("node_type_id", shape=(n,), dtype='i2')
    default.create_dataset("node_group_id", shape=(n,), dtype='S32')  # todo: calculate the max label size
                                                                      # required data type not specified. Optional?
    default.create_dataset("node_group_index", shape=(n,), dtype='i2')

    offset = 0
    csv_rows = []
    csv_columns = set()
    for i, population in enumerate(network.populations):
        # note: this should include populations implicitly defined inside an Assembly
        m = population.size
        node_type_info = {
            "node_type_id": i,
        }
        if target.lower() == "neuron":
            node_type_info["model_type"] = "single_compartment"
        elif target.lower() in ("pynn", "nest"):
            node_type_info["model_type"] = "point_neuron"
            node_type_info["model_template"] = "{}:{}".format(target.lower(),
                                                              population.celltype.__class__.__name__)
        else:
            raise NotImplementedError
        # todo: add "population" column

        index = slice(offset, offset + m)
        group_label = asciify(population.label)
        default["node_id"][index] = population.all_cells.astype('i4')
        default["node_type_id"][index] = i * np.ones((m,))
        default["node_group_id"][index] = np.array([group_label] * m)
        default["node_group_index"][index] = np.arange(m, dtype=int)

        node_group = default.create_group(group_label)

        # parameters
        node_params_group = node_group.create_group("dynamics_params")
        for parameter_name in population.celltype.default_parameters:
            # we could make a single get call to get all params at once, at the expense
            # of higher memory usage. To profile
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
        offset += m

    nodes_file.close()

    # now write csv file
    # todo: consider having one CSV per PyNN cell type
    #       for now, we put NONE for parameters that don't apply
    field_names = sorted(set.union(*(set(row) for row in csv_rows)))  # todo: `node_type_id` must be first
    with open(node_type_path, 'w', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file,
                                    fieldnames=field_names,
                                    delimiter=' ',
                                    quotechar='"')
        csv_writer.writeheader()
        for row in csv_rows:
            print(row)
            for column_name in csv_columns:
                if column_name not in row:
                    row[column_name] = "NONE"
            csv_writer.writerow(row)

    # --- export edges ---
    # - we define a separate group and edge-type for each PyNN Projection

    edge_type_path = Template(config["networks"]["edges"][0]["edge_types_file"]).substitute(NETWORK_DIR=network_dir)
    edges_path = Template(config["networks"]["edges"][0]["edges_file"]).substitute(NETWORK_DIR=network_dir)

    edges_file = h5py.File(edges_path, 'w')

    n = network.count_connections()
    root = edges_file.create_group("edges")  # todo: add attribute with network name
    default_edge_pop = root.create_group("default")  # for now, have a single "edge population"
    default_edge_pop.create_dataset("source_node_id", shape=(n,), dtype='i4')
    default_edge_pop.create_dataset("target_node_id", shape=(n,), dtype='i4')
    default_edge_pop["source_node_id"].attrs["node_population"] = "default"
    default_edge_pop["target_node_id"].attrs["node_population"] = "default"
    default_edge_pop.create_dataset("edge_type_id", shape=(n,), dtype='i2')
    default_edge_pop.create_dataset("edge_group_id", shape=(n,), dtype='S32')  # todo: calculate the max label size
    default_edge_pop.create_dataset("edge_group_index", shape=(n,), dtype='i2')

    offset = 0
    csv_rows = []
    for i, projection in enumerate(network.projections):
        m = projection.size()
        edge_type_info = {
            "edge_type_id": i,
            "model_template": "{}:{}".format(target.lower(),
                                              projection.synapse_type.__class__.__name__)
        }
        index = slice(offset, offset + m)
        group_label = asciify(projection.label)

        edge_group = default_edge_pop.create_group(group_label)

        parameter_names = list(projection.synapse_type.default_parameters)
        values = np.array(
            projection.get(parameter_names, format="list", gather=True, with_address=True)
        )
        source_index = values[:, 0].astype(int)
        target_index = values[:, 1].astype(int)
        source_gids = projection.pre.all_cells[source_index].astype('i4')
        target_gids = projection.post.all_cells[target_index].astype('i4')

        default_edge_pop["source_node_id"][index] = source_index.astype('i4')  # source_gids
        default_edge_pop["target_node_id"][index] = target_index.astype('i4')  # target_gids
        default_edge_pop["edge_type_id"][index] = i * np.ones((m,))
        default_edge_pop["edge_group_id"][index] = np.array([group_label] * m)
        default_edge_pop["edge_group_index"][index] = np.arange(m, dtype=int)

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
        offset += m

    edges_file.close()

    # now write csv file
    field_names = sorted(set.union(*(set(row) for row in csv_rows)))
    with open(edge_type_path, 'w', newline='') as csv_file:
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
    We map a SONATA node group to a PyNN Population, since both have heterogeneous parameter
    namespaces. This may not be entirely satisfactory, since a node group can have multiple
    node types, but needs some experience with different models to figure out.
    """
    with open(config_file) as fp:
        config = json.load(fp)
    if config["target_simulator"] != "PyNN":
        warn("`target_simulator` is not set to 'PyNN'. Proceeding with caution...")
        # could also easily handle target_simulator="NEST" using native models
        # NEURON also possible using native models, but a bit more involved

    base_dir = config["manifest"]["$BASE_DIR"]
    network_dir = Template(config["manifest"]["$NETWORK_DIR"]).substitute(BASE_DIR=base_dir)
    component_dir = Template(config["manifest"]["$COMPONENT_DIR"]).substitute(BASE_DIR=base_dir)

    net = Network()

    for nodes_config in config["networks"]["nodes"]:
        nodes_path = Template(nodes_config["nodes_file"]).substitute(NETWORK_DIR=network_dir)
        node_types_file = Template(nodes_config["node_types_file"]).substitute(NETWORK_DIR=network_dir)

        with open(node_types_file) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=' ', quotechar='"')
            node_types_table = list(csv_reader)
        node_types_map = {}
        for row in node_types_table:
            node_types_map[row["node_type_id"]] = row
        print(node_types_map)

        nodes_file = h5py.File(nodes_path, 'r')
        version = nodes_file.attrs.get("version", None)
        magic = nodes_file.attrs.get("magic", None)
        if magic is not None and magic != MAGIC:
            # for now we assume that not all SONATA files will have the magic attribute set
            raise Exception("Invalid SONATA file")

        node_populations = list(nodes_file["nodes"].keys())

        for np_label in node_populations:
            assembly = sim.Assembly(label=np_label)
            net.assemblies.add(assembly)

            node_groups = np.unique(nodes_file["nodes"][np_label]['node_group_id'].value)

            for ng_label in node_groups:
                print("NODE GROUP {}".format(ng_label))
                mask = nodes_file["nodes"][np_label]['node_group_id'].value == ng_label
                node_group_size = mask.sum()
                print("  size: {}".format(node_group_size))
                node_type_array = nodes_file["nodes"][np_label]['node_type_id'][mask]
                node_types = np.unique(node_type_array)
                print("  node_types: {}".format(node_types))

                cell_types = set()
                for node_type_id in node_types:
                    node_type_id = str(node_type_id)
                    if node_types_map[node_type_id]["model_type"] != "point_neuron":
                        raise NotImplementedError("Only point neurons currently supported.")
                    prefix, cell_type = node_types_map[node_type_id]["model_template"].split(":")
                    if prefix.lower() != "pynn":
                        raise NotImplementedError("Only PyNN networks currently supported.")
                    cell_types.add(cell_type)

                if len(cell_types) != 1:
                    raise Exception("Heterogeneous group, not currently supported.")
                cell_type_name = cell_types.pop()
                cell_type_cls = getattr(sim, cell_type_name)
                print("  cell_type: {}".format(cell_type_cls))

                parameters = {}
                if len(node_types) == 1:  # special case for efficiency
                    for parameter_name in cell_type_cls.default_parameters:
                        value = node_types_map[str(node_types[0])].get(parameter_name, None)
                        if value not in (None, "NONE"):
                            parameters[parameter_name] = float(value)
                else:
                    raise NotImplementedError("todo...")

                dynamics_params_group = nodes_file["nodes"][np_label][ng_label]['dynamics_params']
                group_mask = nodes_file["nodes"][np_label]['node_group_index'].value[mask]
                for key in dynamics_params_group.keys():
                    assert key in cell_type_cls.default_parameters
                    parameters[key] = dynamics_params_group[key].value[group_mask]

                print(parameters)

                # todo: handle spatial structure - nodes_file["nodes"][np_label][ng_label]['x'], etc.

                pop = sim.Population(node_group_size,
                                     cell_type_cls(**parameters),
                                     label=ng_label.decode('utf-8'))
                assembly += pop

    for edges_config in config["networks"]["edges"]:
        edges_path = Template(edges_config["edges_file"]).substitute(NETWORK_DIR=network_dir)
        edge_types_file = Template(edges_config["edge_types_file"]).substitute(NETWORK_DIR=network_dir)

        with open(edge_types_file) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=' ', quotechar='"')
            edge_types_table = list(csv_reader)
        edge_types_map = {}
        for row in edge_types_table:
            edge_types_map[row["edge_type_id"]] = row
        print(edge_types_map)

        edges_file = h5py.File(edges_path, 'r')
        version = edges_file.attrs.get("version", None)
        magic = edges_file.attrs.get("magic", None)
        if magic is not None and magic != MAGIC:
            # for now we assume that not all SONATA files will have the magic attribute set
            raise Exception("Invalid SONATA file")

        edge_populations = list(edges_file["edges"].keys())

        for ep_label in edge_populations:
            source_node_ids = edges_file["edges"][ep_label]["source_node_id"].value
            source_node_population = edges_file["edges"][ep_label]["source_node_id"].attrs["node_population"]
            target_node_ids = edges_file["edges"][ep_label]["target_node_id"].value
            target_node_population = edges_file["edges"][ep_label]["target_node_id"].attrs["node_population"]
            edge_groups = np.unique(edges_file["edges"][ep_label]['edge_group_id'].value)
            for eg_label in edge_groups:
                print("EDGE GROUP {}".format(eg_label))

            pre = net.get_component(source_node_population)[source_node_ids]
            post = net.get_component(target_node_population)[target_node_ids]
            print(pre)
            print(post)
            #prj = sim.Projection()
            #net.projections.add(prj)

    return net



