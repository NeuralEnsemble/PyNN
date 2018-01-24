"""
Functions for exporting and importing networks to/from files
"""

import os
from os.path import join, isdir
from string import Template
import csv
import json
import h5py
import numpy as np


def asciify(label):
    """To be safe, we will use only ascii for strings inside HDF5"""
    return label.replace(u"→", "-").encode('ascii')


def export_to_abc(network, output_path):
    """Export a network in AIBS-BBP Circuit format"""

    # --- define directory layout ---
    config = {
        "target_simulator": "PyNN",
        "manifest": {
            "$BASE_DIR": "${configdir}",
            "$NETWORK_DIR": "$BASE_DIR/networks",
            "$COMPONENT_DIR": "$BASE_DIR/components"
        },
        "components": {
            "morphologies": "$COMPONENT_DIR/morphologies",
            "synaptic_models": "$COMPONENT_DIR/synapse_dynamics",
            "point_neuron_models": "$COMPONENT_DIR/point_neuron_dynamics",
            "mechanisms":"$COMPONENT_DIR/mechanisms",
            "biophysical_neuron_models": "$COMPONENT_DIR/biophysical_neuron_dynamics",
            "templates": "$COMPONENT_DIR/hoc_templates",
        },
        "networks": {
            "node_files": [
                {
                    "nodes": "$NETWORK_DIR/nodes.h5",
                    "node_types": "$NETWORK_DIR/node_types.csv"
                }
            ],
            "edge_files":[
                {
                    "edges": "$NETWORK_DIR/edges.h5",
                    "edge_types": "$NETWORK_DIR/edge_types.csv"
                },
            ]
        }
    }

    base_dir = Template(config["manifest"]["$BASE_DIR"]).substitute(configdir=output_path)
    network_dir = Template(config["manifest"]["$NETWORK_DIR"]).substitute(BASE_DIR=base_dir)
    component_dir = Template(config["manifest"]["$COMPONENT_DIR"]).substitute(BASE_DIR=base_dir)

    for directory in (base_dir, network_dir, component_dir):
        os.makedirs(directory)

    # --- export neuronal morphologies ---
    # - not necessary for the current version of PyNN

    # --- export NMODL files for neuron and synapse models ---
    # - note that this assumes that plasticity mechanisms are
    #   implemented as part of the synapse model rather than as
    #   separate weight-adjuster mechanisms
    # - also note that future versions of this format may
    #   allow exporting mechanisms in LEMS format

    # --- export biophysical neuron channel distribution ---
    # - not necessary for the current version of PyNN

    # --- export nodes ---
    # - we define a separate group and node-type for each PyNN Population
    # - maybe in future we could exploit node-type to support PopulationViews
    #   and Assemblies, but we keep it simple for now

    node_type_path = Template(config["networks"]["node_files"][0]["node_types"]).substitute(NETWORK_DIR=network_dir)
    nodes_path = Template(config["networks"]["node_files"][0]["nodes"]).substitute(NETWORK_DIR=network_dir)

    nodes_file = h5py.File(nodes_path, 'w')

    n = network.count_neurons()
    root = nodes_file.create_group("nodes")  # todo: add attribute with network name
    root.create_dataset("node_gid", shape=(n,), dtype='i4')
    root.create_dataset("node_type_id", shape=(n,), dtype='i2')
    root.create_dataset("node_group", shape=(n,), dtype='S32')  # todo: calculate the max label size
    root.create_dataset("node_group_index", shape=(n,), dtype='i2')

    offset = 0
    csv_rows = []
    for i, population in enumerate(network.populations):  
        # note: this should include populations implicitly defined inside an Assembly
        m = population.size
        node_type_info = {
            "node_type_id": i,
            "model_type": "point_{}".format(population.celltype.__class__.__name__)
        }
        index = slice(offset, offset + m)
        group_label = asciify(population.label)
        root["node_gid"][index] = population.all_cells.astype('i4')
        root["node_type_id"][index] = i * np.ones((m,))
        root["node_group"][index] = np.array([group_label] * m)
        root["node_group_index"][index] = np.arange(m, dtype=int)

        node_group = root.create_group(group_label)

        # parameters
        for parameter_name in population.celltype.default_parameters:
            # we could make a single get call to get all params at once, at the expense
            # of higher memory usage. To profile
            values = population.get(parameter_name, gather=True, simplify=True)
            if isinstance(values, np.ndarray):
                # array, put into the HDF5 file
                node_group.create_dataset(parameter_name, data=values)
            else:
                # scalar, put into the CSV file
                node_type_info[parameter_name] = values

        # positions in space
        x, y, z = population.positions
        node_group.create_dataset('x', data=x)
        node_group.create_dataset('y', data=y)
        node_group.create_dataset('z', data=z)

        csv_rows.append(node_type_info)
        offset += m

    nodes_file.close()

    # now write csv file
    field_names = sorted(set.union(*(set(row) for row in csv_rows)))
    with open(node_type_path, 'w', newline='') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=field_names)
        csv_writer.writeheader()
        for row in csv_rows:
            csv_writer.writerow(row)

    # --- export edges ---
    # - we define a separate group and edge-type for each PyNN Projection

    edge_type_path = Template(config["networks"]["edge_files"][0]["edge_types"]).substitute(NETWORK_DIR=network_dir)
    edges_path = Template(config["networks"]["edge_files"][0]["edges"]).substitute(NETWORK_DIR=network_dir)

    edges_file = h5py.File(edges_path, 'w')

    n = network.count_connections()
    root = edges_file.create_group("edges")  # todo: add attribute with network name
    root.create_dataset("source_gid", shape=(n,), dtype='i4')  # todo: add attribute for network name
    root.create_dataset("target_gid", shape=(n,), dtype='i4')
    root.create_dataset("edge_type_id", shape=(n,), dtype='i2')
    root.create_dataset("edge_group", shape=(n,), dtype='S32')  # todo: calculate the max label size
    root.create_dataset("edge_group_index", shape=(n,), dtype='i2')

    offset = 0
    csv_rows = []
    for i, projection in enumerate(network.projections):
        m = projection.size()
        edge_type_info = {
            "edge_type_id": i,
            "template": "point_{}".format(projection.synapse_type.__class__.__name__)
        }
        index = slice(offset, offset + m)
        group_label = asciify(projection.label)

        edge_group = root.create_group(group_label)

        # - not clear if we should put parameters directly within `edge_group`, or
        #   create a sub-group "dynamics_params"
        parameter_names = list(projection.synapse_type.default_parameters)
        values = np.array(
            projection.get(parameter_names, format="list", gather=True, with_address=True)
        )
        source_index = values[:, 0].astype(int)
        target_index = values[:, 1].astype(int)
        source_gids = projection.pre.all_cells[source_index].astype('i4')
        target_gids = projection.post.all_cells[target_index].astype('i4')

        root["source_gid"][index] = source_gids
        root["target_gid"][index] = target_gids
        root["edge_type_id"][index] = i * np.ones((m,))
        root["edge_group"][index] = np.array([group_label] * m)
        root["edge_group_index"][index] = np.arange(m, dtype=int)

        for j, parameter_name in zip(range(2, values.shape[1]), parameter_names):
            if isinstance(values[:, j], np.ndarray):
                # array, put into the HDF5 file
                edge_group.create_dataset(parameter_name, data=values[:, j])
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
        csv_writer = csv.DictWriter(csv_file, fieldnames=field_names)
        csv_writer.writeheader()
        for row in csv_rows:
            csv_writer.writerow(row)

    # --- write the config file ---
    
    with open(join(output_path, "config.json"), "w") as fp:
        json.dump(config, fp, indent=2)

    # --- export recording configuration ---
    # not part of the standard



