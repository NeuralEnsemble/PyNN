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


    A "node group" in SONATA corresponds approximately to a PyNN Population or PopulationView.
    A "node population" in SONATA corresponds approximately to a PyNN Assembly, except that
    node populations must be disjoint, whereas a given neuron may be in more than one
    PyNN Assembly (it is PyNN Populations that are disjoint).

    In this first version of the export, we simplify this by exporting each Population
    as an implicit Assembly with a single member, at the cost of losing some information about
    the PyNN network structure. This approach could be improved in future.
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
            "nodes": [
                #{
                #    "nodes_file": "$NETWORK_DIR/nodes.h5",
                #    "node_types_file": "$NETWORK_DIR/node_types.csv"
                #}
            ],
            "edges":[
                #{
                #    "edges_file": "$NETWORK_DIR/edges.h5",
                #    "edge_types_file": "$NETWORK_DIR/edge_types.csv"
                #},
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
    # - we define a separate SONATA node population for each PyNN Population
    # - we may in future use node groups to support PopulationViews
    # - Assemblies are not explicitly represented in the SONATA structure,
    #   rather their constituent Populations are exported individually.

    for i, population in enumerate(network.populations):

        config["networks"]["nodes"].append({
            "nodes_file": "$NETWORK_DIR/nodes_{}.h5".format(population.label),
            "node_types_file": "$NETWORK_DIR/node_types_{}.csv".format(population.label)
        })
        node_type_path = Template(config["networks"]["nodes"][i]["node_types_file"]).substitute(NETWORK_DIR=network_dir)
        nodes_path = Template(config["networks"]["nodes"][i]["nodes_file"]).substitute(NETWORK_DIR=network_dir)

        n = population.size
        population_label = asciify(population.label)
        csv_rows = []
        csv_columns = set()
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
        group_label = 0  #"default"
        # todo: add "population" column

        # write HDF5 file
        nodes_file = h5py.File(nodes_path, 'w')
        nodes_file.attrs["version"] = (0, 1)  # ??? unclear what is the required format or the current version!
        nodes_file.attrs["magic"] = MAGIC  # needs to be uint32
        root = nodes_file.create_group("nodes")  # todo: add attribute with network name
        default = root.create_group(population_label)  # we use a single node group for the full Population
        default.create_dataset("node_id", data=population.all_cells.astype('i4'), dtype='i4')
        default.create_dataset("node_type_id", data=i * np.ones((n,)), dtype='i2')
        default.create_dataset("node_group_id", data=np.array([group_label] * n), dtype='i2')  #S32')  # todo: calculate the max label size
                                                                          # required data type not specified. Optional?
        default.create_dataset("node_group_index", data=np.arange(n, dtype=int), dtype='i2')

        # parameters
        node_group = default.create_group(str(group_label))
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

        nodes_file.close()

        # now write csv file
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

    for i, projection in enumerate(network.projections):
        projection_label = asciify(projection.label)
        config["networks"]["edges"].append({
            "edges_file": "$NETWORK_DIR/edges_{}.h5".format(projection_label.decode('utf-8')),
            "edge_types_file": "$NETWORK_DIR/edge_types_{}.csv".format(projection_label.decode('utf-8'))
        })
        edge_type_path = Template(config["networks"]["edges"][i]["edge_types_file"]).substitute(NETWORK_DIR=network_dir)
        edges_path = Template(config["networks"]["edges"][i]["edges_file"]).substitute(NETWORK_DIR=network_dir)

        n = projection.size()
        csv_rows = []
        edge_type_info = {
            "edge_type_id": i,
            "model_template": "{}:{}".format(target.lower(),
                                             projection.synapse_type.__class__.__name__)
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
        default_edge_pop["source_node_id"].attrs["node_population"] = asciify(projection.pre.label)  # todo: handle PopualtionViews
        default_edge_pop["target_node_id"].attrs["node_population"] = asciify(projection.post.label)
        default_edge_pop.create_dataset("edge_type_id", data=i * np.ones((n,)), dtype='i2')
        default_edge_pop.create_dataset("edge_group_id", data=np.array([group_label] * n), dtype='i2') #S32')  # todo: calculate the max label size
        default_edge_pop.create_dataset("edge_group_index", data=np.arange(n, dtype=int), dtype='i2')

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


def sonata_id_to_index(population, id):
    # this relies on SONATA ids being sequential
    if "first_sonata_id" in population.annotations:
        offset = population.annotations["first_sonata_id"]
    else:
        raise Exception("Population not annotated with SONATA ids")
    return id - offset


def import_from_sonata(config_file, sim):
    """
    We map a SONATA population to a PyNN Assembly, since both allow heterogeneous cell types.
    We map a SONATA node group to a PyNN Population, since both have heterogeneous parameter
    namespaces. This may not work in all cases, since a node group can have multiple
    node types.
    We map a SONATA edge group to a PyNN Projection, i.e. a SONATA edge population may
    result in multiple PyNN Projections.
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

                dynamics_params_group = nodes_file["nodes"][np_label][str(ng_label)]['dynamics_params']
                group_mask = nodes_file["nodes"][np_label]['node_group_index'].value[mask]
                for key in dynamics_params_group.keys():
                    assert key in cell_type_cls.default_parameters
                    parameters[key] = dynamics_params_group[key].value[group_mask]

                print(parameters)

                # todo: handle spatial structure - nodes_file["nodes"][np_label][ng_label]['x'], etc.

                pop = sim.Population(node_group_size,
                                     cell_type_cls(**parameters),
                                     #label=ng_label.decode('utf-8'))
                                     label=str(ng_label))

                assembly += pop
                assembly.annotations["first_sonata_id"] = nodes_file["nodes"][np_label]['node_id'].value.min()
                print(assembly.annotations)

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
            source_node_population = edges_file["edges"][ep_label]["source_node_id"].attrs["node_population"].decode('utf-8')
            target_node_ids = edges_file["edges"][ep_label]["target_node_id"].value
            target_node_population = edges_file["edges"][ep_label]["target_node_id"].attrs["node_population"].decode('utf-8')
            edge_groups = np.unique(edges_file["edges"][ep_label]['edge_group_id'].value)

            pre = net.get_component(source_node_population)
            post = net.get_component(target_node_population)
            print(pre)
            print(post)

            synapse_params = {}
            for eg_label in edge_groups:
                print("EDGE GROUP {}".format(eg_label))
                mask = edges_file["edges"][ep_label]['edge_group_id'].value == eg_label
                source_ids = source_node_ids[mask]
                target_ids = target_node_ids[mask]
                source_index = sonata_id_to_index(pre, source_ids)
                target_index = sonata_id_to_index(post, target_ids)

                edge_type_array = edges_file["edges"][ep_label]['edge_type_id'][mask]
                edge_types = np.unique(edge_type_array)
                print("  edge_types: {}".format(edge_types))

                synapse_types = set()
                for edge_type_id in edge_types:
                    edge_type_id = str(edge_type_id)
                    prefix, synapse_type = edge_types_map[edge_type_id]["model_template"].split(":")
                    if prefix.lower() != "pynn":
                        raise NotImplementedError("Only PyNN networks currently supported.")
                    synapse_types.add(synapse_type)

                if len(synapse_types) != 1:
                    raise Exception("Heterogeneous group, not currently supported.")
                synapse_type_name = synapse_types.pop()
                synapse_type_cls = getattr(sim, synapse_type_name)
                print("  synapse_type: {}".format(synapse_type_cls))

                params_group = edges_file["edges"][ep_label][str(eg_label)]['dynamics_params']
                synapse_params[eg_label] = {}
                for key in params_group.keys():
                    synapse_params[eg_label][key] = edges_file["edges"][ep_label][str(eg_label)]['dynamics_params'][key].value

                print(source_ids)
                print(source_index)
                print(target_ids)
                print(target_index)
                print(synapse_params)
                conn_list = np.array((source_index,
                                      target_index,
                                      synapse_params[eg_label]["weight"],  # todo: also check for "syn_weight"
                                      synapse_params[eg_label]["delay"])).transpose()
                connector = sim.FromListConnector(conn_list)  # todo: handle other possible parameters
                syn = synapse_type_cls()  # todo: handle parameters from csv file
                prj = sim.Projection(pre, post, connector, syn,
                                     label="{}-{}".format(ep_label, eg_label))
                net.projections.add(prj)

    return net



