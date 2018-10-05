# encoding: utf-8
"""
Functions for exporting and importing networks to/from files
"""

import os
from os.path import join, isdir, exists
from collections import defaultdict
import shutil
from string import Template
import csv
from warnings import warn
import json
import h5py
import numpy as np
from pyNN.network import Network
try:
    basestring
except NameError:  # Python 3
    basestring = str


MAGIC = 0x0a7a


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
        with open(node_type_path, 'w') as csv_file:
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


def sonata_id_to_index(population, id):
    # this relies on SONATA ids being sequential
    if "first_sonata_id" in population.annotations:
        offset = population.annotations["first_sonata_id"]
    else:
        raise Exception("Population not annotated with SONATA ids")
    return id - offset


class NodePopulation(object):
    """ """

    @classmethod
    def from_data(cls, name, h5_data, node_types_map, config):
        """ """

        obj = cls()
        obj.name = name
        obj.node_groups = []
        obj.node_ids = h5_data['node_id']

        for ng_label in np.unique(h5_data['node_group_id'].value):
            mask = h5_data['node_group_id'].value == ng_label
            print("NODE GROUP {}, size {}".format(ng_label, mask.sum()))

            node_type_array = h5_data['node_type_id'][mask]
            node_group_index = h5_data['node_group_index'][mask]
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
        assembly = sim.Assembly(label=self.name)
        assembly.annotations["first_sonata_id"] = self.node_ids.value.min()
        for node_group in self.node_groups:
            pop = node_group.to_population(sim)
            assembly += pop
        return assembly


class NodeGroup(object):
    """ """

    def __len__(self):
        return self.node_types_array.size

    @property
    def size(self):
        return len(self)

    @classmethod
    def from_data(cls, id, node_types_array, index, h5_data, node_types_map, config):
        """ """
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
                parameters[key] = dynamics_params_group[key].value[index]

        obj.parameters = parameters
        print(parameters)

        return obj

    def __repr__(self):
        return "NodeGroup(id='{}', parameters={})".format(self.id, self.parameters)

    def get_cell_type(self, sim):
        """ """
        cell_types = set()
        model_types = self.parameters["model_type"]
        for node_type_id, model_type in model_types.items():
            if model_type not in ("point_neuron", "point_process", "virtual"):
                raise NotImplementedError("Only point neurons currently supported.")

            if model_type == "virtual":
                cell_types.add("SpikeSourceArray")
            else:
                prefix, cell_type = self.parameters["model_template"][node_type_id].split(":")
                if prefix.lower() not in ("pynn", "nrn"):
                    raise NotImplementedError("Only PyNN and NEURON-native networks currently supported.")
                cell_types.add(cell_type)

        if len(cell_types) != 1:
            raise Exception("Heterogeneous group, not currently supported.")

        cell_type_name = cell_types.pop()
        cell_type_cls = getattr(sim, cell_type_name)
        print("  cell_type: {}".format(cell_type_cls))
        return cell_type_cls

    def to_population(self, sim):
        """ """
        cell_type_cls = self.get_cell_type(sim)
        parameters = {}
        annotations = {}

        for name, value in self.parameters.items():
            if name in cell_type_cls.default_parameters:
                parameters[name] = condense(value, self.node_types_array)
            else:
                annotations[name] = value
        # todo: handle spatial structure - nodes_file["nodes"][np_label][ng_label]['x'], etc.

        cell_type = cell_type_cls(**parameters)
        pop = sim.Population(self.size,
                             cell_type,
                             label=self.id)
        pop.annotate(**annotations)
        print("--------> {}".format(pop))
        # todo: create PopulationViews if multiple node_types
        return pop


def condense(value, node_types_array):
    """ """
    if isinstance(value, np.ndarray):
        return value
    elif isinstance(value, dict):
        assert len(value) > 0
        value_array = np.array(list(value.values()))
        if np.all(value_array == value_array[0]):
            return value_array[0]
        else:
            new_value = np.ones_like(node_types_array) * np.nan
            for node_type_id, val in value.items():
                new_value[node_types_array == node_type_id] = val
            return new_value
    else:
        raise TypeError("Unexpected type. Expected Numpy array or dict, got {}".format(type(value)))


# class NodeType(object):

#     def __init__(self, **parameters):
#         for k, v in parameters.items():
#             setattr(self, k, v)


class EdgePopulation(object):
    """ """

    @classmethod
    def from_data(cls, name, h5_data, edge_types_map, config):
        """ """

        obj = cls()
        obj.name = name
        obj.source_node_ids = h5_data["source_node_id"].value
        obj.source_node_population = h5_data["source_node_id"].attrs["node_population"].decode('utf-8')
        obj.target_node_ids = h5_data["target_node_id"].value
        obj.target_node_population = h5_data["target_node_id"].attrs["node_population"].decode('utf-8')

        obj.edge_groups = []
        for eg_label in np.unique(h5_data['edge_group_id'].value):
            mask = h5_data['edge_group_id'].value == eg_label
            print("EDGE GROUP {}, size {}".format(eg_label, mask.sum()))

            edge_type_array = h5_data['edge_type_id'][mask]
            edge_group_index = h5_data['edge_group_index'][mask]
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
        pre = net.get_component(self.source_node_population)
        post = net.get_component(self.target_node_population)
        projections = []
        for edge_group in self.edge_groups:
            projection = edge_group.to_projection(pre, post, sim, self.name)
            projections.append(projection)
        return projections


class EdgeGroup(object):
    """ """

    @classmethod
    def from_data(cls, id, edge_types_array, index, source_ids, target_ids,
                  h5_data, edge_types_map, config):
        """ """
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
                parameters[key] = dynamics_params_group[key].value[index]

        obj.parameters = parameters
        print(parameters)
        return obj

    def __repr__(self):
        return "EdgeGroup(id='{}', parameters={})".format(self.id, self.parameters)

    def get_synapse_and_receptor_type(self, sim):
        """ """
        synapse_types = set()

        model_templates = self.parameters.get("model_template", None)
        if model_templates:
            for edge_type_id, model_template in model_templates.items():
                prefix, syn_type = model_template.split(":")
                if prefix.lower() not in ("pynn", "nrn"):
                    raise NotImplementedError("Only PyNN and NEURON-native networks currently supported.")
                synapse_types.add(syn_type)

            if len(synapse_types) != 1:
                raise Exception("Heterogeneous group, not currently supported.")

            synapse_type_name = synapse_types.pop()
        else:
            synapse_type_name = "StaticSynapse"

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
            receptor_type = "default"  # temporary hack to make 300-cell example work, due to PyNN bug #597
                                       # value should really be None.

        print("  synapse_type: {}".format(synapse_type_cls))
        print("  receptor_type: {}".format(receptor_type))
        return synapse_type_cls, receptor_type

    def to_projection(self, pre, post, sim, edge_population_name):
        """ """

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
        print("--------> {}".format(prj))
        return prj



def load_config(config_file):
    """ """
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
            if obj.startswith('$'):
                return Template(obj).substitute(**substitutions)
            else:
                return obj

    return traverse(config)




def read_types_file(file_path, node_or_edge):
    with open(file_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=' ', quotechar='"')
        types_table = list(csv_reader)
    types_map = {}
    id_label = "{}_type_id".format(node_or_edge)
    for row in types_table:
        types_map[int(row[id_label])] = row  # NodeType(**row)
    print(types_map)
    return types_map


def import_from_sonata(config_file, sim):
    """
    We map a SONATA population to a PyNN Assembly, since both allow heterogeneous cell types.
    We map a SONATA node group to a PyNN Population, since both have homogeneous parameter
    namespaces.
    SONATA node types are used to give different parameters to different subsets of nodes in a group.
    This can be handled in PyNN by indexing and, equivalently, by defining PopulationViews.
    We map a SONATA edge group to a PyNN Projection, i.e. a SONATA edge population may
    result in multiple PyNN Projections.
    """
    config = load_config(config_file)

    if config.get("target_simulator", None) != "PyNN":
        warn("`target_simulator` is not set to 'PyNN'. Proceeding with caution...")
        # could also easily handle target_simulator="NEST" using native models
        # NEURON also possible using native models, but a bit more involved
        # seems that target_simulator is sometimes in the circuit_config


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
    for edges_config in config["networks"]["edges"]:

        # Load edge types into edge_types_map
        edge_types_map = read_types_file(edges_config["edge_types_file"], 'edge')

        # Open edges file, check it is valid
        edges_file = h5py.File(edges_config["edges_file"], 'r')
        version = edges_file.attrs.get("version", None)
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
        net.assemblies.add(assembly)
    for edge_population in sonata_edge_populations:
        projections = edge_population.to_projections(net, sim)
        net.projections.update(projections)

    return net
