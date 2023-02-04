"""
Enables creating neuronal network models in PyNN from a 9ML description.

Classes:
    Network -- container for a network model.

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

import nineml.user as nineml
import nineml.abstraction as al
import pyNN.nineml
import pyNN.random
import pyNN.space
from .utility import build_random_distribution


def scale(quantity):
    """Primitive unit handling. Should probably use Piquant, quantities, or similar."""
    factors = {
        'ms': 1,
        'mV': 1,
        's': 1000,
        'V': 1000,
        'Hz': 1,
        'nF': 1,
        'nA': 1,
        'Mohm': 1,  # ??
        'unknown': 1,
        'unitless': 1,
    }
    if quantity.units:
        return quantity.value * factors[quantity.units.name]
    else:  # dimensionless
        return quantity.value


def resolve_parameters(nineml_component, random_distributions, resolve="properties",
                       qualified_names=True):
    """
    Turn a 9ML ParameterSet or InitialValueSet into a Python dict, including turning 9ML
    RandomDistribution objects into PyNN RandomDistribution objects.
    """
    P = {}
    for p in getattr(nineml_component, resolve):
        if qualified_names:
            qname = "%s_%s" % (nineml_component.name, p.name)
        else:
            qname = p.name
        if p.is_random():
            rd = p.random_distribution
            if rd.name in random_distributions:
                P[qname] = random_distributions[rd.name]
            else:
                rand_distr = build_random_distribution(rd)
                P[qname] = rand_distr
                random_distributions[rd.name] = rand_distr
        elif p.value in ('True', 'False'):
            P[qname] = eval(p.value)
        elif isinstance(p.value, str):
            P[qname] = p.value
        else:
            P[qname] = scale(p)
    return P


def _build_structure(nineml_structure):
    """
    Return a PyNN Structure object that corresponds to the provided 9ML
    Structure object.

    For now, we do this by mapping names rather than parsing the 9ML abstraction
    layer file.
    """
    if nineml_structure:
        # ideally should parse abstraction layer file
        # for now we'll just match file names
        P = nineml_structure.parameters
        if "Grid2D" in nineml_structure.definition.url:
            pyNN_structure = pyNN.space.Grid2D(
                aspect_ratio=P["aspect_ratio"].value,
                dx=P["dx"].value,
                dy=P["dy"].value,
                x0=P["x0"].value,
                y0=P["y0"].value,
                fill_order=P["fill_order"].value)
        elif "Grid3D" in nineml_structure.definition.url:
            pyNN_structure = pyNN.space.Grid3D(
                aspect_ratioXY=P["aspect_ratioXY"].value,
                aspect_ratioXZ=P["aspect_ratioXZ"].value,
                dx=P["dx"].value,
                dy=P["dy"].value,
                dz=P["dz"].value,
                x0=P["x0"].value,
                y0=P["y0"].value,
                z0=P["z0"].value,
                fill_order=P["fill_order"].value)
        elif "Line" in nineml_structure.definition.url:
            pyNN_structure = pyNN.space.Line(
                dx=P["dx"].value,
                x0=P["x0"].value,
                y0=P["y0"].value,
                z0=P["z0"].value)
        else:
            raise Exception("nineml_structure %s not supported by PyNN" % nineml_structure)
    else:
        pyNN_structure = None
    return pyNN_structure


def _generate_variable_name(name):
    return name.replace(" ", "_").replace("-", "")


class Network(object):
    """
    Container for a neuronal network model, created from a 9ML user-layer file.

    There is not a one-to-one mapping between 9ML and PyNN concepts. The two
    main differences are:
        (1) a 9ML Group contains both neurons (populations) and connections
            (projections), whereas a PyNN Assembly contains only neurons: the
            connections are contained in global Projections.
        (2) in 9ML, the post-synaptic response is defined in the projection,
            whereas in PyNN it is a property of the target population.

    Attributes:
        assemblies  -- a dict containing PyNN Assembly objects
        projections -- a dict containing PyNN Projection objects
    """

    def __init__(self, sim, nineml_model):
        """
        Instantiate a network from a 9ML file, in the specified simulator.
        """
        global random_distributions
        self.sim = sim
        if isinstance(nineml_model, str):
            self.nineml_model = nineml.Network.read(nineml_model)
        elif isinstance(nineml_model, nineml.Network):
            self.nineml_model = nineml_model
        else:
            raise TypeError(
                "nineml_model must be a nineml.Network instance or the path to a NineML XML file.")
        self.random_distributions = {}
        self.populations = {}
        self.assemblies = {}
        self.projections = {}
        _tmp = __import__(sim.__name__, globals(), locals(), ["nineml"])
        self._nineml_module = _tmp.nineml
        self._build()

    def _build(self):

        # extract post-synaptic response definitions from projections
        self.psr_map = {}
        for projection in self.nineml_model.projections.values():
            if isinstance(projection.destination, nineml.Selection):
                target_populations = projection.destination.evaluate()
                # target_populations = [x[0] for x in projection.destination.evaluate()]
                # just take the population, not the slice
            else:
                assert isinstance(projection.destination, nineml.Population)
                target_populations = [projection.destination]
            for target_population in target_populations:
                if target_population.name in self.psr_map:
                    target_psr = self.psr_map[target_population.name]
                    target_psr['port_connections'].update(
                        projection.port_connections)
                    # hack? what about clashes?
                    target_psr['response_component'] = projection.response
                else:
                    target_psr = {'port_connections': set(projection.port_connections),
                                  'response_component': projection.response}

        # create populations
        for population in self.nineml_model.populations.values():
            self._build_population(population)
        for selection in self.nineml_model.selections.values():
            self._evaluate_selection(selection)

        # create projections
        for projection in self.nineml_model.projections.values():
            self._build_projection(projection)

    def _generate_cell_type_and_parameters(self, nineml_population):
        """

        """
        neuron_model = nineml_population.cell.component_class
        neuron_namespace = _generate_variable_name(nineml_population.cell.name)
        synapse_models = {}
        response_components = {}
        connections = []
        weight_vars = {}
        if nineml_population.name in self.psr_map:
            for pc in self.psr_map[nineml_population.name]['port_connections']:
                if pc._receive_role == 'destination' and pc._send_role == 'response':
                    synapse_name = _generate_variable_name(pc.sender.name)
                    synapse_models[synapse_name] = pc.send_class
                    assert pc.send_class.query.analog_ports_map[pc.send_port].mode == 'send'
                    assert neuron_model.query.analog_ports_map[pc.receive_port].mode in (
                        'recv', 'reduce')
                    connections.append(("%s.%s" % (synapse_name, pc.send_port),
                                        "%s.%s" % (neuron_namespace, pc.receive_port)))
                    #    else:
                    #        assert neuron_model.query.analog_ports_map[nrn_port].mode == 'send'
                    #        connections.append((
                    #            "%s.%s" % (neuron_namespace, nrn_port),
                    #            "%s.%s" % (synapse_name, psr_port)
                    #        ))
                elif pc._receive_role == 'response' and pc._send_role == 'destination':
                    raise NotImplementedError
                elif pc._receive_role == 'response' and pc._send_role == 'plasticity':
                    synapse_name = _generate_variable_name(pc.receiver.name)
                    weight_vars[synapse_name] = "%s_%s" % (synapse_name, pc.receive_port)
                else:
                    raise Exception("Unexpected")
            response_components[synapse_name] = \
                self.psr_map[nineml_population.name]['response_component']
        subnodes = {neuron_namespace: neuron_model}
        subnodes.update(synapse_models)
        combined_model = al.Dynamics(name=_generate_variable_name(nineml_population.name),
                                     subnodes=subnodes)
        # now connect ports
        for connection in connections:
            combined_model.connect_ports(*connection)

        celltype_cls = self._nineml_module.nineml_cell_type(
            combined_model.name, combined_model, weight_vars)
        cell_params = resolve_parameters(nineml_population.cell, self.random_distributions)
        for response_component in response_components.values():
            cell_params.update(resolve_parameters(response_component, self.random_distributions))

        return celltype_cls, cell_params

    def _build_population(self, nineml_population):
        # assert isinstance(nineml_population.cell, nineml.SpikingNodeType)
        # to implement in NineML library
        n = nineml_population.size
        if nineml_population.positions is not None:
            pyNN_structure = _build_structure(nineml_population.positions.structure)
        else:
            pyNN_structure = None
        # TODO: handle explicit list of positions
        cell_class, cell_params = self._generate_cell_type_and_parameters(nineml_population)

        p_obj = self.sim.Population(n, cell_class,
                                    cell_params,
                                    structure=pyNN_structure,
                                    initial_values=resolve_parameters(nineml_population.cell,
                                                                      self.random_distributions,
                                                                      resolve="initial_values"),
                                    label=nineml_population.name)
        self.populations[p_obj.label] = p_obj

    def _evaluate_selection(self, nineml_selection):
        new_assembly = self.sim.Assembly(label=nineml_selection.name)
        for population in nineml_selection.evaluate():
            new_assembly += self.populations[population.name]
        # for population, selector in nineml_selection.populations:
        #    parent = self.populations['population.name']
        #    if selector is not None:
        #        view = eval("parent[%s]" % selector)
        #        view.label = nineml_selection.name
        #        new_assembly += view
        #    else:
        #        new_assembly += parent
        self.assemblies[nineml_selection.name] = new_assembly

    def _build_connector(self, nineml_projection):
        connector_params = resolve_parameters(
            nineml_projection.connectivity, self.random_distributions, qualified_names=False)
        translations = {'number': ('n', int)}  # todo: complete for all standard connectors
        translated_params = {}
        for name, value in connector_params.items():
            tname, f = translations[name]
            translated_params[tname] = f(value)
        builtin_connectors = {
            'RandomFanIn': self.sim.FixedNumberPreConnector,
            'RandomFanOut': self.sim.FixedNumberPostConnector,
            'AllToAll': self.sim.AllToAllConnector,
            'OneToOne': self.sim.OneToOneConnector
        }
        connector_type = builtin_connectors[nineml_projection.connectivity.component_class.name]
        connector = connector_type(**translated_params)
        return connector
        # inline_csa = nineml_projection.rule.definition.component._connection_rule[0]
        # cset = inline_csa(*connector_params.values()).cset
        # TODO: csa should handle named parameters; handle random params
        # return self.sim.CSAConnector(cset)

    def _build_synapse_dynamics(self, nineml_projection):
        # for now, just use static synapse
        # HACK  - only works if there is a single parameter called "weight"
        #         to be sorted out when we try some real plastic synapses
        parameters = resolve_parameters(
            nineml_projection.plasticity, self.random_distributions,
            "properties", qualified_names=False)
        parameters.update(resolve_parameters(
            nineml_projection.plasticity, self.random_distributions,
            "initial_values", qualified_names=False))
        parameters["delay"] = nineml_projection.delay.value
        return self.sim.StaticSynapse(**parameters)

    def _build_projection(self, nineml_projection):
        populations = {}
        for p in self.populations.values():
            populations[p.label] = p
        for a in self.assemblies.values():
            populations[a.label] = a

        connector = self._build_connector(nineml_projection)
        receptor_type = nineml_projection.response.name
        try:
            assert receptor_type in populations[nineml_projection.destination.name].receptor_types
        except Exception:
            raise
        synapse_dynamics = self._build_synapse_dynamics(nineml_projection)

        prj_obj = self.sim.Projection(populations[nineml_projection.source.name],
                                      populations[nineml_projection.destination.name],
                                      connector,
                                      receptor_type=receptor_type,
                                      synapse_type=synapse_dynamics,
                                      label=nineml_projection.name)
        # need to add assembly label to make the name unique
        self.projections[prj_obj.label] = prj_obj

    def describe(self):
        description = "Network model generated from a 9ML description, consisting of:\n  "
        description += "\n  ".join(a.describe() for a in self.assemblies.values()) + "\n"
        description += "\n  ".join(prj.describe() for prj in self.projections.values())
        return description


if __name__ == "__main__":
    # For testing purposes: read in the network and print its description
    # if using the nineml or neuroml backend, re-export the network as XML
    # (this doesn't work, but it should).
    import os
    from pyNN.utility import get_script_args
    nineml_file, simulator_name = get_script_args(
        2, "Please specify the 9ML file and the simulator backend.")
    exec("import pyNN.%s as sim" % simulator_name)

    sim.setup(filename="%s_export.xml" % os.path.splitext(nineml_file)[0])  # noqa: F821
    network = Network(sim, nineml_file)                                     # noqa: F821
    print(network.describe())
    sim.end()                                                               # noqa: F821
