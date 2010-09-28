"""
Enables creating neuronal network models in PyNN from a 9ML description.

For now, there is no support for the 9ML abstraction layer. Instead we use a
mapping between URIs and PyNN classes. The URIs may point to an abstraction layer
file which is asserted to represent the same model as the PyNN class. In time,
the plan is to implement all these abstraction layer files, and then this module
will actually parse them rather than using name mapping.

Classes:
    Network -- container for a network model.
    
"""

import nineml.user_layer as nineml
import pyNN.nineml
import pyNN.random
import pyNN.space
import re

def generate_spiking_node_description_map():
    """
    Return a mapping between URIs and StandardCellType classes.
    """
    map = {}
    for m in pyNN.nineml.list_standard_models():
        definition_url = m.spiking_mechanism_definition_url
        if definition_url in map:
            map[definition_url].add(m)
        else:
            map[definition_url] = set([m])
    return map

def generate_post_synaptic_response_description_map():
    """
    Return a mapping between URIs and StandardCellType classes.
    """
    map = {}
    for m in pyNN.nineml.list_standard_models():
        if m.synapse_types: # exclude spike sources
            definition_url = m.synaptic_mechanism_definition_urls['excitatory']
            assert definition_url == m.synaptic_mechanism_definition_urls['inhibitory']
            if definition_url in map:
                map[definition_url].add(m)
            else:
                map[definition_url] = set([m])
    return map

def generate_connector_map():
    """
    Return a mapping between URIs and Connector classes.
    """
    map = {}
    for c in pyNN.nineml.connectors.list_connectors():
        map[c.definition_url] = c
    return map

def reverse_map(D):
    """
    Return a dict having D.values() as its keys and D.keys() as its values.
    """
    E = {}
    for k,v in D.items():
        if v in E:
            raise KeyError("Cannot reverse this mapping, as it is not one-to-one ('%s' would map to both '%s' and '%s')" % (v, E[v], k))
        E[v] = k
    return E

def scale(parameter_name, unit):
    """Primitive unit handling. Should probably use Piquant, or similar."""
    ms = 1
    mV = 1
    s = 1000
    V = 1000
    Hz = 1
    nF = 1
    nA = 1
    standard_units = { # there is a very similar dict in pyNN.nineml.utility
        'tau_m': ms,
        'v_thresh': mV,
        'tau_refrac': ms,
        'v_reset': mV,
        'rate': Hz,
        'duration': ms,
        'start': ms,
        'cm': nF,
        'e_rev_E': mV,
        'e_rev_I': mV,
        'tau_syn_E': ms,
        'tau_syn_I': ms,
        'i_offset': nA,
        'v_init': mV,
        'v_rest': mV,
        'delay': ms,
    }
    if unit == 'unknown':
        return 1.0
    else:
        original_unit = eval(unit)
        return standard_units[parameter_name]*original_unit

def resolve_parameters(P, random_distributions):
    """
    Turn a 9ML ParameterSet into a Python dict, including turning 9ML
    RandomDistribution objects into PyNN RandomDistribution objects.
    """
    random_parameters = {}
    for name,p in P.items():
        if isinstance(p.value, nineml.RandomDistribution):
            rd = p.value
            if rd.name in random_distributions:
                random_parameters[name] = random_distributions[rd.name]
            else:
                rd_name = reverse_map(pyNN.nineml.utility.random_distribution_url_map)[rd.definition.url]
                rd_param_names = pyNN.nineml.utility.random_distribution_parameter_map[rd_name]
                rd_params = [rd.parameters[rdp_name].value for rdp_name in rd_param_names]
                rand_distr = pyNN.random.RandomDistribution(rd_name, rd_params)
                random_parameters[name] = rand_distr
                random_distributions[rd.name] = rand_distr
            P[name] = -999
        elif p.value in ('True', 'False'):
            P[name] = eval(p.value)
        elif isinstance(p.value, basestring):
            P[name] = p.value
        else:
            P[name] = p.value*scale(name, p.unit)
    return P, random_parameters
    
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
                                aspect_ratioXY=P["aspect_ratioXZ"].value,
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
    
    def __init__(self, sim, nineml_file):
        """
        Instantiate a network from a 9ML file, in the specified simulator.
        """
        global random_distributions
        assert isinstance(nineml_file, basestring)
        self.sim = sim
        self.nineml_model = nineml.parse(nineml_file)
        self.random_distributions = {}
        self.assemblies = {}
        self.projections = {}
        self._build()
        
    def _build(self):
        for group in self.nineml_model.groups.values():
            self._handle_group(group)
            
    def _handle_group(self, group):
        # create an Assembly
        self.assemblies[group.name] = self.sim.Assembly(group.name)
        
        # extract post-synaptic response definitions from projections
        self.psr_map = {}
        for projection in group.projections.values():
            if projection.target.name in self.psr_map:
                self.psr_map[projection.target.name].add(projection.synaptic_response)
            else:
                self.psr_map[projection.target.name] = set([projection.synaptic_response])
        
        # create populations
        for population in group.populations.values():
            self._build_population(population, self.assemblies[group.name])
        for selection in group.selections.values():
            self._evaluate_selection(selection, self.assemblies[group.name])
        
        # create projections
        for projection in group.projections.values():
            self._build_projection(projection, self.assemblies[group.name])
    
    def _determine_cell_type(self, nineml_population):
        """Determine cell class from standardcell--catalog mapping"""
        cellclass_map = generate_spiking_node_description_map()
        possible_cell_classes_from_spiking_node = cellclass_map[nineml_population.prototype.definition.url]
        if nineml_population.name in self.psr_map:
            synapse_definition_urls = set(psr.definition.url for psr in self.psr_map[nineml_population.name])
            assert len(synapse_definition_urls) == 1 # exc and inh always same model at present
            synapse_definition_url = list(synapse_definition_urls)[0]
            synapsetype_map = generate_post_synaptic_response_description_map()
            possible_cell_classes_from_psr = synapsetype_map[synapse_definition_url]
            possible_cell_classes = possible_cell_classes_from_spiking_node.intersection(possible_cell_classes_from_psr)
            assert len(possible_cell_classes) == 1
            cell_class = list(possible_cell_classes)[0]
        else:
            print "Population '%s' is not the target of any Projections, so we can choose any synapse model." % nineml_population.name
            cell_class = list(possible_cell_classes_from_spiking_node)[0]
        return cell_class
    
    def _determine_cell_type_and_parameters(self, nineml_population):
        """
        Determine cell class from standardcell--catalog mapping and cellparams
        from nineml_population.prototype.parameters and standardcell.translations
        """
        cell_class = self._determine_cell_type(nineml_population)
        
        nineml_params = nineml_population.prototype.parameters
        if "i_offset" in cell_class.default_parameters:
            nineml_params["offsetCurrent"] = nineml.Parameter("offsetCurrent", 0.0, "nA")
        
        if nineml_population.name in self.psr_map:
            synapse_params = list(self.psr_map[nineml_population.name])[0].parameters
            for target in "excitatory", "inhibitory":
                for name, value in synapse_params.items():
                    nineml_params["%s_%s" % (target, name)] = value
        else: # take default values, since the synapses won't be used in any case.
            synapse_params = {"excitatory_decayTimeConstant": nineml.Parameter("decayTimeConstant", 5.0, "ms"),
                              "inhibitory_decayTimeConstant": nineml.Parameter("decayTimeConstant", 5.0, "ms"),
                              "excitatory_reversalPotential": nineml.Parameter("reversalPotential", 0.0, "mV"),
                              "inhibitory_reversalPotential": nineml.Parameter("reversalPotential", -70.0, "mV")}
            nineml_params.update(synapse_params)
        
        cell_params = cell_class.reverse_translate(nineml_params)
        cell_params, random_params = resolve_parameters(cell_params, self.random_distributions)
        return cell_class, cell_params, random_params
    
    def _build_population(self, nineml_population, assembly):
            if isinstance(nineml_population.prototype, nineml.SpikingNodeType):
                n = nineml_population.number
                pyNN_structure = _build_structure(nineml_population.positions.structure)
                cell_class, cell_params, random_parameters = self._determine_cell_type_and_parameters(nineml_population)
                
                p_obj = self.sim.Population(n, getattr(self.sim, cell_class.__name__),
                                            cell_params,
                                            structure=pyNN_structure,
                                            label=nineml_population.name)
                for name, rd in random_parameters.items():
                    p_obj.rset(name, rd)
                
            elif isinstance(nineml_population.prototype, nineml.Group):
                raise NotImplementedError
            else:
                raise Exception()
            
            assembly.populations.append(p_obj)

    def _evaluate_selection(self, nineml_selection, assembly):
        selection = str(nineml_selection.condition)
        # look away now, this isn't pretty
        pattern = re.compile(r'\(\("population\[@name\]"\) == \("(?P<name>[\w ]+)"\)\) and \("population\[@id\]" in "(?P<slice>\d*:\d*:\d*)"\)')
        match = pattern.match(selection)
        if match:
            name = match.groupdict()["name"]
            slice = match.groupdict()["slice"]
            parent = assembly.get_population(name)
            view = eval("parent[%s]" % slice)
            view.label = nineml_selection.name
            assembly.populations.append(view)
        else:
            raise Exception("Can't evaluate selection")

    def _build_connector(self, nineml_projection):
        connector_cls = generate_connector_map()[nineml_projection.rule.definition.url]
        connector_params, random_connector_params = resolve_parameters(nineml_projection.rule.parameters,
                                                                       self.random_distributions)
        synapse_parameters = nineml_projection.connection_type.parameters
        connector_params['weights'] = synapse_parameters['weight'].value
        connector_params['delays'] = synapse_parameters['delay'].value*scale('delay', synapse_parameters['delay'].unit)
        return connector_cls(**connector_params)

    def _determine_postsynaptic_target_name(self, nineml_projection, populations):
        target = None
        post_celltype = getattr(pyNN.nineml.cells,
                                populations[nineml_projection.target.name].celltype.__class__.__name__)
        try:
            target_map = reverse_map(post_celltype.synaptic_mechanism_definition_urls)
            target = target_map[nineml_projection.synaptic_response.definition.url]
        except KeyError: # post_celltype.synaptic_mechanism_definition_urls is not a one-to-one mapping
            for name in post_celltype.synaptic_mechanism_definition_urls.keys():
                if nineml_projection.synaptic_response.name.find(name) == 0: # very fragile
                    target = name
                    break
            if target is None:    
                raise Exception("Unable to determine post-synaptic target corresponding to %s.\nPost-synaptic cell supports %s" % (nineml_projection.synaptic_response, post_celltype.synaptic_mechanism_definition_urls))
                # we could at this point use weights and is_conductance to infer it
        return target

    def _build_synapse_dynamics(self, nineml_projection):
        if nineml_projection.connection_type.definition.url != pyNN.nineml.utility.catalog_url + "/connectiontypes/static_synapse.xml":
            raise Exception("Dynamic synapses not yet supported by the 9ML-->PyNN converter.")
        return None

    def _build_projection(self, nineml_projection, assembly):
        populations = {}
        for p in assembly.populations:
            populations[p.label] = p
            
        connector = self._build_connector(nineml_projection)
        target = self._determine_postsynaptic_target_name(nineml_projection, populations)
        synapse_dynamics = self._build_synapse_dynamics(nineml_projection)
        
        prj_obj = self.sim.Projection(
                    populations[nineml_projection.source.name],
                    populations[nineml_projection.target.name],
                    connector,
                    target=target,
                    synapse_dynamics=synapse_dynamics,
                    label=nineml_projection.name)
        self.projections[prj_obj.label] = prj_obj # need to add assembly label to make the name unique

    def describe(self):
        description = "Network model generated from a 9ML description, consisting of:\n  "
        description += "\n  ".join(a.describe() for a in self.assemblies.values()) + "\n"
        description += "\n  ".join(prj.describe() for prj in self.projections.values())
        return description


if __name__ == "__main__":
    # For testing purposes: read in the network and print its description
    # if using the nineml or neuroml backend, re-export the network as XML (this doesn't work, but it should).
    import sys, os
    from pyNN.utility import get_script_args
    nineml_file, simulator_name = get_script_args(2, "Please specify the 9ML file and the simulator backend.")  
    exec("import pyNN.%s as sim" % simulator_name)
    
    sim.setup(filename="%s_export.xml" % os.path.splitext(nineml_file)[0])
    network = Network(sim, nineml_file)
    print network.describe()
    sim.end()
