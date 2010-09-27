import nineml.user_layer as nineml
import pyNN.nineml
import pyNN.random
import pyNN.space
import math
from pprint import pprint


def generate_spiking_node_description_map():
    map = {}
    for m in pyNN.nineml.list_standard_models():
        definition_url = m.spiking_mechanism_definition_url
        if definition_url in map:
            map[definition_url].add(m)
        else:
            map[definition_url] = set([m])
    return map

def generate_post_synaptic_response_description_map():
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
    map = {}
    for c in pyNN.nineml.connectors.list_connectors():
        map[c.definition_url] = c
    return map

def reverse_map(D):
    E = {}
    for k,v in D.items():
        if v in D:
            raise Exception()
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


def resolve_parameters(P):
    global random_distributions
    random_parameters = {}
    for name,p in P.items():
        if isinstance(p.value, nineml.RandomDistribution):
            rd = p.value
            if rd.name in random_distributions:
                random_parameters[name] = random_distributions[rd.name]
            else:
                rd_name = reverse_map(pyNN.nineml.utility.random_distribution_url_map)[rd.definition.url]
                rd_param_names = pyNN.nineml.utility.random_distribution_parameter_map[rd_name]
                rd_params = [rd.parameters.parameters[rdp_name].value for rdp_name in rd_param_names]
                print "----->", rd_name, rd_params
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
    

class Network(object):
    
    def __init__(self, sim, nineml_file):
        global random_distributions
        assert isinstance(nineml_file, basestring)
        nineml_model = nineml.parse(nineml_file)

        random_distributions = {}
        
        for group in nineml_model.groups.values():
            
            # extract post-synaptic response definitions from projections
            psr_map = {}
            for projection in group.projections.values():
                if projection.target.name in psr_map:
                    psr_map[projection.target.name].add(projection.synaptic_response)
                else:
                    psr_map[projection.target.name] = set([projection.synaptic_response])
            print "psr map:", psr_map
            
            # create populations
            populations = {}
            for population in group.populations.values():
                print "\n***", population.name
                if isinstance(population.prototype, nineml.SpikingNodeType):
                    #generate dims from population.number and population.positions.structure
                    n = population.number
                    structure = population.positions.structure
                    if structure:
                        # ideally should parse abstraction layer file
                        # for now we'll just match file names
                        P = structure.parameters.parameters
                        if "Grid2D" in structure.definition.url:
                            pyNN_structure = pyNN.space.Grid2D(
                                                aspect_ratio=P["aspect_ratio"].value,
                                                dx=P["dx"].value,
                                                dy=P["dy"].value,
                                                x0=P["x0"].value,
                                                y0=P["y0"].value,
                                                fill_order=P["fill_order"].value)
                        elif "Grid3D" in structure.definition.url:
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
                        elif "Line" in structure.definition.url:
                            pyNN_structure = pyNN.space.Line(
                                                dx=P["dx"].value,
                                                x0=P["x0"].value,
                                                y0=P["y0"].value,
                                                z0=P["z0"].value)
                        else:
                            raise Exception("Structure %s not supported by PyNN" % structure)
                        
                    #determine cell class from standardcell--catalog mapping            
                    cellclass_map = generate_spiking_node_description_map()
                    possible_cell_classes_from_spiking_node = cellclass_map[population.prototype.definition.url]
                    if population.name in psr_map:
                        synapse_definition_urls = set(psr.definition.url for psr in psr_map[population.name])
                        assert len(synapse_definition_urls) == 1 # exc and inh always same model at present
                        synapse_definition_url = list(synapse_definition_urls)[0]
                        synapsetype_map = generate_post_synaptic_response_description_map()
                        possible_cell_classes_from_psr = synapsetype_map[synapse_definition_url]
                        possible_cell_classes = possible_cell_classes_from_spiking_node.intersection(possible_cell_classes_from_psr)
                        assert len(possible_cell_classes) == 1
                        cell_class = list(possible_cell_classes)[0]
                    else:
                        print "Population is not the target of any Projections, so we can choose any synapse model."
                        cell_class = list(possible_cell_classes_from_spiking_node)[0]
                    print "cell_class", cell_class
                    
                    # determine cellparams from population.prototype.parameters and standardcell.translations
                    nineml_params = population.prototype.parameters.parameters
                    if "i_offset" in cell_class.default_parameters:
                        nineml_params["offsetCurrent"] = nineml.Parameter("offsetCurrent", 0.0, "nA")
        #            if "v_init" in cell_class.default_parameters:
        #                v_rest, v_unit = nineml_params["restingPotential"].value, nineml_params["restingPotential"].unit
        #                nineml_params["initialMembranePotential"] = nineml.Parameter("initialMembranePotential", v_rest, v_unit)
                    
                    if population.name in psr_map:
                        synapse_params = list(psr_map[population.name])[0].parameters.parameters
                        #print "synapse_params", synapse_params
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
                    
                    pprint(cell_params)
                    cell_params, random_parameters = resolve_parameters(cell_params)
                    pprint(cell_params)
                    p_obj = sim.Population(n, getattr(sim, cell_class.__name__),
                                           cell_params,
                                           structure=pyNN_structure,
                                           label=population.name)
                    for name, rd in random_parameters.items():
                        p_obj.rset(name, rd)
                    
                elif isinstance(population.prototype, nineml.Group):
                    raise NotImplementedError
                else:
                    raise Exception()
                populations[population.name] = p_obj
                
            # create projections
            for projection in group.projections.values():
                connector_cls = generate_connector_map()[projection.rule.definition.url]
                print "Connector params:",
                pprint(projection.rule.parameters.parameters)
                connector_params, random_connector_params = resolve_parameters(projection.rule.parameters.parameters)
                synapse_parameters = projection.connection_type.parameters.parameters
                connector_params['weights'] = synapse_parameters['weight'].value
                connector_params['delays'] = synapse_parameters['delay'].value*scale('delay', synapse_parameters['delay'].unit)
                connector = connector_cls(**connector_params)
                
                post_celltype = getattr(pyNN.nineml.cells,
                                        populations[projection.target.name].celltype.__class__.__name__)
                target_map = reverse_map(post_celltype.synaptic_mechanism_definition_urls)
                target = target_map[projection.synaptic_response.definition.url]
                
                if projection.connection_type.definition.url != pyNN.nineml.utility.catalog_url + "/connectiontypes/static_synapse.xml":
                    raise Exception("Dynamic synapses not yet supported by the 9ML-->PyNN converter.")
                
                prj_obj = sim.Projection(
                            populations[projection.source.name],
                            populations[projection.target.name],
                            connector,
                            target=target,
                            #synapse_dynamics=,
                            label=projection.name)
                    

