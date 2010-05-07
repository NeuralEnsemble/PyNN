
import pyNN.cells
import pyNN.common
import pyNN.standardmodels
import pprint
from nineml.user_layer import parse

class IF_curr_exp(pyNN.cells.IF_curr_exp):
    
    translations = pyNN.standardmodels.build_translations(
        ('v_rest',     'None'),
        ('v_reset',    'V_r'),
        ('cm',         'None'),
        ('tau_m',      'tau'),
        ('tau_refrac', 'tau_rp'),
        ('tau_syn_E',  'None'),
        ('tau_syn_I',  'None'),
        ('v_thresh',   'theta'),
        ('i_offset',   'None'),
        ('v_init',     'None'),
    )
    
    def __init__(self, parameters):
        pyNN.cells.IF_curr_exp.__init__(self, parameters)
    
class SpikeSourcePoisson(pyNN.cells.SpikeSourcePoisson):

    translations = pyNN.standardmodels.build_translations(
        ('rate',     'nu_ext'),
        ('start',    'None'),
        ('duration', 'None'),
    )

spiking_models = {
    'IaF_tau': IF_curr_exp,
    'Poisson': SpikeSourcePoisson,
}

connectors = {
    'convergent': {
        'number-based-random': {
            'model': 'FixedNumberPreConnector',
            'translations': {'n': 'number'},
        },
        
    },
    'divergent': {
        'number-based-random': {
            'model': 'FixedNumberPostConnector',
            'translations': {'n': 'number'},
        },
    },
    'unknown': {
        'unknown': {
            'model': 'UnknownConnector',
            'translations': {},
        }
    }
}

def scale(parameter_type, unit):
    """Primitive unit handling. Should probably use Piquant, or similar."""
    ms = 1
    mV = 1
    s = 1000
    V = 1000
    Hz = 1
    standard_units = {
        'membraneConstant': ms,
        'threshold': mV,
        'refractoryTime': ms,
        'resetPotential': mV,
        'frequency': Hz,
        'delay': ms,
        'step': mV,
    }
    original_unit = eval(unit)
    return standard_units[parameter_type]*original_unit


class Network(object):
    
    def __init__(self, nineml_file):
        assert isinstance(nineml_file, basestring) # is a URL or file
        nineml_model = parse(nineml_file)
        nineml_model.resolve_references()
        nineml_model.resolve_synapse_types()
        self._build_cell_types(nineml_model.node_types)
        self._build_populations(nineml_model.populations)
        self._build_sets(nineml_model.sets)
        self._build_projections(nineml_model.assemblies)
                                

    def _build_cell_types(self, node_types):
        self.cell_types = {}
        for node_type in node_types:
            if node_type.type in spiking_models:
                standard_cell_model = spiking_models[node_type.type]({})
                model_name = standard_cell_model.__class__.__name__
                nineml_parameters = {}
                for p in node_type.parameters:
                    nineml_parameters[p.name] = p.value*scale(p.type, p.unit)
                parameters = standard_cell_model.reverse_translate(nineml_parameters)
                self.cell_types[node_type.name] = {'model': model_name,
                                                   'parameters': parameters}
            else:
                raise Exception("PyNN cannot handle cell type '%s'" % node_type.type)
                
    def _build_populations(self, populations):
        self.populations = {}
        for p in populations:
            self.populations[p.name] = '%s = sim.Population(%s, sim.%s, cell_types["%s"]["parameters"], label="%s")' % (
                p.name.replace(" ", "_").lower(),
                p.number,
                self.cell_types[p.prototype]["model"],
                p.prototype,
                p.name)

    def _build_sets(self, sets):
        self.sets = {}
        for set in sets:
            self.sets[set.name] = set.sources[0].references

    def _build_projections(self, assemblies):
        self.projections = []
        for assembly in assemblies:
            for projection in assembly.projections:
                connector_type = connectors[projection.rule.direction][projection.rule.type]["model"]
                connector_params = {}
                for name, tr_name in connectors[projection.rule.direction][projection.rule.type]["translations"].items():
                    connector_params[name] = getattr(projection.rule, tr_name)
                synapse_parameters = {}
                for p in projection.synapse_type.parameters:
                    synapse_parameters[p.type] = p
                connector_params['weights'] = synapse_parameters['step'].value*scale('step', synapse_parameters['step'].unit)
                connector_params['delays'] = synapse_parameters['delay'].value*scale('delay', synapse_parameters['delay'].unit)
                if projection.synapse_type.type == 'Delta':
                    if float(connector_params['weights']) >= 0:
                        target_type = 'excitatory'
                    else:
                        target_type = 'inhibitory'
                else:
                    raise Exception('Only Delta synapses supported so far')
                if projection.target in self.populations:        
                    self.projections.append('%s_projection = sim.Projection(%s, %s,\nsim.%s(%s), target="%s", label="%s")' % (
                        projection.name.lower(),
                        projection.source.replace(" ", "_").lower(),
                        projection.target.replace(" ", "_").lower(),
                        connector_type,
                        ",".join("%s=%s" % item for item in connector_params.items()),
                        target_type,
                        projection.name))
                elif projection.target in self.sets:
                    for target in self.sets[projection.target]:
                        self.projections.append('%s2%s_projection = sim.Projection(%s, %s,\n sim.%s(%s), target="%s", label="%s to %s")' % (
                        projection.name.lower(),
                        target.replace(" ", "_").lower(),
                        projection.source.replace(" ", "_").lower(),
                        target.replace(" ", "_").lower(),
                        connector_type,
                        ",".join("%s=%s" % item for item in connector_params.items()),
                        target_type,
                        projection.name,
                        target))
                else:
                    raise Exception("Unknown target '%s'" % projection.target)

    def export(self):
        lines = []
        lines.append("import pyNN.nest as sim")
        lines.append("sim.setup()")
        lines.append("cell_types = %s" % pprint.pformat(self.cell_types))
        lines.extend(self.populations.values())
        lines.extend(self.projections)
        return lines
    
    

