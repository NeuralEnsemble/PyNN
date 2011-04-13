from __future__ import absolute_import
import subprocess
import neuron
from pyNN.models import BaseCellType
import nineml.abstraction_layer as nineml
import logging
import os
from itertools import chain

h = neuron.h
logger = logging.getLogger("PyNN")

NMODL_DIR = "nineml_mechanisms"

class NineMLCell(object):
    
    def __init__(self, **parameters):
        self.type = parameters.pop("type")
        self.source_section = h.Section()
        self.source = getattr(h, self.type.model_name)(0.5, sec=self.source_section)
        for param, value in parameters.items():
            setattr(self.source, param, value)
        # for recording
        self.spike_times = h.Vector(0)
        self.traces = {}
        self.recording_time = False
    
    def __getattr__(self, name):
        try:
            return self.__getattribute__(name)
        except AttributeError:
            if name in self.type.synapse_types:
                return self.source # source is also target
            else:
                raise AttributeError("'NineMLCell' object has no attribute or synapse type '%s'" % name)

    def record(self, active):
        if active:
            rec = h.NetCon(self.source, None)
            rec.record(self.spike_times)
        else:
            self.spike_times = h.Vector(0)

    def memb_init(self):
        # this is a bit of a hack
        for var in self.type.recordable:
            if hasattr(self, "%s_init" % var):
                initial_value = getattr(self, "%s_init" % var)
                logger.debug("Initialising %s to %g" % (var, initial_value))
                setattr(self.source, var, initial_value)


class NineMLCellType(BaseCellType):
    model = NineMLCell
    
    def __init__(self, parameters):
        BaseCellType.__init__(self, parameters)
        self.parameters["type"] = self


def _compile_nmodl(nineml_component):
    if not os.path.exists(NMODL_DIR):
        os.makedirs(NMODL_DIR)
    cwd = os.getcwd()
    os.chdir(NMODL_DIR)
    xml_file = "%s.xml" % nineml_component.name
    logger.debug("Writing NineML component to %s" % xml_file)
    nineml_component.write(xml_file)
    nineml2nmodl = __import__("9ml2nmodl")
    nineml2nmodl.write_nmodl(xml_file)
    p = subprocess.check_call(["nrnivmodl"])
    os.chdir(cwd)
    neuron.load_mechanisms(NMODL_DIR)


class _build_nineml_celltype(type):
    """
    Metaclass for building NineMLCellType subclasses
    """
    def __new__(cls, name, bases, dct):
        assert len(dct["synapse_models"]) == 1, "For now, can't handle multiple synapse models"
        combined_model = join(dct["neuron_model"],
                              dct["synapse_models"].values()[0],
                              dct["port_map"],
                              name=name)
        dct["combined_model"] = combined_model
        dct["default_parameters"] = dict((name, 1.0)
                                      for name in combined_model.parameters)
        dct["default_initial_values"] = dict((name, 0.0)
                                          for name in combined_model.state_variables)
        dct["synapse_types"] = dct["synapse_models"].keys() #really need an ordered dict
        dct["injectable"] = True # need to determine this. How??
        dct["recordable"] = [port.name for port in combined_model.analog_ports] + ['spikes']
        dct["standard_receptor_type"] = (dct["synapse_types"] == ('excitatory', 'inhibitory'))
        dct["conductance_based"] = True # how to determine this??
        dct["model_name"] = name
        logger.debug("Creating class '%s' with bases %s and dictionary %s" % (name, bases, dct))
        _compile_nmodl(combined_model)
        return type.__new__(cls, name, bases, dct)
    
    

def nineml_cell_type(name, neuron_model, port_map={}, **synapse_models):
    """
    Return a new NineMLCellType subclass.
    """
    return _build_nineml_celltype(name, (NineMLCellType,),
                                  {'neuron_model': neuron_model,
                                   'synapse_models': synapse_models,
                                   'port_map': port_map})
    

def join(c1, c2, port_map=[], name=None):
    """Create a NineML component by joining the two given components."""
    logger.debug("Joining components %s and %s with port map %s" % (c1, c2, port_map))
    logger.debug("New component will have name '%s'" % name)
    bindings = [] # TODO: combine bindings from c1 and c2
    all_ports = c1.ports_map.copy()
    all_ports.update(c2.ports_map)
    for port_name, port in all_ports.items():
        if isinstance(port, nineml.EventPort):
            all_ports.pop(port_name)
    for name1, name2 in port_map:
        assert name1 in c1.ports_map
        assert name2 in c2.ports_map
        all_ports.pop(name1)
        if name1 != name2:
            #c2.substitute(name2, name1) # need to implement this
            all_ports.pop(name2)
        
        port1 = c1.ports_map[name1]
        port2 = c2.ports_map[name2]
        assert port1.mode != port2.mode
        if port1.mode == 'send':
            send_port = port1
            port_name = name1
        else:
            send_port = port2
            port_name = name2
        if send_port.expr:
            func_args = c1.non_parameter_symbols.union(c2.non_parameter_symbols).intersection(send_port.expr.names)
            lhs = "%s(%s)" % (port_name, ",".join(func_args))
            bindings.append(nineml.Binding(lhs, send_port.expr.rhs))
            for eq in chain(c1.equations, c2.equations):
                if port_name in eq.names:
                    eq.rhs = eq.rhs_name_transform({port_name: lhs})
    regime_map = {}
    for r1 in c1.regimes:
        regime_map[r1.name] = {}
        for r2 in c2.regimes:
            kwargs = {'name': "%s_AND_%s" % (r1.name, r2.name)}
            new_regime = nineml.Regime(*r1.nodes.union(r2.nodes), **kwargs)
            regime_map[r1.name][r2.name] = new_regime
    transitions = []
    for r1 in c1.regimes:
        for r2 in c2.regimes:
            for t in r1.transitions:
                new_transition = nineml.Transition(*t.nodes,
                                                   from_=regime_map[r1.name][r2.name],
                                                   to=regime_map[t.to.name][r2.name],
                                                   condition=t.condition)
                transitions.append(new_transition)
            for t in r2.transitions:
                new_transition = nineml.Transition(*t.nodes,
                                                   from_=regime_map[r1.name][r2.name],
                                                   to=regime_map[r1.name][t.to.name],
                                                   condition=t.condition)
                transitions.append(new_transition)
    regimes = []
    for d in regime_map.values():
        regimes.extend(d.values())
    name = name or "%s__%s" % (c1.name, c2.name)
    return nineml.Component(name,
                            regimes=regimes,
                            transitions=transitions,
                            ports=all_ports.values(),
                            bindings=bindings)



        
    