"""
Support cell types defined in 9ML with NEURON.

Requires the 9ml2nmodl script to be on the path.

Classes:
    NineMLCell       - a single neuron instance
    NineMLCellType   - base class for cell types, not used directly

Functions:
    nineml_cell_type - return a new NineMLCellType subclass

Constants:
    NMODL_DIR        - subdirectory to which NMODL mechanisms will be written

:copyright: Copyright 2006-2011 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

from __future__ import absolute_import # Not compatible with Python 2.4
import subprocess
import neuron
from pyNN.models import BaseCellType
import nineml.abstraction_layer as nineml
import logging
import os
import re
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


def _compile_nmodl(nineml_component, weight_variables): # weight variables should really be within component
    """
    Generate NMODL code for the 9ML component, run "nrnivmodl" and then load
    the mechanisms into NEURON.
    """
    if not os.path.exists(NMODL_DIR):
        os.makedirs(NMODL_DIR)
    cwd = os.getcwd()
    os.chdir(NMODL_DIR)
    xml_file = "%s.xml" % nineml_component.name
    logger.debug("Writing NineML component to %s" % xml_file)
    nineml_component.write(xml_file)
    nineml2nmodl = __import__("9ml2nmodl")
    nineml2nmodl.write_nmodl(xml_file, weight_variables) # weight variables should really come from xml file
    p = subprocess.check_call(["nrnivmodl"])
    os.chdir(cwd)
    neuron.load_mechanisms(NMODL_DIR)


def _add_prefix(synapse_model, prefix, port_map):
    """
    Add a prefix to all variables in `synapse_model`, except for variables with
    receive ports and specified in `port_map`.
    """
    synapse_model.__cache__ = {}
    exclude = []
    new_port_map = []
    for name1, name2 in port_map:
        if synapse_model.ports_map[name2].mode == 'recv':
            exclude.append(name2)
            new_port_map.append((name1, name2))
        else:
            new_port_map.append((name1, prefix + '_' + name2))
    synapse_model.add_prefix(prefix + '_', exclude=exclude)
    return new_port_map


class _build_nineml_celltype(type):
    """
    Metaclass for building NineMLCellType subclasses
    """
    def __new__(cls, name, bases, dct):
        # join the neuron and synapse components into a single component
        combined_model = dct["neuron_model"]
        for label in dct["synapse_models"].keys():
            port_map = dct["port_map"][label]
            port_map = _add_prefix(dct["synapse_models"][label], label, port_map)
            dct["weight_variables"][label] = label + "_" + dct["weight_variables"][label]
            combined_model = join(combined_model,
                                  dct["synapse_models"][label],
                                  port_map,
                                  name=name)
        dct["combined_model"] = combined_model
        # set class attributes required for a PyNN cell type class
        dct["default_parameters"] = dict((name, 1.0)
                                      for name in combined_model.parameters)
        dct["default_initial_values"] = dict((name, 0.0)
                                          for name in combined_model.state_variables)
        dct["synapse_types"] = dct["synapse_models"].keys() #really need an ordered dict
        dct["injectable"] = True # need to determine this. How??
        dct["recordable"] = [port.name for port in combined_model.analog_ports] + ['spikes', 'regime']
        dct["standard_receptor_type"] = (dct["synapse_types"] == ('excitatory', 'inhibitory'))
        dct["conductance_based"] = True # how to determine this??
        dct["model_name"] = name
        logger.debug("Creating class '%s' with bases %s and dictionary %s" % (name, bases, dct))
        # generate and compile NMODL code, then load the mechanism into NEUORN
        _compile_nmodl(combined_model, dct["weight_variables"]) # weight variables should really be stored within combined_model
        return type.__new__(cls, name, bases, dct)
    
    

def nineml_cell_type(name, neuron_model, port_map={}, weight_variables={}, **synapse_models):
    """
    Return a new NineMLCellType subclass.
    """
    return _build_nineml_celltype(name, (NineMLCellType,),
                                  {'neuron_model': neuron_model,
                                   'synapse_models': synapse_models,
                                   'port_map': port_map,
                                   'weight_variables': weight_variables})


def join(c1, c2, port_map=[], name=None):
    """Create a NineML component by joining the two given components."""
    logger.debug("Joining components %s and %s with port map %s" % (c1, c2, port_map))
    logger.debug("New component will have name '%s'" % name)
    # combine bindings from c1 and c2
    bindings = {}
    for b in chain(c1.bindings, c2.bindings):
        bindings[b.name] = b
    # combine ports (some will later be removed)
    all_ports = c1.ports_map.copy()
    all_ports.update(c2.ports_map)
    # event ports do not be passed to the constructor, as they are attached to transitions
    for port_name, port in all_ports.items():
        if isinstance(port, nineml.EventPort):
            all_ports.pop(port_name)
    # connect ports.
    # currently, when ports are connected they disappear. It might be better to
    # explicitly keep the ports in the new component but mark them as connected
    for name1, name2 in port_map:
        assert name1 in c1.ports_map, "%s is not in %s" % (name1, c1.ports_map.keys())
        assert name2 in c2.ports_map, "%s is not in %s" % (name2, c2.ports_map.keys())

        port1 = c1.ports_map[name1]
        port2 = c2.ports_map[name2]
        assert port1.mode != port2.mode
        if port1.mode == 'send':
            send_port = port1
            recv_port = port2
            send_port_name = name1
            recv_port_name = name2
        else:
            send_port = port2
            recv_port = port1
            send_port_name = name2
            recv_port_name = name1
        # when connecting ports in which the send port has an expression, need
        # to create a binding for this expression in the new component
        if send_port.expr:
            func_args = c1.non_parameter_symbols.union(c2.non_parameter_symbols).intersection(send_port.expr.names)
            lhs = "%s(%s)" % (send_port_name, ",".join(func_args))
            send_binding = nineml.Binding(lhs, send_port.expr.rhs)
            bindings[send_binding.name] = send_binding
            for eq in chain(c1.equations, c2.equations):
                if send_port_name in eq.names:
                    eq.rhs = eq.rhs_name_transform({send_port_name: lhs})            
            if recv_port.mode == 'reduce':
                # need to retain reduce ports as they can be connected to in a future join
                if recv_port_name in bindings:
                    # this reduce port has already been connected to, so combine using its reduce_op
                    reduce_binding = bindings[recv_port_name]
                    func_args = func_args.union(reduce_binding.args)
                    lhs = "%s(%s)" % (recv_port_name, ",".join(func_args))
                    rhs = recv_port.reduce_op.join([reduce_binding.rhs, send_binding.lhs])
                else:
                    # this is the first time this reduce port has been connected to
                    lhs = "%s(%s)" % (recv_port_name, ",".join(func_args))
                    rhs = send_binding.lhs
                bindings[recv_port_name] = nineml.Binding(lhs, rhs)
                recv_port.connected = True
            else:
                all_ports.pop(name1)
        else:
            if recv_port.mode == 'reduce':
                raise NotImplementedError
            else:
                all_ports.pop(name1)

        if name1 != name2:
            #c2.substitute(name2, name1) # need to implement this. Currently this all only works if name1 == name2
                                         # probably needs to happen sooner in the function
            all_ports.pop(name2)

    # where parameters have become bindings due to connecting ports, replace
    # bare names with function calls in the equations
    for bname, binding in bindings.items():
        for eq in chain(c1.equations, c2.equations):
            if bname in eq.names:
                print "#### replacing %s by %s" % (bname, binding.lhs)
                pattern = re.compile(r'%s(\([\w\, ]*\))?' % bname)
                m = pattern.search(eq.rhs)
                if m:
                    eq.rhs = pattern.sub(binding.lhs, eq.rhs)
                else:
                    eq.rhs = eq.rhs_name_transform({bname: binding.lhs})

    # create new regimes from all possible combinations of the regimes from the
    # two components
    regime_map = {}
    for r1 in c1.regimes:
        regime_map[r1.name] = {}
        for r2 in c2.regimes:
            if r1.name == r2.name:
                new_name = r1.name
            else:
                new_name = "%s_AND_%s" % (r1.name, r2.name)
            kwargs = {'name': new_name}
            new_regime = nineml.Regime(*r1.nodes.union(r2.nodes), **kwargs)
            regime_map[r1.name][r2.name] = new_regime
    # create transitions between all the new regimes
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
                            bindings=bindings.values())
