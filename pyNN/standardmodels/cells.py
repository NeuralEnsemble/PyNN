"""
Definition of default parameters (and hence, standard parameter names) for
standard cell models.

Plain integrate-and-fire models:
    IF_curr_exp
    IF_curr_alpha
    IF_cond_exp
    IF_cond_alpha

Integrate-and-fire with adaptation:
    IF_cond_exp_gsfa_grr
    EIF_cond_alpha_isfa_ista
    EIF_cond_exp_isfa_ista

Integrate-and-fire model for use with the FACETS hardware
    IF_facets_hardware1

Hodgkin-Huxley model
    HH_cond_exp

Spike sources (input neurons)
    SpikeSourcePoisson
    SpikeSourceArray
    SpikeSourceInhGamma

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.
"""

# flake8: noqa (ignore E221)

from copy import deepcopy
import operator
from functools import reduce

from ..parameters import ArrayParameter, Sequence
from .base import StandardCellType, StandardCellTypeComponent


class IF_curr_alpha(StandardCellType):
    """
    Leaky integrate and fire model with fixed threshold and alpha-function-
    shaped post-synaptic current.
    """

    default_parameters = {
        'v_rest':   -65.0,  # Resting membrane potential in mV.
        'cm':         1.0,  # Capacity of the membrane in nF
        'tau_m':     20.0,  # Membrane time constant in ms.
        'tau_refrac': 0.1,  # Duration of refractory period in ms.
        'tau_syn_E':  0.5,  # Rise time of the excitatory synaptic alpha function in ms.
        'tau_syn_I':  0.5,  # Rise time of the inhibitory synaptic alpha function in ms.
        'i_offset':   0.0,  # Offset current in nA
        'v_reset':  -65.0,  # Reset potential after a spike in mV.
        'v_thresh': -50.0,  # Spike threshold in mV.
    }
    recordable = ['spikes', 'v']
    conductance_based = False
    default_initial_values = {
        'v': -65.0,  # 'v_rest',
        'isyn_exc': 0.0,
        'isyn_inh': 0.0,
    }
    units = {
        'v': 'mV',
        'isyn_exc': 'nA',
        'isyn_inh': 'nA',
        'v_rest': 'mV',
        'cm': 'nF',
        'tau_m': 'ms',
        'tau_refrac': 'ms',
        'tau_syn_E': 'ms',
        'tau_syn_I': 'ms',
        'i_offset': 'nA',
        'v_reset': 'mV',
        'v_thresh': 'mV',
    }


class IF_curr_exp(StandardCellType):
    """
    Leaky integrate and fire model with fixed threshold and
    decaying-exponential post-synaptic current. (Separate synaptic currents for
    excitatory and inhibitory synapses.
    """

    default_parameters = {
        'v_rest':   -65.0,  # Resting membrane potential in mV.
        'cm':         1.0,  # Capacity of the membrane in nF
        'tau_m':     20.0,  # Membrane time constant in ms.
        'tau_refrac': 0.1,  # Duration of refractory period in ms.
        'tau_syn_E':  5.0,  # Decay time of excitatory synaptic current in ms.
        'tau_syn_I':  5.0,  # Decay time of inhibitory synaptic current in ms.
        'i_offset':   0.0,  # Offset current in nA
        'v_reset':  -65.0,  # Reset potential after a spike in mV.
        'v_thresh': -50.0,  # Spike threshold in mV.
    }
    recordable = ['spikes', 'v']
    conductance_based = False
    default_initial_values = {
        'v': -65.0,  # 'v_rest',
        'isyn_exc': 0.0,
        'isyn_inh': 0.0,
    }
    units = {
        'v': 'mV',
        'isyn_exc': 'nA',
        'isyn_inh': 'nA',
        'v_rest': 'mV',
        'cm': 'nF',
        'tau_m': 'ms',
        'tau_refrac': 'ms',
        'tau_syn_E': 'ms',
        'tau_syn_I': 'ms',
        'i_offset': 'nA',
        'v_reset': 'mV',
        'v_thresh': 'mV',
    }


class IF_curr_delta(StandardCellType):
    """
    Leaky integrate and fire model with fixed threshold.
    Synaptic inputs produce step changes in membrane potential.
    """

    default_parameters = {
        'v_rest':   -65.0,  # Resting membrane potential in mV.
        'cm':         1.0,  # Capacity of the membrane in nF
        'tau_m':     20.0,  # Membrane time constant in ms.
        'tau_refrac': 0.1,  # Duration of refractory period in ms.
        'i_offset':   0.0,  # Offset current in nA
        'v_reset':  -65.0,  # Reset potential after a spike in mV.
        'v_thresh': -50.0,  # Spike threshold in mV.
    }
    recordable = ['spikes', 'v']
    conductance_based = False
    voltage_based_synapses = True
    default_initial_values = {
        'v': -65.0,
    }
    units = {
        'v': 'mV',
        'v_rest': 'mV',
        'cm': 'nF',
        'tau_m': 'ms',
        'tau_refrac': 'ms',
        'i_offset': 'nA',
        'v_reset': 'mV',
        'v_thresh': 'mV',
    }


class IF_cond_alpha(StandardCellType):
    """
    Leaky integrate and fire model with fixed threshold and alpha-function-
    shaped post-synaptic conductance.
    """

    default_parameters = {
        'v_rest':   -65.0,  # Resting membrane potential in mV.
        'cm':         1.0,  # Capacity of the membrane in nF
        'tau_m':     20.0,  # Membrane time constant in ms.
        'tau_refrac': 0.1,  # Duration of refractory period in ms.
        'tau_syn_E':  0.3,  # Rise time of the excitatory synaptic alpha function in ms.
        'tau_syn_I':  0.5,  # Rise time of the inhibitory synaptic alpha function in ms.
        'e_rev_E':    0.0,  # Reversal potential for excitatory input in mV
        'e_rev_I':  -70.0,  # Reversal potential for inhibitory input in mV
        'v_thresh': -50.0,  # Spike threshold in mV.
        'v_reset':  -65.0,  # Reset potential after a spike in mV.
        'i_offset':   0.0,  # Offset current in nA
    }
    recordable = ['spikes', 'v', 'gsyn_exc', 'gsyn_inh']
    default_initial_values = {
        'v': -65.0,  # 'v_rest',
        'gsyn_exc': 0.0,
        'gsyn_inh': 0.0,
    }
    units = {
        'v': 'mV',
        'gsyn_exc': 'uS',
        'gsyn_inh': 'uS',
        'v_rest': 'mV',
        'cm': 'nF',
        'tau_m': 'ms',
        'tau_refrac': 'ms',
        'tau_syn_E': 'ms',
        'tau_syn_I': 'ms',
        'e_rev_E': 'mV',
        'e_rev_I': 'mV',
        'v_thresh': 'mV',
        'v_reset': 'mV',
        'i_offset': 'nA',
    }


class IF_cond_exp(StandardCellType):
    """
    Leaky integrate and fire model with fixed threshold and
    exponentially-decaying post-synaptic conductance.
    """

    default_parameters = {
        'v_rest':   -65.0,  # Resting membrane potential in mV.
        'cm':         1.0,  # Capacity of the membrane in nF
        'tau_m':     20.0,  # Membrane time constant in ms.
        'tau_refrac': 0.1,  # Duration of refractory period in ms.
        'tau_syn_E':  5.0,  # Decay time of the excitatory synaptic conductance in ms.
        'tau_syn_I':  5.0,  # Decay time of the inhibitory synaptic conductance in ms.
        'e_rev_E':    0.0,  # Reversal potential for excitatory input in mV
        'e_rev_I':  -70.0,  # Reversal potential for inhibitory input in mV
        'v_thresh': -50.0,  # Spike threshold in mV.
        'v_reset':  -65.0,  # Reset potential after a spike in mV.
        'i_offset':   0.0,  # Offset current in nA
    }
    recordable = ['spikes', 'v', 'gsyn_exc', 'gsyn_inh']
    default_initial_values = {
        'v': -65.0,  # 'v_rest',
        'gsyn_exc': 0.0,
        'gsyn_inh': 0.0,
    }
    units = {
        'v': 'mV',
        'gsyn_exc': 'uS',
        'gsyn_inh': 'uS',
        'v_rest': 'mV',
        'cm': 'nF',
        'tau_m': 'ms',
        'tau_refrac': 'ms',
        'tau_syn_E': 'ms',
        'tau_syn_I': 'ms',
        'e_rev_E': 'mV',
        'e_rev_I': 'mV',
        'v_thresh': 'mV',
        'v_reset': 'mV',
        'i_offset': 'nA',
    }


class IF_cond_exp_gsfa_grr(StandardCellType):
    """
    Linear leaky integrate and fire model with fixed threshold,
    decaying-exponential post-synaptic conductance, conductance based
    spike-frequency adaptation, and a conductance-based relative refractory
    mechanism.

    See: Muller et al (2007) Spike-frequency adapting neural ensembles: Beyond
    mean-adaptation and renewal theories. Neural Computation 19: 2958-3010.

    See also: EIF_cond_alpha_isfa_ista
    """

    default_parameters = {
        'v_rest':    -65.0,  # Resting membrane potential in mV.
        'cm':          1.0,  # Capacity of the membrane in nF
        'tau_m':      20.0,  # Membrane time constant in ms.
        'tau_refrac':  0.1,  # Duration of refractory period in ms.
        'tau_syn_E':   5.0,  # Decay time of the excitatory synaptic conductance in ms.
        'tau_syn_I':   5.0,  # Decay time of the inhibitory synaptic conductance in ms.
        'e_rev_E':     0.0,  # Reversal potential for excitatory input in mV
        'e_rev_I':   -70.0,  # Reversal potential for inhibitory input in mV
        'v_thresh':  -50.0,  # Spike threshold in mV.
        'v_reset':   -65.0,  # Reset potential after a spike in mV.
        'i_offset':    0.0,  # Offset current in nA
        'tau_sfa':   100.0,  # Time constant of spike-frequency adaptation in ms
        'e_rev_sfa': -75.0,  # spike-frequency adaptation conductance reversal potential in mV
        'q_sfa':      15.0,  # Quantal spike-frequency adaptation conductance increase in nS
        'tau_rr':      2.0,  # Time constant of the relative refractory mechanism in ms
        'e_rev_rr':  -75.0,  # relative refractory mechanism conductance reversal potential in mV
        'q_rr':     3000.0   # Quantal relative refractory conductance increase in nS
    }
    recordable = ['spikes', 'v', 'g_r', 'g_s', 'gsyn_exc', 'gsyn_inh']
    default_initial_values = {
        'v': -65.0,  # 'v_rest',
        'g_r': 0.0,
        'g_s': 0.0,
        'gsyn_exc': 0.0,
        'gsyn_inh': 0.0,
    }
    units = {
        'v': 'mV',
        'g_r': 'nS',
        'g_s': 'nS',
        'gsyn_exc': 'uS',
        'gsyn_inh': 'uS',
        'v_rest': 'mV',
        'cm': 'nF',
        'tau_m': 'ms',
        'tau_refrac': 'ms',
        'tau_syn_E': 'ms',
        'tau_syn_I': 'ms',
        'e_rev_E': 'mV',
        'e_rev_I': 'mV',
        'v_thresh': 'mV',
        'v_reset': 'mV',
        'i_offset': 'nA',
        'tau_sfa': 'ms',
        'e_rev_sfa': 'mV',
        'q_sfa': 'nS',
        'tau_rr': 'ms',
        'e_rev_rr': 'mV',
        'q_rr': 'nS',
    }


class IF_facets_hardware1(StandardCellType):
    """
    Leaky integrate and fire model with conductance-based synapses and fixed
    threshold as it is resembled by the FACETS Hardware Stage 1.

    The following parameters can be assumed for a corresponding software
    simulation: cm = 0.2 nF, tau_refrac = 1.0 ms, e_rev_E = 0.0 mV.
    For further details regarding the hardware model see the FACETS-internal Wiki:
    https://facets.kip.uni-heidelberg.de/private/wiki/index.php/WP7_NNM
    """

    default_parameters = {
        'g_leak':    40.0,  # nS
        'tau_syn_E': 30.0,  # ms
        'tau_syn_I': 30.0,  # ms
        'v_reset':  -80.0,  # mV
        'e_rev_I':  -80.0,  # mV,
        'v_rest':   -65.0,  # mV
        'v_thresh': -55.0   # mV
    }
    recordable = ['spikes', 'v', 'gsyn_exc', 'gsyn_inh']
    default_initial_values = {
        'v': -65.0,  # 'v_rest',
        'gsyn_exc': 0.0,
        'gsyn_inh': 0.0,
    }
    units = {
        'v': 'mV',
        'gsyn_exc': 'uS',
        'gsyn_inh': 'uS',
        'g_leak': 'nS',
        'tau_syn_E': 'ms',
        'tau_syn_I': 'ms',
        'v_reset': 'mV',
        'e_rev_I': 'mV',
        'v_rest': 'mV',
        'v_thresh': 'mV',
    }


class HH_cond_exp(StandardCellType):
    """Single-compartment Hodgkin-Huxley model.
    Reference:
    Traub & Miles, Neuronal Networks of the Hippocampus, Cambridge, 1991.
    """

    default_parameters = {
        'gbar_Na':     20.0,   # uS
        'gbar_K':       6.0,   # uS
        'g_leak':       0.01,  # uS
        'cm':           0.2,   # nF
        'v_offset':   -63.0,   # mV
        'e_rev_Na':    50.0,
        'e_rev_K':    -90.0,
        'e_rev_leak': -65.0,
        'e_rev_E':      0.0,
        'e_rev_I':    -80.0,
        'tau_syn_E':    0.2,   # ms
        'tau_syn_I':    2.0,
        'i_offset':     0.0,   # nA

    }
    recordable = ['spikes', 'v', 'gsyn_exc', 'gsyn_inh']
    receptor_types = ('excitatory', 'inhibitory', 'source_section.gap')
    default_initial_values = {
        'v': -65.0,  # 'v_rest',
        'gsyn_exc': 0.0,
        'gsyn_inh': 0.0,
        'h': 1.0,
        'm': 0.0,
        'n': 0.0,
    }
    units = {
        'v': 'mV',
        'gsyn_exc': 'uS',
        'gsyn_inh': 'uS',
        'gbar_Na': 'uS',
        'gbar_K': 'uS',
        'g_leak': 'uS',
        'cm': 'nF',
        'v_offset': 'mV',
        'e_rev_Na': 'mV',
        'e_rev_K': 'mV',
        'e_rev_leak': 'mV',
        'e_rev_E': 'mV',
        'e_rev_I': 'mV',
        'tau_syn_E': 'ms',
        'tau_syn_I': 'ms',
        'i_offset': 'nA',
        'h': '',
        'm': '',
        'n': '',
    }


class EIF_cond_alpha_isfa_ista(StandardCellType):
    """
    Exponential integrate and fire neuron with spike triggered and
    sub-threshold adaptation currents (isfa, ista reps.) according to:

    Brette R and Gerstner W (2005) Adaptive Exponential Integrate-and-Fire Model
    as an Effective Description of Neuronal Activity. J Neurophysiol 94:3637-3642

    See also: IF_cond_exp_gsfa_grr, EIF_cond_exp_isfa_ista
    """

    default_parameters = {
        'cm':         0.281,   # Capacitance of the membrane in nF
        'tau_refrac': 0.1,     # Duration of refractory period in ms.
        'v_spike':  -40.0,     # Spike detection threshold in mV.
        'v_reset':  -70.6,     # Reset value for V_m after a spike. In mV.
        'v_rest':   -70.6,     # Resting membrane potential (Leak reversal potential) in mV.
        'tau_m':      9.3667,  # Membrane time constant in ms
        'i_offset':   0.0,     # Offset current in nA
        'a':          4.0,     # Subthreshold adaptation conductance in nS.
        'b':          0.0805,  # Spike-triggered adaptation in nA
        'delta_T':    2.0,     # Slope factor in mV
        'tau_w':    144.0,     # Adaptation time constant in ms
        'v_thresh': -50.4,     # Spike initiation threshold in mV
        'e_rev_E':    0.0,     # Excitatory reversal potential in mV.
        'tau_syn_E':  5.0,     # Rise time of excitatory synaptic conductance in ms (alpha function). # noqa: E501
        'e_rev_I':  -80.0,     # Inhibitory reversal potential in mV.
        'tau_syn_I':  5.0,     # Rise time of the inhibitory synaptic conductance in ms (alpha function). # noqa: E501
    }
    recordable = ['spikes', 'v', 'w', 'gsyn_exc', 'gsyn_inh']
    default_initial_values = {
        'v': -70.6,  # 'v_rest',
        'w': 0.0,
        'gsyn_exc': 0.0,
        'gsyn_inh': 0.0,
    }
    units = {
        'v': 'mV',
        'w': 'nA',
        'gsyn_exc': 'uS',
        'gsyn_inh': 'uS',
        'cm': 'nF',
        'tau_refrac': 'ms',
        'v_spike': 'mV',
        'v_reset': 'mV',
        'v_rest': 'mV',
        'tau_m': 'ms',
        'i_offset': 'nA',
        'a': 'nS',
        'b': 'nA',
        'delta_T': 'mV',
        'tau_w': 'ms',
        'v_thresh': 'mV',
        'e_rev_E': 'mV',
        'tau_syn_E': 'ms',
        'e_rev_I': 'mV',
        'tau_syn_I': 'ms',
    }


class EIF_cond_exp_isfa_ista(StandardCellType):
    """
    Exponential integrate and fire neuron with spike triggered and
    sub-threshold adaptation currents (isfa, ista reps.) according to:

    Brette R and Gerstner W (2005) Adaptive Exponential Integrate-and-Fire Model
    as an Effective Description of Neuronal Activity. J Neurophysiol 94:3637-3642

    See also: IF_cond_exp_gsfa_grr, EIF_cond_alpha_isfa_ista
    """

    default_parameters = {
        'cm':         0.281,   # Capacitance of the membrane in nF
        'tau_refrac': 0.1,     # Duration of refractory period in ms.
        'v_spike':  -40.0,     # Spike detection threshold in mV.
        'v_reset':  -70.6,     # Reset value for V_m after a spike. In mV.
        'v_rest':   -70.6,     # Resting membrane potential (Leak reversal potential) in mV.
        'tau_m':      9.3667,  # Membrane time constant in ms
        'i_offset':   0.0,     # Offset current in nA
        'a':          4.0,     # Subthreshold adaptation conductance in nS.
        'b':          0.0805,  # Spike-triggered adaptation in nA
        'delta_T':    2.0,     # Slope factor in mV
        'tau_w':    144.0,     # Adaptation time constant in ms
        'v_thresh': -50.4,     # Spike initiation threshold in mV
        'e_rev_E':    0.0,     # Excitatory reversal potential in mV.
        'tau_syn_E':  5.0,     # Decay time constant of excitatory synaptic conductance in ms.
        'e_rev_I':  -80.0,     # Inhibitory reversal potential in mV.
        'tau_syn_I':  5.0,     # Decay time constant of the inhibitory synaptic conductance in ms.
    }
    recordable = ['spikes', 'v', 'w', 'gsyn_exc', 'gsyn_inh']
    default_initial_values = {
        'v': -70.6,  # 'v_rest',
        'w': 0.0,
        'gsyn_exc': 0.0,
        'gsyn_inh': 0.0,
    }
    units = {
        'v': 'mV',
        'w': 'nA',
        'gsyn_exc': 'uS',
        'gsyn_inh': 'uS',
        'cm': 'nF',
        'tau_refrac': 'ms',
        'v_spike': 'mV',
        'v_reset': 'mV',
        'v_rest': 'mV',
        'tau_m': 'ms',
        'i_offset': 'nA',
        'a': 'nS',
        'b': 'nA',
        'delta_T': 'mV',
        'tau_w': 'ms',
        'v_thresh': 'mV',
        'e_rev_E': 'mV',
        'tau_syn_E': 'ms',
        'e_rev_I': 'mV',
        'tau_syn_I': 'ms',
    }


class LIF(StandardCellTypeComponent):
    """
    Leaky integrate and fire neuron
    """

    default_parameters = {
        'cm':         1.0,     # Capacitance of the membrane in nF
        'tau_refrac': 0.1,     # Duration of refractory period in ms.
        'v_reset':  -65.0,     # Reset value for V_m after a spike. In mV.
        'v_rest':   -65.0,     # Resting membrane potential (Leak reversal potential) in mV.
        'tau_m':     20.0,     # Membrane time constant in ms
        'i_offset':   0.0,     # Offset current in nA
        'v_thresh': -50.0,     # Spike initiation threshold in mV
    }
    recordable = ['spikes', 'v']
    injectable = True
    default_initial_values = {
        'v': -70.6,  # 'v_rest'
    }
    units = {
        'v': 'mV',
        'w': 'nA',
        'cm': 'nF',
        'tau_refrac': 'ms',
        'v_reset': 'mV',
        'v_rest': 'mV',
        'tau_m': 'ms',
        'i_offset': 'nA',
        'v_thresh': 'mV',
    }


class AdExp(StandardCellTypeComponent):
    """
    Exponential integrate and fire neuron with spike triggered and
    sub-threshold adaptation currents according to:

    Brette R and Gerstner W (2005) Adaptive Exponential Integrate-and-Fire Model
    as an Effective Description of Neuronal Activity. J Neurophysiol 94:3637-3642
    """

    default_parameters = {
        'cm':         0.281,   # Capacitance of the membrane in nF
        'tau_refrac': 0.1,     # Duration of refractory period in ms.
        'v_spike':  -40.0,     # Spike detection threshold in mV.
        'v_reset':  -70.6,     # Reset value for V_m after a spike. In mV.
        'v_rest':   -70.6,     # Resting membrane potential (Leak reversal potential) in mV.
        'tau_m':      9.3667,  # Membrane time constant in ms
        'i_offset':   0.0,     # Offset current in nA
        'a':          4.0,     # Subthreshold adaptation conductance in nS.
        'b':          0.0805,  # Spike-triggered adaptation in nA
        'delta_T':    2.0,     # Slope factor in mV
        'tau_w':    144.0,     # Adaptation time constant in ms
        'v_thresh': -50.4,     # Spike initiation threshold in mV
    }
    recordable = ['spikes', 'v', 'w']
    injectable = True
    default_initial_values = {
        'v': -70.6,  # 'v_rest',
        'w': 0.0
    }
    units = {
        'v': 'mV',
        'w': 'nA',
        'cm': 'nF',
        'tau_refrac': 'ms',
        'v_spike': 'mV',
        'v_reset': 'mV',
        'v_rest': 'mV',
        'tau_m': 'ms',
        'i_offset': 'nA',
        'a': 'nS',
        'b': 'nA',
        'delta_T': 'mV',
        'tau_w': 'ms',
        'v_thresh': 'mV',
    }


class PointNeuron(StandardCellType):

    def __init__(self, neuron, **post_synaptic_receptors):
        self.neuron = neuron
        self.post_synaptic_receptors = post_synaptic_receptors
        for psr in post_synaptic_receptors.values():
            psr.set_parent(self)
        self.parameter_space = deepcopy(self.neuron.parameter_space)
        for name, psr in self.post_synaptic_receptors.items():
            self.parameter_space.add_child(name, psr.parameter_space)

    @property
    def receptor_types(self):
        return list(sorted(self.post_synaptic_receptors.keys()))

    @property
    def conductance_based(self):
        psr_conductance_based = set(psr.conductance_based
                                    for psr in self.post_synaptic_receptors.values())
        if len(psr_conductance_based) > 1:
            raise Exception("Cannot mix conductance-based and current-based synaptic receptors")
        psr_conductance_based, = psr_conductance_based
        return psr_conductance_based

    @property
    def recordable(self):
        return self.neuron.recordable + [
            f"{receptor_type_name}.{variable}"
            for receptor_type_name in self.receptor_types
            for variable in self.post_synaptic_receptors[receptor_type_name].recordable
        ]

    @property
    def scale_factors(self):
        scf = self.neuron.scale_factors.copy()
        for name, psr in self.post_synaptic_receptors.items():
            for variable, scale_factor in psr.scale_factors.items():
                scf[f"{name}.{variable}"] = scale_factor
        return scf

    @property
    def units(self):
        _units = self.neuron.units.copy()
        for name, psr in self.post_synaptic_receptors.items():
            for variable, un in psr.units.items():
                _units[f"{name}.{variable}"] = un
        return _units

    @property
    def default_initial_values(self):
        divs = self.neuron.default_initial_values.copy()
        for name, psr in self.post_synaptic_receptors.items():
            for variable, div in psr.default_initial_values.items():
                divs[f"{name}.{variable}"] = div
        return divs

    def simple_parameters(self):
        """Return a list of parameters for which there is a one-to-one
        correspondance between standard and native parameter values."""
        return self.neuron.simple_parameters() + list(
            set.union(*[set(psr.simple_parameters())
                        for psr in self.post_synaptic_receptors.values()]))

    def scaled_parameters(self):
        """Return a list of parameters for which there is a unit change between
        standard and native parameter values."""
        return self.neuron.scaled_parameters() + list(
            set.union(*[set(psr.scaled_parameters())
                        for psr in self.post_synaptic_receptors.values()]))

    def computed_parameters(self):
        """Return a list of parameters whose values must be computed from
        more than one other parameter."""
        return self.neuron.computed_parameters() + list(
            set.union(*[set(psr.computed_parameters())
                        for psr in self.post_synaptic_receptors.values()]))

    def computed_parameters_include(self, parameter_names):
        return (
            self.neuron.computed_parameters_include(parameter_names)
            or reduce(operator.or_,                                         # noqa: W503
                      [psr.computed_parameters_include(parameter_names)
                       for psr in self.post_synaptic_receptors.values()])
        )


class Izhikevich(StandardCellType):
    """
    Izhikevich spiking model with a quadratic non-linearity according to:

    E. Izhikevich (2003), IEEE transactions on neural networks, 14(6)

        dv/dt = 0.04*v^2 + 5*v + 140 - u + I
        du/dt = a*(b*v - u)

    Synapses are modeled as Dirac delta currents (voltage step), as in the original model

    NOTE: name should probably be changed to match standard nomenclature,
    e.g. QIF_cond_delta_etc_etc, although keeping "Izhikevich" as an alias would be good

    """

    default_parameters = {
        'a':       0.02,  # (/ms)
        'b':        0.2,  # (/ms)
        'c':      -65.0,  # (mV) aka 'v_reset'
        'd':        2.0,  # (mV/ms) Reset value for u after a spike.
        'i_offset': 0.0   # (nA)
    }
    recordable = ['spikes', 'v', 'u']
    conductance_based = False
    voltage_based_synapses = True
    default_initial_values = {
        'v': -70.0,  # mV
        'u': -14.0   # mV/ms
    }
    units = {
        'v': 'mV',
        'u': 'mV/ms',
        'a': '/ms',
        'b': '/ms',
        'c': 'mV',
        'd': 'mV/ms',
        'i_offset': 'nA',
    }


class GIF_cond_exp(StandardCellType):
    """
    The GIF model is a leaky integrate-and-fire model including a spike-triggered current eta(t),
    a moving threshold gamma(t) and stochastic spike emission.

    References:
      [1] Mensi, S., Naud, R., Pozzorini, C., Avermann, M., Petersen, C. C., &
      Gerstner, W. (2012). Parameter extraction and classification of three cortical
      neuron types reveals two distinct adaptation mechanisms.
      Journal of Neurophysiology, 107(6), 1756-1775.
      [2] Pozzorini, C., Mensi, S., Hagens, O., Naud, R., Koch, C., & Gerstner, W.
      (2015). Automated High-Throughput Characterization of Single Neurons by Means of
      Simplified Spiking Models. PLoS Comput Biol, 11(6), e1004275.
    """

    default_parameters = {
        'v_rest':     -65.0,  # Resting membrane potential in mV.
        'cm':           1.0,  # Capacity of the membrane in nF
        'tau_m':       20.0,  # Membrane time constant in ms.
        'tau_refrac':   4.0,  # Duration of refractory period in ms.
        'tau_syn_E':    5.0,  # Decay time of the excitatory synaptic conductance in ms.
        'tau_syn_I':    5.0,  # Decay time of the inhibitory synaptic conductance in ms.
        'e_rev_E':      0.0,  # Reversal potential for excitatory input in mV
        'e_rev_I':    -70.0,  # Reversal potential for inhibitory input in mV
        'v_reset':    -65.0,  # Reset potential after a spike in mV.
        'i_offset':     0.0,  # Offset current in nA
        'delta_v':      0.5,  # Threshold sharpness in mV.
        'v_t_star':   -48.0,  # Threshold baseline in mV.
        'lambda0':      1.0,  # Firing intensity at threshold in Hz.
        'tau_eta':    ArrayParameter([1.0, 10.0, 100.0]),  # Time constants for spike-triggered current in ms.        # noqa: E501
        'tau_gamma':  ArrayParameter([1.0, 10.0, 100.0]),  # Time constants for spike-frequency adaptation in ms.     # noqa: E501
        'a_eta':      ArrayParameter([1.0, 1.0, 1.0]),     # Post-spike increments for spike-triggered current in ms. # noqa: E501
        'a_gamma':    ArrayParameter([1.0, 1.0, 1.0]),     # Post-spike increments for moving threshold in mV         # noqa: E501
    }

    recordable = ['spikes', 'v', 'gsyn_exc', 'gsyn_inh', 'i_eta', 'v_t']
    default_initial_values = {
        'v': -65.0,
        'v_t': -48.0,
        'i_eta': 0.0,
        'gsyn_exc': 0.0,
        'gsyn_inh': 0.0,
    }
    units = {
        'v': 'mV',
        'gsyn_exc': 'uS',
        'gsyn_inh': 'uS',
        'i_eta': 'nA',
        'v_t': 'mV',
        'v_rest': 'mV',
        'cm': 'nF',
        'tau_m': 'ms',
        'tau_refrac': 'ms',
        'tau_syn_E': 'ms',
        'tau_syn_I': 'ms',
        'e_rev_E': 'mV',
        'e_rev_I': 'mV',
        'v_reset': 'mV',
        'i_offset': 'nA',
        'delta_v': 'mV',
        'v_t_star': 'mV',
        'lambda0': 'Hz',
        'tau_eta': 'ms',
        'tau_gamma': 'ms',
        'a_eta': 'nA',
        'a_gamma': 'mV',
    }


class SpikeSourcePoisson(StandardCellType):
    """Spike source, generating spikes according to a Poisson process."""

    default_parameters = {
        'rate':     1.0,  # Mean spike frequency (Hz)
        'start':    0.0,  # Start time (ms)
        'duration': 1e10  # Duration of spike sequence (ms)
    }
    recordable = ['spikes']
    injectable = False
    receptor_types = ()
    units = {
        'rate': 'Hz',
        'start': 'ms',
        'duration': 'ms',
    }


class SpikeSourcePoissonRefractory(StandardCellType):
    """Spike source, generating spikes according to a Poisson process with dead time"""

    default_parameters = {
        'rate':       1.0,  # Mean spike frequency (Hz)
        'tau_refrac': 0.0,  # Minimum time between spikes (ms)
        'start':      0.0,  # Start time (ms)
        'duration':   1e10  # Duration of spike sequence (ms)
    }
    recordable = ['spikes']
    injectable = False
    receptor_types = ()
    units = {
        'rate': 'Hz',
        'tau_refrac': 'ms',
        'start': 'ms',
        'duration': 'ms',
    }


class SpikeSourceGamma(StandardCellType):
    """Spike source, generating spikes according to a gamma process.

    The mean inter-spike interval is given by alpha/beta
    """

    default_parameters = {
        'alpha':       2,  # shape (order) parameter of the gamma distribution
        'beta':      1.0,  # rate parameter of the gamma distribution (Hz)
        'start':     0.0,  # Start time (ms)
        'duration': 1e10,  # Duration of spike sequence (ms)
    }
    recordable = ['spikes']
    injectable = False
    receptor_types = ()
    units = {
        'alpha': 'dimensionless',
        'beta': 'Hz',
        'start': 'ms',
        'duration': 'ms',
    }


class SpikeSourceInhGamma(StandardCellType):
    """
    Spike source, generating realizations of an inhomogeneous gamma process,
    employing the thinning method.

    See: Muller et al (2007) Spike-frequency adapting neural ensembles: Beyond
    mean-adaptation and renewal theories. Neural Computation 19: 2958-3010.
    """

    default_parameters = {
        'a':        Sequence([1.0]),  # time histogram of parameter a of a gamma distribution (dimensionless)  # noqa: E501
        'b':        Sequence([1.0]),  # time histogram of parameter b of a gamma distribution (seconds)        # noqa: E501
        'tbins':    Sequence([0.0]),  # time bins of the time histogram of a,b in units of ms
        'start':    0.0,              # Start time (ms)
        'duration': 1e10              # Duration of spike sequence (ms)
    }
    recordable = ['spikes']
    injectable = False
    receptor_types = ()
    units = {
        'a': 'dimensionless',
        'b': 's',
        'tbins': 'ms',
        'start': 'ms',
        'duration': 'ms',
    }


class SpikeSourceArray(StandardCellType):
    """Spike source generating spikes at the times given in the spike_times array."""

    default_parameters = {'spike_times': Sequence([])}  # list or numpy array containing spike times in milliseconds.  # noqa: E501
    recordable = ['spikes']
    injectable = False
    receptor_types = ()
    units = {
        'spike_times': 'ms',
    }
