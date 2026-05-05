"""
Example of using a cell type defined in NESTML.

This example requires that PyNN be installed using the "NESTML" option, i.e.

  $ pip install PyNN[NESTML]

or you can install NESTML directly:

  $ pip install nestml


Usage: python wang_buzsaki_current_injection.py [-h] [--plot-figure] simulator

positional arguments:
  simulator      nest or spinnaker

optional arguments:
  -h, --help     show this help message and exit
  --plot-figure  Plot the simulation results to a file
  --debug        Print debugging information

"""

from pyNN.utility import get_simulator, init_logging, normalized_filename
from pyNN.random import NumpyRNG, RandomDistribution


# === Configure the simulator ================================================

sim, options = get_simulator(
    (
        "--plot-figure",
        "Plot the simulation results to a file.",
        {"action": "store_true"},
    ),
    ("--debug", "Print debugging information"),
)

if options.debug:
    init_logging(None, debug=True)

# === Register the NESTML cell type (must happen before sim.setup())

# note that the NESTML definition can also be stored in a separate file

nestml_description = """
model wb_cond_exp:
    state:
        r integer = 0 # number of steps in the current refractory phase

        V_m mV = E_L    # Membrane potential

        Inact_h real = alpha_h_init / ( alpha_h_init + beta_h_init )
        Act_n real = alpha_n_init / ( alpha_n_init + beta_n_init )

    equations:
        # synapses: exponential conductance
        kernel g_inh = exp(-t / tau_syn_inh)
        kernel g_exc = exp(-t / tau_syn_exc)

        recordable inline I_syn_exc pA = convolve(g_exc, exc_spikes) * nS * ( V_m - E_exc )
        recordable inline I_syn_inh pA = convolve(g_inh, inh_spikes) * nS * ( V_m - E_inh )

        inline I_Na  pA = g_Na * _subexpr(V_m) * Inact_h * ( V_m - E_Na )
        inline I_K   pA = g_K * Act_n**4 * ( V_m - E_K )
        inline I_L   pA = g_L * ( V_m - E_L )

        V_m' =( -( I_Na + I_K + I_L ) + I_e + I_stim + I_syn_exc - I_syn_inh ) / C_m
        Act_n' = ( alpha_n(V_m) * ( 1 - Act_n ) - beta_n(V_m) * Act_n )  # n-variable
        Inact_h' = ( alpha_h(V_m) * ( 1 - Inact_h ) - beta_h(V_m) * Inact_h ) # h-variable

    parameters:
        t_ref ms = 2 ms           # Refractory period
        g_Na nS = 3500 nS         # Sodium peak conductance
        g_K nS = 900 nS           # Potassium peak conductance
        g_L nS = 10 nS            # Leak conductance
        C_m pF = 100 pF           # Membrane capacitance
        E_Na mV = 55 mV           # Sodium reversal potential
        E_K mV = -90 mV           # Potassium reversal potential
        E_L mV = -65 mV           # Leak reversal potential (aka resting potential)
        V_Tr mV = -55 mV          # Spike threshold
        tau_syn_exc ms = 0.2 ms   # Rise time of the excitatory synaptic alpha function
        tau_syn_inh ms = 10 ms    # Rise time of the inhibitory synaptic alpha function
        E_exc mV = 0 mV           # Excitatory synaptic reversal potential
        E_inh mV = -75 mV         # Inhibitory synaptic reversal potential

        # constant external input current
        I_e pA = 0 pA

    internals:
        RefractoryCounts integer = steps(t_ref) # refractory time in steps

        alpha_n_init 1/ms = -0.05/(ms*mV) * (E_L + 34.0 mV) / (exp(-0.1 * (E_L + 34.0 mV)) - 1.0)
        beta_n_init  1/ms = 0.625/ms * exp(-(E_L + 44.0 mV) / 80.0 mV)
        alpha_h_init 1/ms = 0.35/ms * exp(-(E_L + 58.0 mV) / 20.0 mV)
        beta_h_init  1/ms = 5.0 / (exp(-0.1 / mV * (E_L + 28.0 mV)) + 1.0) /ms

    input:
        inh_spikes <- inhibitory spike
        exc_spikes <- excitatory spike
        I_stim pA <- continuous

    output:
        spike

    update:
        U_old mV = V_m
        integrate_odes()
        # sending spikes: crossing 0 mV, pseudo-refractoriness and local maximum...
        if r > 0: # is refractory?
            r -= 1
        elif V_m > V_Tr and U_old > V_m: # threshold && maximum
            r = RefractoryCounts
            emit_spike()

    function _subexpr(V_m mV) real:
        return alpha_m(V_m)**3 / ( alpha_m(V_m) + beta_m(V_m) )**3

    function alpha_m(V_m mV) 1/ms:
        return 0.1/(ms*mV) * (V_m + 35.0 mV) / (1.0 - exp(-0.1 mV * (V_m + 35.0 mV)))

    function beta_m(V_m mV) 1/ms:
        return 4.0/(ms) * exp(-(V_m + 60.0 mV) / 18.0 mV)

    function alpha_n(V_m mV) 1/ms:
        return -0.05/(ms*mV) * (V_m + 34.0 mV) / (exp(-0.1 * (V_m + 34.0 mV)) - 1.0)

    function beta_n(V_m mV) 1/ms:
        return 0.625/ms * exp(-(V_m + 44.0 mV) / 80.0 mV)

    function alpha_h(V_m mV) 1/ms:
        return 0.35/ms * exp(-(V_m + 58.0 mV) / 20.0 mV)

    function beta_h(V_m mV) 1/ms:
        return 5.0 / (exp(-0.1 / mV * (V_m + 28.0 mV)) + 1.0) /ms
"""

celltype_cls = sim.nestml.nestml_cell_type("wb_cond_exp", nestml_description)

sim.setup(timestep=0.01, min_delay=1.0)

# add some variability between neurons
rng = NumpyRNG(seed=1309463846)
rnd = lambda min, max: RandomDistribution("uniform", (min, max), rng=rng)

celltype = celltype_cls(
    g_Na=rnd(3400, 3600),  # nS
    g_K=rnd(850, 950),  # nS
    g_L=rnd(8, 12),  # nS
    C_m=rnd(80, 120),  # pF
    V_Tr=rnd(-56, -54),  # mV
    I_e=100.0,  # pA
)


# === Build and instrument the network =======================================

cells = sim.Population(5, celltype, label=celltype_cls.__name__)

cells.record(["V_m", "Act_n", "Inact_h"])


# === Run the simulation =====================================================

sim.run(100.0)


# === Save the results, optionally plot a figure =============================

filename = normalized_filename("Results", "nestml_example", "pkl", options.simulator)
cells.write_data(filename, annotations={"script_name": __file__})

if options.plot_figure:
    from pyNN.utility.plotting import Figure, Panel

    figure_filename = filename.replace("pkl", "png")
    Figure(
        Panel(
            cells.get_data().segments[0].filter(name="V_m")[0],
            ylabel="Membrane potential (mV)",
            data_labels=[cells.label],
            yticks=True,
        ),
        Panel(
            cells.get_data().segments[0].filter(name="Act_n")[0],
            ylabel="Activation variable (n)",
            data_labels=[cells.label],
            yticks=True,
            ylim=(0, 1),
        ),
        Panel(
            cells.get_data().segments[0].filter(name="Inact_h")[0],
            xticks=True,
            xlabel="Time (ms)",
            ylabel="Inactivation variable (h)",
            data_labels=[cells.label],
            yticks=True,
            ylim=(0, 1),
        ),
        title="Responses of Wang-Buzsaki neuron model, defined in NESTML, to current injection",
        annotations="Simulated with %s" % options.simulator.upper(),
    ).save(figure_filename)
    print(figure_filename)

# === Clean up and quit ========================================================

sim.end()
