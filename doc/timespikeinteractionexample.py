from pyNN.nest import *
from matplotlib.pyplot import *
def test_sim(on_or_off_grid, sim_time):
    setup(timestep=1.0, min_delay=1.0, max_delay=1.0, spike_precision=on_or_off_grid)
    src = Population(1, SpikeSourceArray, cellparams={'spike_times': [0.5]})
    cm        = 250.0
    tau_m     =  10.0
    tau_syn_E =   1.0
    weight    = cm/tau_m * numpy.power(tau_syn_E/tau_m, -tau_m/(tau_m-tau_syn_E)) * 20.5
    nrn = Population(1, IF_curr_exp, cellparams={'cm':         cm,
                                                 'tau_m':      tau_m,
                                                 'tau_syn_E':  tau_syn_E,
                                                 'tau_refrac':  2.0,
                                                 'v_thresh':   20.0,
                                                 'v_rest':      0.0,
                                                 'v_reset':     0.0,
                                                 'i_offset':    0.0})
    nrn.initialize('v', 0.0)
    prj = Projection(src, nrn, OneToOneConnector(weights=weight))
    nrn.record_v()
    run(sim_time)
    Vm = nrn.get_v()
    end()
    return numpy.transpose(Vm)[1:3]
sim_time = 10.0
off = test_sim('off_grid', sim_time)
on  = test_sim('on_grid', sim_time)
subplot(1,2,1)
plot(off[0], off[1],color='0.7',linewidth=7, label='off-grid')
plot(on[0], on[1],'k', label='on-grid')
ylim(-0.5,21)
xlim(0,9)
xlabel('time [ms]')
ylabel('V [mV]')
legend()
subplot(1,2,2)
plot(off[0],off[1],color='0.7',linewidth=7)
plot(on[0],on[1],'k')
ylim(-0.05,2.1)
xlim(0,9)
xlabel('time [ms]')
ylabel('V [mV]')
show()

