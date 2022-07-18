"""
Plot graphs showing the results of running the VAbenchmarks.py script.

Usage: VAbenchmark_graphs.py [-h] [-s SORT] [-o OUTPUT_FILE] [-a ANNOTATION]
                             datafile [datafile ...]

positional arguments:
  datafile              a list of data files in a Neo-supported format

optional arguments:
  -h, --help            show this help message and exit
  -s SORT, --sort SORT  field to sort by (default='simulator')
  -o OUTPUT_FILE, --output-file OUTPUT_FILE
                        output filename
  -a ANNOTATION, --annotation ANNOTATION
                        additional annotation (optional)

"""

import argparse
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from quantities import mV
from neo.io import get_io


def plot_signal(panel, signal, index, colour='b', linewidth='1', label='', fake_aps=False,
                hide_axis_labels=False):
    label = "%s (Neuron %d)" % (label, signal.array_annotations["channel_index"][index])
    if fake_aps:  # add fake APs for plotting
        v_thresh = fake_aps
        spike_indices = signal >= v_thresh - 0.05 * mV
        signal[spike_indices] = 0.0 * mV
    panel.plot(signal.times, signal.magnitude[:, index], colour, linewidth=linewidth, label=label)
    #plt.setp(plt.gca().get_xticklabels(), visible=False)
    if not hide_axis_labels:
        panel.set_xlabel("time (%s)" % signal.times.units._dimensionality.string)
        panel.set_ylabel("%s (%s)" % (signal.name, signal.units._dimensionality.string))


def plot_hist(panel, hist, bins, width, xlabel=None, ylabel=None,
              label=None, xticks=None, xticklabels=None, xmin=None, ymax=None):
    if xlabel:
        panel.set_xlabel(xlabel)
    if ylabel:
        panel.set_ylabel(ylabel)
    for t, n in zip(bins[:-1], hist):
        panel.bar(t, n, width=width, color='b')
    if xmin:
        panel.set_xlim(xmin=xmin)
    if ymax:
        panel.set_ylim(ymax=ymax)
    if xticks is not None:
        panel.set_xticks(xticks)
    if xticklabels:
        panel.set_xticklabels(xticklabels)
    panel.text(0.8, 0.8, label, transform=panel.transAxes)


def plot_vm_traces(panel, segment, label, hide_axis_labels=False):
    for array in segment.analogsignals:
        sorted_channels = sorted(array.array_annotations["channel_index"])
        for j in range(2):
            i = array.array_annotations["channel_index"].tolist().index(j)
            print("plotting '%s' for %s" % (array.name, label))
            col = 'rbgmck'[j % 6]
            plot_signal(panel, array, i, colour=col, linewidth=1, label=label,
                        fake_aps=-50 * mV, hide_axis_labels=hide_axis_labels)
        panel.set_title(label)


def plot_spiketrains(panel, segment, label, hide_axis_labels=False):
    print("plotting spikes for %s" % label)
    channel_ids, spike_times = segment.spiketrains.multiplexed
    panel.plot(channel_ids, spike_times, '.', markersize=0.2)
    if not hide_axis_labels:
        panel.set_ylabel(segment.name)
    #plt.setp(plt.gca().get_xticklabels(), visible=False)


def plot_isi_hist(panel, segment, label, hide_axis_labels=False):
    print("plotting ISI histogram (%s)" % label)
    bin_width = 0.2
    bins_log = np.arange(0, 8, 0.2)
    bins = np.exp(bins_log)
    all_isis = np.concatenate([np.diff(np.array(st)) for st in segment.spiketrains])
    isihist, bins = np.histogram(all_isis, bins)
    xlabel = "Inter-spike interval (ms)"
    ylabel = "n in bin"
    if hide_axis_labels:
        xlabel = None
        ylabel = None
    plot_hist(panel, isihist, bins_log, bin_width, label=label,
              xlabel=xlabel, xticks=np.log([10, 100, 1000]),
              xticklabels=['10', '100', '1000'], xmin=np.log(2),
              ylabel=ylabel)


def plot_cvisi_hist(panel, segment, label, hide_axis_labels=False):
    print("plotting CV(ISI) histogram (%s)" % label)

    def cv_isi(spiketrain):
        isi = np.diff(np.array(spiketrain))
        return np.std(isi) / np.mean(isi)
    cvs = np.fromiter((cv_isi(st) for st in segment.spiketrains if st.size > 2),
                      dtype=float)
    bin_width = 0.1
    bins = np.arange(0, 2, bin_width)
    cvhist, bins = np.histogram(cvs[~np.isnan(cvs)], bins)
    xlabel = "CV(ISI)"
    ylabel = "n in bin"
    if hide_axis_labels:
        xlabel = None
        ylabel = None
    plot_hist(panel, cvhist, bins, bin_width, label=label,
              xlabel=xlabel, xticks=np.arange(0, 2, 0.5),
              ylabel=ylabel)


def sort_by_annotation(name, objects):
    sorted_objects = defaultdict(list)
    for obj in objects:
        sorted_objects[obj.annotations[name]].append(obj)
    return sorted_objects


def plot(datafiles, output_file, sort_by='simulator', annotation=None):
    blocks = [get_io(datafile).read_block() for datafile in datafiles]
    # note: Neo needs a pretty printer that is not tied to IPython
    # for block in blocks:
    #    print(block.describe())
    script_name = blocks[0].annotations['script_name']
    for block in blocks[1:]:
        assert block.annotations['script_name'] == script_name

    fig_settings = {  # pass these in a configuration file?
        'lines.linewidth': 0.5,
        'axes.linewidth': 0.5,
        'axes.labelsize': 'small',
        'legend.fontsize': 'small',
        'font.size': 8,
        'savefig.dpi': 200,
    }
    plt.rcParams.update(fig_settings)
    CM = 1 / 2.54
    plt.figure(1, figsize=(15 * CM * len(blocks), 20 * CM))
    gs = gridspec.GridSpec(4, 2 * len(blocks), hspace=0.25, wspace=0.25)

    sorted_blocks = sort_by_annotation(sort_by, blocks)
    hide_axis_labels = False
    for k, (label, block_list) in enumerate(sorted_blocks.items()):
        segments = {}
        for block in block_list:
            for name in ("exc", "inh"):
                if name in block.name.lower():
                    segments[name] = block.segments[0]

        # Plot membrane potential traces
        plot_vm_traces(plt.subplot(gs[0, 2 * k:2 * k + 2]),
                       segments['exc'], label, hide_axis_labels)

        # Plot spike rasters
        plot_spiketrains(plt.subplot(gs[1, 2 * k:2 * k + 2]),
                         segments['exc'], label, hide_axis_labels)

        # Inter-spike-interval histograms
        # Histograms of coefficients of variation of ISI
        plot_isi_hist(plt.subplot(gs[2, 2 * k]), segments['exc'], 'exc', hide_axis_labels)
        plot_cvisi_hist(plt.subplot(gs[3, 2 * k]), segments['exc'], 'exc', hide_axis_labels)
        hide_axis_labels = True
        plot_isi_hist(plt.subplot(gs[2, 2 * k + 1]), segments['inh'], 'inh', hide_axis_labels)
        plot_cvisi_hist(plt.subplot(gs[3, 2 * k + 1]), segments['inh'], 'inh', hide_axis_labels)

    plt.savefig(output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datafiles", metavar="datafile", nargs="+",
                        help="a list of data files in a Neo-supported format")
    parser.add_argument("-s", "--sort", default="simulator",
                        help="field to sort by (default='%(default)s' - also try 'mpi_processes')")
    parser.add_argument("-o", "--output-file", default="output.png",
                        help="output filename")
    parser.add_argument("-a", "--annotation", help="additional annotation (optional)")
    args = parser.parse_args()
    plot(args.datafiles, output_file=args.output_file,
         sort_by=args.sort, annotation=args.annotation)


if __name__ == "__main__":
    main()
