"""
Simple tools for plotting Neo-format data.

These tools are intended for quickly producing basic plots with simple
formatting. If you need to produce more complex and/or publication-quality
figures, it will probably be easier to use matplotlib or another plotting
package directly rather than trying to extend this module.

:copyright: Copyright 2006-2023 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

import sys
from collections import defaultdict
from itertools import repeat
from os import path, makedirs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from quantities import ms
from neo import AnalogSignal, IrregularlySampledSignal, SpikeTrain
from neo.core.spiketrainlist import SpikeTrainList


DEFAULT_FIG_SETTINGS = {
    'lines.linewidth': 0.5,
    'axes.linewidth': 0.5,
    'axes.labelsize': 'small',
    'legend.fontsize': 'small',
    'font.size': 8,
    'savefig.dpi': 150,
}


def handle_options(ax, options):
    if "xticks" not in options or options.pop("xticks") is False:
        plt.setp(ax.get_xticklabels(), visible=False)
    if "xlabel" in options:
        ax.set_xlabel(options.pop("xlabel"))
    if "yticks" not in options or options.pop("yticks") is False:
        plt.setp(ax.get_yticklabels(), visible=False)
    if "ylabel" in options:
        ax.set_ylabel(options.pop("ylabel"))
    if "ylim" in options:
        ax.set_ylim(options.pop("ylim"))
    if "xlim" in options:
        ax.set_xlim(options.pop("xlim"))


def plot_signal(ax, signal, index=None, label='', **options):
    """
    Plot a single channel from an AnalogSignal.
    """
    if "ylabel" in options:
        if options["ylabel"] == "auto":
            options["ylabel"] = "%s (%s)" % (signal.name,
                                             signal.units._dimensionality.string)
    handle_options(ax, options)
    if index is None:
        label = "%s (Neuron %d)" % (label, signal.array_annotations["channel_index"] or 0)
    else:
        label = "%s (Neuron %d)" % (label, signal.array_annotations["channel_index"][index])
        signal = signal[:, index]
    ax.plot(signal.times.rescale(ms), signal.magnitude, label=label, **options)
    ax.legend()


def plot_signals(ax, signal_array, label_prefix='', **options):
    """
    Plot all channels in an AnalogSignal in a single panel.
    """
    if "ylabel" in options:
        if options["ylabel"] == "auto":
            options["ylabel"] = "%s (%s)" % (signal_array.name,
                                             signal_array.units._dimensionality.string)
    handle_options(ax, options)
    offset = options.pop("y_offset", None)
    show_legend = options.pop("legend", True)
    if "channel_index" in signal_array.array_annotations:
        channel_iterator = signal_array.array_annotations["channel_index"].argsort()
    else:
        channel_iterator = range(signal_array.shape[1])
    for i in channel_iterator:
        if "channel_index" in signal_array.array_annotations:
            channel = signal_array.array_annotations["channel_index"][i]
            if label_prefix:
                label = "%s (Neuron %d)" % (label_prefix, channel)
            else:
                label = "Neuron %d" % channel
        elif label_prefix:
            label = "%s (%d)" % (label_prefix, i)
        else:
            label = str(i)
        signal = signal_array[:, i]
        if offset:
            signal += i * offset
        ax.plot(signal.times.rescale(ms), signal.magnitude, label=label, **options)
    if show_legend:
        ax.legend()


def plot_spiketrains(ax, spiketrains, label='', **options):
    """
    Plot all spike trains in a Segment in a raster plot.
    """
    ax.set_xlim(spiketrains[0].t_start, spiketrains[0].t_stop)
    handle_options(ax, options)
    max_index = 0
    min_index = sys.maxsize
    for spiketrain in spiketrains:
        ax.plot(spiketrain,
                np.ones_like(spiketrain) * spiketrain.annotations['source_index'],
                'k.', **options)
        max_index = max(max_index, spiketrain.annotations['source_index'])
        min_index = min(min_index, spiketrain.annotations['source_index'])
    ax.set_ylabel("Neuron index")
    ax.set_ylim(-0.5 + min_index, max_index + 0.5)
    if label:
        plt.text(0.95, 0.95, label,
                 transform=ax.transAxes, ha='right', va='top',
                 bbox=dict(facecolor='white', alpha=1.0))


def plot_spiketrainlist(ax, spiketrains, label='', **options):
    """
    Plot all spike trains in a Segment in a raster plot.
    """
    ax.set_xlim(spiketrains.t_start, spiketrains.t_stop)
    handle_options(ax, options)
    channel_ids, spike_times = spiketrains.multiplexed
    max_id = max(spiketrains.all_channel_ids)
    min_id = min(spiketrains.all_channel_ids)
    ax.plot(spike_times, channel_ids, 'k.', **options)
    ax.set_ylabel("Neuron index")
    ax.set_ylim(-0.5 + min_id, max_id + 0.5)
    if label:
        plt.text(0.95, 0.95, label,
                 transform=ax.transAxes, ha='right', va='top',
                 bbox=dict(facecolor='white', alpha=1.0))


def plot_array_as_image(ax, arr, label='', **options):
    """
    Plots a numpy array as an image.
    """
    handle_options(ax, options)
    show_legend = options.pop("legend", True)
    plt.pcolormesh(arr, **options)
    ax.set_aspect('equal')
    if label:
        plt.text(0.95, 0.95, label,
                 transform=ax.transAxes, ha='right', va='top',
                 bbox=dict(facecolor='white', alpha=1.0))
    if show_legend:
        plt.colorbar()


def scatterplot(ax, data_table, label='', **options):
    handle_options(ax, options)
    if options.pop("show_fit", False):
        plt.plot(data_table.x, data_table.y_fit, 'k-')
    plt.scatter(data_table.x, data_table.y, **options)
    if label:
        plt.text(0.95, 0.95, label,
                 transform=ax.transAxes, ha='right', va='top',
                 bbox=dict(facecolor='white', alpha=1.0))


def plot_hist(ax, histogram, label='', **options):
    handle_options(ax, options)
    for t, n in histogram:
        ax.bar(t, n, width=histogram.bin_width, color=None)
    if label:
        plt.text(0.95, 0.95, label,
                 transform=ax.transAxes, ha='right', va='top',
                 bbox=dict(facecolor='white', alpha=1.0))


def variable_names(segment):
    """
    List the names of all the AnalogSignals (used for the variable name by
    PyNN) in the given segment.
    """
    return set(signal.name for signal in segment.analogsignals)


class Figure(object):
    """
    Provide simple, declarative specification of multi-panel figures.

    Example::

      Figure(
          Panel(segment.filter(name="v")[0], ylabel="Membrane potential (mV)")
          Panel(segment.spiketrains, xlabel="Time (ms)"),
          title="Network activity",
      ).save("figure3.png")

    Valid options are:
        `settings`:
            for figure settings, e.g. {'font.size': 9}
        `annotations`:
            a (multi-line) string to be printed at the bottom of the figure.
        `title`:
            a string to be printed at the top of the figure.
    """

    def __init__(self, *panels, **options):
        n_panels = len(panels)
        if "settings" in options and options["settings"] is not None:
            settings = options["settings"]
        else:
            settings = DEFAULT_FIG_SETTINGS
        plt.rcParams.update(settings)
        width, height = options.get("size", (6, 2 * n_panels + 1.2))
        self.fig = plt.figure(1, figsize=(width, height))
        gs = gridspec.GridSpec(n_panels, 1)
        if "annotations" in options:
            gs.update(bottom=1.2 / height)  # leave space for annotations
        gs.update(top=1 - 0.8 / height, hspace=0.25)
        # print(gs.get_grid_positions(self.fig))

        for i, panel in enumerate(panels):
            panel.plot(plt.subplot(gs[i, 0]))

        if "title" in options:
            self.fig.text(0.5, 1 - 0.5 / height, options["title"],
                          ha="center", va="top", fontsize="large")
        if "annotations" in options:
            plt.figtext(0.01, 0.01, options["annotations"], fontsize=6, verticalalignment='bottom')

    def save(self, filename):
        """
        Save the figure to file. The format is taken from the file extension.
        """
        dirname = path.dirname(filename)
        if dirname and not path.exists(dirname):
            makedirs(dirname)
        self.fig.savefig(filename)

    def show(self):
        plt.show()


class Panel(object):
    """
    Represents a single panel in a multi-panel figure.

    A panel is a Matplotlib Axes or Subplot instance. A data item may be an
    AnalogSignal, AnalogSignal, or a list of SpikeTrains. The Panel will
    automatically choose an appropriate representation. Multiple data items may
    be plotted in the same panel.

    Valid options are any valid Matplotlib formatting options that should be
    applied to the Axes/Subplot, plus in addition:

        `data_labels`:
            a list of strings of the same length as the number of data items.
        `line_properties`:
            a list of dicts containing Matplotlib formatting options, of the
            same length as the number of data items.

    """

    def __init__(self, *data, **options):
        self.data = list(data)
        self.options = options
        self.data_labels = options.pop("data_labels", repeat(None))
        self.line_properties = options.pop("line_properties", repeat({}))

    def plot(self, axes):
        """
        Plot the Panel's data in the provided Axes/Subplot instance.
        """
        for datum, label, properties in zip(self.data, self.data_labels, self.line_properties):
            properties.update(self.options)
            if isinstance(datum, DataTable):
                scatterplot(axes, datum, label=label, **properties)
            elif isinstance(datum, Histogram):
                plot_hist(axes, datum, label=label, **properties)
            elif isinstance(datum, (AnalogSignal, IrregularlySampledSignal)):
                plot_signals(axes, datum, label_prefix=label, **properties)
            elif isinstance(datum, list) and len(datum) > 0 and isinstance(datum[0], SpikeTrain):
                plot_spiketrains(axes, datum, label=label, **properties)
            elif isinstance(datum, SpikeTrainList):
                plot_spiketrainlist(axes, datum, label=label, **properties)
            elif isinstance(datum, np.ndarray):
                if datum.ndim == 2:
                    plot_array_as_image(axes, datum, label=label, **properties)
                else:
                    raise Exception("Can't handle arrays with %s dimensions" % datum.ndim)
            else:
                raise Exception("Can't handle type %s" % type(datum))


def comparison_plot(segments, labels, title='', annotations=None,
                    fig_settings=None, with_spikes=True):
    """
    Given a list of segments, plot all the data they contain so as to be able
    to compare them.

    Return a Figure instance.
    """
    variables_to_plot = set.union(*(variable_names(s) for s in segments))
    print("Plotting the following variables: %s" % ", ".join(variables_to_plot))

    # group signal arrays by name
    n_seg = len(segments)
    by_var_and_channel = defaultdict(lambda: defaultdict(list))
    line_properties = []
    units = {}
    for k, (segment, label) in enumerate(zip(segments, labels)):
        lw = 2 * (n_seg - k) - 1
        col = 'bcgmkr'[k % 6]
        line_properties.append({"linewidth": lw, "color": col})
        for array in segment.analogsignals:
            # rescale signals to the same units, for a given variable name
            if array.name not in units:
                units[array.name] = array.units
            elif array.units != units[array.name]:
                array = array.rescale(units[array.name])
            for i in array.array_annotations["channel_index"].argsort():
                channel = array.array_annotations["channel_index"][i]
                signal = array[:, i]
                by_var_and_channel[array.name][channel].append(signal)
    # each panel plots the signals for a given variable.
    panels = []
    for by_channel in by_var_and_channel.values():
        for array_list in by_channel.values():
            ylabel = array_list[0].name
            if ylabel:
                ylabel += " ({})".format(array_list[0].dimensionality)
            panels.append(
                Panel(*array_list,
                      line_properties=line_properties,
                      yticks=True,
                      ylabel=ylabel,
                      data_labels=labels))
    if with_spikes and len(segments[0].spiketrains) > 0:
        panels += [Panel(segment.spiketrains, data_labels=[label])
                   for segment, label in zip(segments, labels)]
    panels[-1].options["xticks"] = True
    panels[-1].options["xlabel"] = "Time (ms)"
    fig = Figure(*panels,
                 title=title,
                 settings=fig_settings,
                 annotations=annotations)
    return fig


class DataTable(object):
    """A lightweight encapsulation of x, y data for scatterplots."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def fit_curve(self, f, p0, **fitting_parameters):
        from scipy.optimize import curve_fit
        self._f = f
        self._p0 = p0
        self._popt, self._pcov = curve_fit(f, self.x, self.y, p0, **fitting_parameters)
        return self._popt, self._pcov

    @property
    def y_fit(self):
        return self._f(self.x, *self._popt)


class Histogram(object):
    """A lightweight encapsulation of histogram data."""

    def __init__(self, data):
        self.data = data
        self.evaluated = False

    def evaluate(self):
        if not self.evaluated:
            n_bins = int(np.sqrt(len(self.data)))
            self.values, self.bins = np.histogram(self.data, bins=n_bins)
            self.bin_width = self.bins[1] - self.bins[0]
            self.evaluated = True

    def __iter__(self):
        """Iterate over the bars of the histogram"""
        self.evaluate()
        for x, y in zip(self.bins[:-1], self.values):
            yield (x, y)


def isi_histogram(segment):
    all_isis = np.concatenate([np.diff(np.array(st)) for st in segment.spiketrains])
    return Histogram(all_isis)


def connection_plot(projection, positive="O", zero=".", empty=" ", spacer=""):
    """Produce a simple text-based representation of a connectivity matrix"""
    connection_array = projection.get("weight", format="array")
    image = np.zeros_like(connection_array, dtype=str)
    # ignore the complaint that x > 0 is invalid for NaN
    old_settings = np.seterr(invalid="ignore")
    image[connection_array > 0] = positive
    image[connection_array == 0] = zero
    np.seterr(**old_settings)  # restore original floating point error settings
    image[np.isnan(connection_array)] = empty
    return "\n".join([spacer.join(row) for row in image])
