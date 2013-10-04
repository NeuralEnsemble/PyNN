"""
Simple tools for plotting Neo-format data.

These tools are intended for quickly producing basic plots with simple
formatting. If you need to produce more complex and/or publication-quality
figures, it will probably be easier to use matplotlib or another plotting
package directly rather than trying to extend this module.

:copyright: Copyright 2006-2013 by the PyNN team, see AUTHORS.
:license: CeCILL, see LICENSE for details.

"""

from collections import defaultdict
from numbers import Number
from itertools import repeat
from os import path, makedirs
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from quantities import ms
from neo import AnalogSignalArray, AnalogSignal, SpikeTrain


DEFAULT_FIG_SETTINGS = {
    'lines.linewidth': 0.5,
    'axes.linewidth': 0.5,
    'axes.labelsize': 'small',
    'legend.fontsize': 'small',
    'font.size': 8,
}


def handle_options(ax, options):
    if "xticks" not in options or options.pop("xticks") is False:
        plt.setp(ax.get_xticklabels(), visible=False)
    if "xlabel" in options:
        ax.set_xlabel(options.pop("xlabel"))


def plot_signal(ax, signal, index=None, label='', **options):
    """
    Plot an AnalogSignal or one signal from an AnalogSignalArray.
    """
    handle_options(ax, options)
    if index is None:
        label = "%s (Neuron %d)" % (label, signal.channel_index)
    else:
        label = "%s (Neuron %d)" % (label, signal.channel_index[index])
        signal = signal[:, index]
    ax.plot(signal.times.rescale(ms), signal, label=label, **options)
    ax.set_ylabel("%s (%s)" % (signal.name, signal.units._dimensionality.string))
    plt.legend()


def plot_signals(ax, signal_array, label_prefix='', **options):
    """
    Plot all signals in an AnalogSignalArray in a single panel.
    """
    handle_options(ax, options)
    ax.set_ylabel(
        options.pop("ylabel",
                    "%s (%s)" % (signal_array.name,
                                 signal_array.units._dimensionality.string)))
    for i in signal_array.channel_index.argsort():
        channel = signal_array.channel_index[i]
        signal = signal_array[:, i]
        label = "%s (Neuron %d)" % (label_prefix, channel)
        ax.plot(signal.times.rescale(ms), signal, label=label, **options)
    plt.legend()


def plot_spiketrains(ax, spiketrains, label='', **options):
    """
    Plot all spike trains in a Segment in a raster plot.
    """
    handle_options(ax, options)
    max_index = 0
    for spiketrain in spiketrains:
        plt.plot(spiketrain,
                 np.ones_like(spiketrain) * spiketrain.annotations['source_index'],
                 'k.')
        max_index = max(max_index, spiketrain.annotations['source_index'])
    ax.set_ylabel("Neuron index")
    plt.xlim(0, spiketrain.t_stop/ms)
    plt.ylim(-0.5, max_index - 0.5)
    if label:
        plt.text(0.95, 0.95, label,
                 transform=ax.transAxes, ha='right', va='top',
                 bbox=dict(facecolor='white', alpha=1.0))
    

def variable_names(segment):
    """
    List the names of all the AnalogSignalArrays (used for the variable name by
    PyNN) in the given segment.
    """
    return set(signal.name for signal in segment.analogsignalarrays)


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
    
        `settings` - for figure settings, e.g. {'font.size': 9}
        `annotations` - a (multi-line) string to be printed at the bottom of the figure.
        `title` - a string to be printed at the top of the figure.
    """
    
    def __init__(self, *panels, **options):
        n_panels = len(panels)
        if "settings" in options and options["settings"] is not None:
            settings = options["settings"]
        else:
            settings = DEFAULT_FIG_SETTINGS
        plt.rcParams.update(settings)
        width, height = options.get("size", (6, 2*n_panels + 1.2))
        self.fig = plt.figure(1, figsize=(width, height))
        gs = gridspec.GridSpec(n_panels, 1)
        if "annotations" in options:
            gs.update(bottom=1.2/height)  # leave space for annotations
        gs.update(top=1 - 0.8/height, hspace=0.1) 
        print gs.get_grid_positions(self.fig)
        
        for i, panel in enumerate(panels):
            panel.plot(plt.subplot(gs[i, 0]))
        
        if "title" in options:
            self.fig.text(0.5, 1 - 0.5/height, options["title"],
                          ha="center", va="top", fontsize="large")
        if "annotations" in options:
            plt.figtext(0.01, 0.01, options["annotations"], fontsize=6, verticalalignment='bottom')

    def save(self, filename):
        """
        Save the figure to file. The format is taken from the file extension.
        """
        dirname = path.dirname(filename)
        if not path.exists(dirname):
            makedirs(dirname)
        self.fig.savefig(filename)


class Panel(object):
    """
    Represents a single panel in a multi-panel figure.
    
    A panel is a Matplotlib Axes or Subplot instance. A data item may be an
    AnalogSignal, AnalogSignalArray, or a list of SpikeTrains. The Panel will
    automatically choose an appropriate representation. Multiple data items may
    be plotted in the same panel.
    
    Valid options any valid Matplotlib formatting options that should be applied
    to the Axes/Subplot, plus in addition:
    
        `data_labels`: a list of strings of the same length as the number of
                       data items.
        `line_properties`: a list of dicts containing Matplotlib formatting
                           options, of the same length as the number of data
                           items.
        
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
            if isinstance(datum, AnalogSignal):
                plot_signal(axes, datum, label=label, **properties)
            elif isinstance(datum, AnalogSignalArray):
                plot_signals(axes, datum, label_prefix=label, **properties)
            elif isinstance(datum, list) and len(datum) > 0 and isinstance(datum[0], SpikeTrain):
                plot_spiketrains(axes, datum, label=label, **properties)
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
    print "Plotting the following variables: %s" % ", ".join(variables_to_plot)

    # group signal arrays by name        
    n_seg = len(segments)
    by_var_and_channel = defaultdict(lambda: defaultdict(list))
    line_properties = []
    for k, (segment, label) in enumerate(zip(segments, labels)):
        lw = 2*(n_seg - k) - 1
        col = 'rbgmck'[k%6]
        line_properties.append({"linewidth": lw, "color": col})
        for array in segment.analogsignalarrays:
            for i in array.channel_index.argsort():
                channel = array.channel_index[i]
                signal = array[:, i]
                by_var_and_channel[array.name][channel].append(signal)
    # each panel plots the signals for a given variable.
    panels = []
    for by_channel in by_var_and_channel.values():    
        panels += [Panel(*array_list,
                         line_properties=line_properties,
                         data_labels=labels) for array_list in by_channel.values()]
    if with_spikes:
        panels += [Panel(segment.spiketrains, data_labels=[label])
                   for segment, label in zip(segments, labels)]
    panels[-1].options["xticks"] = True
    panels[-1].options["xlabel"] = "Time (ms)"
    fig = Figure(*panels,
                 title=title,
                 settings=fig_settings,
                 annotations=annotations)
    return fig
