"""
Helper function for drawing current-source plots
"""

import matplotlib.pyplot as plt


def plot_current_source(t, i_inj, v, i_range=None, v_range=None,
                        i_ticks=None, v_ticks=None, t_range=None):
    """
    Plot voltage and current traces
    """
    fig = plt.figure(figsize=(8, 3))
    fig.dpi = 120

    ax = fig.add_axes((0.1, 0.4, 0.85, 0.5), frameon=False)
    ax.plot(t, v, 'b')
    if v_range:
        ax.set_ylim(*v_range)
    ax.set_ylabel('V (mV)')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    if v_ticks:
        ax.yaxis.set_ticks(v_ticks)
    if t_range:
        ax.set_xlim(*t_range)

    # add the left axis line back in
    xmin, xmax = ax.get_xaxis().get_view_interval()
    ymin, ymax = ax.get_yaxis().get_view_interval()
    ax.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=3))

    ax = fig.add_axes((0.1, 0.14, 0.85, 0.25), frameon=False)
    ax.plot(t, i_inj, 'g')
    if i_range:
        ax.set_ylim(*i_range)
    ax.set_ylabel('I (nA)')
    ax.set_xlabel('Time (ms)')
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()
    if i_ticks:
        ax.yaxis.set_ticks(i_ticks)
    if t_range:
        ax.set_xlim(*t_range)

    # add the bottom and left axis lines back in
    xmin, xmax = ax.get_xaxis().get_view_interval()
    ymin, ymax = ax.get_yaxis().get_view_interval()
    ax.add_artist(plt.Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=3))
    ax.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=3))

    plt.savefig("tmp.png")
    return fig
