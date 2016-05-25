"""
Helper function for drawing current-source plots
"""


def plot_current_source(t, i_inj, v):
    """
    Plot voltage and current traces
    """
    fig = plt.figure(figsize=(8, 3))
    fig.dpi = 120

    ax = fig.add_axes((0.1, 0.4, 0.85, 0.5), frameon=False)
    ax.plot(t, v, 'b')
    ax.set_ylim(-65.5, -59.5)
    ax.set_ylabel('V (mV)')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_ticks((-65, -64, -63, -62, -61, -60))

    # add the left axis line back in
    xmin, xmax = ax.get_xaxis().get_view_interval()
    ymin, ymax = ax.get_yaxis().get_view_interval()
    ax.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=3))

    ax = fig.add_axes((0.1, 0.14, 0.85, 0.25), frameon=False)
    ax.plot(t, i_inj, 'g')
    ax.set_ylim(-0.1, 0.55)
    ax.set_ylabel('I (nA)')
    ax.set_xlabel('Time (ms)')
    ax.yaxis.tick_left()
    ax.xaxis.tick_bottom()
    ax.yaxis.set_ticks((0.0, 0.2, 0.4))

    # add the bottom and left axis lines back in
    xmin, xmax = ax.get_xaxis().get_view_interval()
    ymin, ymax = ax.get_yaxis().get_view_interval()
    ax.add_artist(plt.Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=3))
    ax.add_artist(plt.Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=3))

    fig.show()
    return fig
