import matplotlib.pyplot as plt
import numpy as np

def se2_align_plot(t, fa, gt=None, est=None, figax=None, line_kwargs={}, marker_kwargs={}):
    """
    Function for creating SE(2) frame alignment plots

    Args:
        t (np.array, shape=(n,)): time array
        fa (np.array, shape=(n,m,3)): frame alignments with m alignments per time and alignments given in x, y, theta order
        gt (np.array, shape=(n,3), optional): ground truth alignment. Defaults to None.
        est (np.array, shape=(n,3), optional): estimate frame alignment. Defaults to None.
        figax (tuple(matplotlib fig, matplotlib ax), optional): figure and axis. Defaults to None.
        line_kwargs (dict, optional): kwargs for gt and est lines. Defaults to {}.

    Returns:
        matplotlib Figure: figure
        matplotlib Axis: axis
    """
    fig, ax = plt.subplots(3, 1, figsize=(12,10)) if figax is None else figax
    
    fa = np.array(fa)
    if est is not None:
        est = np.array(est)
    if gt is not None:
        gt = np.array(gt)
    # z_sel = np.array(z_sel)
    for i in range(3):
        mult = 180/np.pi if i == 2 else 1
        ax[i].plot(t, fa[:,:,i]*mult, 'o', color='navy', label='_nolegend_', **marker_kwargs)
        if gt is not None:
            ax[i].plot(t, gt[:,i]*mult, color='lime', label='_nolegend_', **line_kwargs)
        if est is not None:
            ax[i].plot(t, est[:,2*i]*mult, color='red', label='_nolegend_', **line_kwargs)
        # ax[i].plot(t, z_sel[:,i], color='red')
    [ax_i.grid(True) for ax_i in ax]
    [ax_i.set_ylabel(lb) for ax_i, lb in zip(ax, ['x (m)', 'y (m)', 'theta (deg)'])]
    ax[2].set_xlabel("time (s)")

    for i in range(3):
        xlim = ax[i].get_xlim()
        ax[i].set_xlim([min(xlim[0], t[0]), max(xlim[1], t[-1])])
    
    xlim = ax[0].get_xlim()
    ylim = ax[0].get_ylim()
    ax[0].plot(1e4, 1e4, 'o', color='navy', label='measurement')
    if gt is not None:
        ax[0].plot([1e4, 1e4], [1e4, 1e4], color='lime', label='ground truth', **line_kwargs)
    if est is not None:
        ax[0].plot([1e4, 1e4], [1e4, 1e4], color='red', label='estimate', **line_kwargs)
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylim)
    ax[0].legend()
    return fig, ax