import matplotlib.pyplot as plt
import numpy as np

def se2_align_plot(t, fa, gt=None, est=None, figax=None, line_kwargs={}, marker_kwargs={}, fa_obj=None, marker_objective_coloring=False, legacy=False):
    """
    Function for creating SE(2) frame alignment plots

    Args:
        t (np.array, shape=(n,)): time array
        fa (np.array, shape=(n,m,3)): frame alignments with m alignments per time and alignments given in x, y, theta order
        gt (np.array, shape=(n,3), optional): ground truth alignment. Defaults to None.
        est (np.array, shape=(n,3), optional): estimate frame alignment. Defaults to None.
        figax (tuple(matplotlib fig, matplotlib ax), optional): figure and axis. Defaults to None.
        line_kwargs (dict, optional): kwargs for gt and est lines. Defaults to {}.
        fa_obj (np.array, shape=(n,m), optional): frame alignment objective values. Defaults to None.
        marker_objective_coloring (bool, optional): whether to color markers by objective value. Defaults to False.

    Returns:
        matplotlib Figure: figure
        matplotlib Axis: axis
    """
    assert not marker_objective_coloring or fa_obj is not None, "fa_obj must be provided if marker_objective_coloring is True"

    fig, ax = plt.subplots(3, 1, figsize=(12,10)) if figax is None else figax
    n = len(t)
    m = fa.shape[1]
    assert fa.shape == (n,m,3), f"fa must have shape (n,m,3), got {fa.shape}"
    assert gt is None or gt.shape == (n,3), f"gt must have shape (n,3), got {gt.shape if gt is not None else None}"
    if legacy:
        assert est is None or est.shape == (n,6), f"est must have shape (n,6), got {est.shape if est is not None else None}"
    else:
        assert est is None or est.shape == (n,3), f"est must have shape (n,3), got {est.shape if est is not None else None}"
        # insert blank columns for legacy compatibility
        est = np.insert(est, [1,2,3], 0, axis=1)
    assert fa_obj is None or fa_obj.shape == (n,m), f"fa_obj must have shape (n,m), got {fa_obj.shape if fa_obj is not None else None}"
    
    fa = np.array(fa)
    if est is not None:
        est = np.array(est)
    if gt is not None:
        gt = np.array(gt)
    # z_sel = np.array(z_sel)
    for i in range(3):
        mult = 180/np.pi if i == 2 else 1
        if marker_objective_coloring:
            colored_kwargs = marker_kwargs.copy()
            for c, j in zip(['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple'], range(7)):
                colored_kwargs['color'] = c
                for k in range(n):
                    obj_max = np.nanmax(fa_obj[k,:])
                    # if not np.isnan(obj_max):
                    #     import ipdb; ipdb.set_trace()
                    filtered_fa = fa[k,np.bitwise_and(fa_obj[k] <= obj_max*(1-j/7), fa_obj[k] > obj_max*(1-(j+1)/7)),i]
                    if len(filtered_fa) == 0:
                        continue
                    # print(filtered_fa)
                    # print(t[k])
                    # import ipdb; ipdb.set_trace()
                    ax[i].plot(np.ones(filtered_fa.shape)*t[k], filtered_fa*mult, 'o', label='_nolegend_', **colored_kwargs)
        else:
            ax[i].plot(t, fa[:,:,i]*mult, 'o', color='navy', label='_nolegend_', **marker_kwargs)
        if gt is not None:
            ax[i].plot(t, gt[:,i]*mult, color='lime', label='_nolegend_', **line_kwargs)
        if est is not None:
            ax[i].plot(t, est[:,2*i]*mult, color='red', label='_nolegend_', **line_kwargs)
        # ax[i].plot(t, z_sel[:,i], color='red')
    [ax_i.grid(True) for ax_i in ax]
    [ax_i.set_ylabel(lb) for ax_i, lb in zip(ax, ['x (m)', 'y (m)', r'$\theta$ (deg)'])]
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