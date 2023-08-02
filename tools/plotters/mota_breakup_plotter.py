import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/masonbp/ford/motlee/tools')
from metric_plotter import parse_metric_file
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, default='/home/masonbp/ford/data/mot_static/static-2022/iros/mota_brokenup_data.yaml')
parser.add_argument('-o', '--output', type=str, default=None)
args = parser.parse_args()
metric_file = args.input
mh = parse_metric_file(metric_file, ['mota', 'fp', 'fn', 'switch'])


plt.style.use('/home/masonbp/computer/python/matplotlib/publication.mplstyle')

# fig, ax = plt.subplots(2, 1, sharex=True)
# x, y = mh.get_metric_along_line('mota', lambda x: x[1] == 0.0, 0, ret_type='avg')
# # x = x[:x.index(1.5)]
# # y = y[:len(x)]
# # for i in range(len(x)):
# #     x[i] *= np.sqrt(2/3)
# ax[1].plot(x,y,'--o')
# x, y = mh.get_metric_along_line('fp', lambda x: x[1] == 0.0, 0, ret_type='avg')
# # x = x[:x.index(1.5)]
# # y = y[:len(x)]
# # for i in range(len(x)):
# #     x[i] *= np.sqrt(2/3)
# ax[0].plot(x,y,'--o')
# x, y = mh.get_metric_along_line('fn', lambda x: x[1] == 0.0, 0, ret_type='avg')
# # x = x[:x.index(1.5)]
# # y = y[:len(x)]
# # for i in range(len(x)):
# #     x[i] *= np.sqrt(2/3)
# ax[0].plot(x,y,'--o')
# x, y = mh.get_metric_along_line('switch', lambda x: x[1] == 0.0, 0, ret_type='avg')
# # x = x[:x.index(1.5)]
# # y = y[:len(x)]
# # for i in range(len(x)):
# #     x[i] *= np.sqrt(2/3)
# ax[0].plot(x,y,'--o')


if False:
    fig, ax = plt.subplots(2, 1, sharex=True)
    # x, y = mh.get_metric_along_line('mota', lambda x: x[1] == 0.0, 0, ret_type='avg')
    # ax[1].plot(x,y,'--o')
    # x, y = mh.get_metric_along_line('fp', lambda x: x[1] == 0.0, 0, ret_type='avg')
    # ax[0].plot(x,y,'--o')
    # x, y = mh.get_metric_along_line('fn', lambda x: x[1] == 0.0, 0, ret_type='avg')
    # ax[0].plot(x,y,'--o')
    # x, y = mh.get_metric_along_line('switch', lambda x: x[1] == 0.0, 0, ret_type='avg')
    # ax[0].plot(x,y,'--o')

    x, y = mh.get_metric_along_line('mota', lambda x: abs(x[1] - x[0] * 8.1712) < .01, 0, ret_type='avg')
    ax[1].plot(x,y,'--o')
    x, y = mh.get_metric_along_line('fp', lambda x: abs(x[1] - x[0] * 8.1712) < .01, 0, ret_type='avg')
    ax[0].plot(x,y,'--o')
    x, y = mh.get_metric_along_line('fn', lambda x: abs(x[1] - x[0] * 8.1712) < .01, 0, ret_type='avg')
    ax[0].plot(x,y,'--o')
    x, y = mh.get_metric_along_line('switch', lambda x: abs(x[1] - x[0] * 8.1712) < .01, 0, ret_type='avg')
    ax[0].plot(x,y,'--o')


    # for i in range(len(X_sorted[1:])):
    #     ax[i].hist(Y_sorted[i+1])
    #     ax[i].set_title(f'standard deviation \ntranslation \nerror = {X_sorted[i]}')
    # ax[0].set_xlabel('Number of inconsistencies')
    # ax[0].set_ylabel('Frequency')
    ax[0].set_ylabel('Average per frame')
    ax[1].set_ylabel('Average MOTA')
    ax[1].set_xlabel('Translation Error Standard Dev (m)')
    ax[0].legend(['false positives', 'misses', 'mismatches'])
    ax[0].grid(True)
    ax[1].grid(True)

    plt.show()
else:
    # plt.tight_layout()
    width = 3.487
    height = width / 1.618 * .7

    fig, ax1 = plt.subplots(figsize=(width, height), dpi=200)
    # ax1.set_in_layout(True)

    color = 'tab:blue'
    ax1.set_ylabel('Average Number Per Frame')
    ax1.set_xlabel('Translation Error Standard Deviation (m)', color=color)
    ax1.set_xlabel('Translation Error Standard Deviation (m)', color=color)
    ax1.tick_params(axis='x', labelcolor=color)
    ax2 = ax1.twiny()
    
    color = 'tab:purple'
    ax2.set_xlabel('Heading Error Standard Deviation (deg)', color=color)
    ax2.tick_params(axis='x', labelcolor=color)

    line_kwargs = {'linestyle': '-', 'marker': 's', 'markersize': 1.5, 'linewidth': 1.25}
    colors = ['peru', 'tomato', 'maroon']
    x, y = mh.get_metric_along_line('fp', lambda x: abs(x[1] - x[0] * 8.1712) < .01, 0, ret_type='avg')
    ax1.plot(x,y,**line_kwargs, color=colors[0])
    x, y = mh.get_metric_along_line('fn', lambda x: abs(x[1] - x[0] * 8.1712) < .01, 0, ret_type='avg')
    ax1.plot(x,y,**line_kwargs, color=colors[1])
    x, y = mh.get_metric_along_line('switch', lambda x: abs(x[1] - x[0] * 8.1712) < .01, 0, ret_type='avg')
    ax1.plot(x,y,**line_kwargs, color=colors[2])
    
    lims = ax1.get_ylim()
    ax1.set_ylim([lims[0], lims[1]]) 
    
    ax2.plot([0, 8.1712*x[-1]], [-10, -10])
    ax1.grid(True)
    ax1.legend(['false positives', 'misses', 'mismatches'])
    fig.set_dpi(240)
    # plt.text(2.6, 3.3, 'Casao')
    
    plt.tight_layout()
    
    if args.output:
        fig.subplots_adjust(
                top=0.78,
                bottom=0.22,
                left=0.08,
                right=0.98)
                # wspace=0.05, 
                # hspace=0.05)
        plt.savefig(args.output)
    else:
        plt.show()