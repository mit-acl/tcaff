import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
sys.path.insert(0, '/home/masonbp/ford/motlee/tools')
from metric_plotter import parse_metric_file

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', default=None, type=str)
parser.add_argument('-p', '--publish', action='store_true')
args = parser.parse_args()

# og_file = '/home/masonbp/ford-project/data/mot_metrics/run03_rot_tran.txt'
# fix1_file = '/home/masonbp/ford-project/data/mot_metrics/run03_realign.txt'
# fix2_file = '/home/masonbp/ford-project/data/mot_metrics/run03_realign_grow_tau_3.yaml'
# fix3_file = '/home/masonbp/ford-project/data/mot_metrics/run03_realign.txt'
# og_file = '/home/masonbp/ford-project/data/mot_metrics/run03_half2/no_fix_2.yaml'
og_file = '/home/masonbp/ford/data/mot_static/static-2022/iros/mota_casao.yaml'
fix1_file = '/home/masonbp/ford/data/mot_static/static-2022/iros/mota_motlee.yaml'
fix2_file = '/home/masonbp/ford/data/mot_static/static-2022/iros/mota_motlee_reactive_tau.yaml'
num_methods_to_plot = 2
# plot_type = 'trans'
# plot_type = 'rot'
plot_type = 'rot_trans'

mh_og = parse_metric_file(og_file, ['mota', 'fp', 'fn', 'switch'])
# print(mh_og)
x_t, y_t = mh_og.get_metric_along_line('mota', lambda x: x[1] == 0.0, 0, ret_type='avg') # translation
x_r, y_r = mh_og.get_metric_along_line('mota', lambda x: x[0] == 0.0, 1, ret_type='avg') # rotation
x_c, y_c = mh_og.get_metric_along_line('mota', lambda x: abs(x[1] - x[0] * 8.1712) < .01, 1, ret_type='avg') # combined

if num_methods_to_plot > 0:
    mh_fix1 = parse_metric_file(fix1_file, ['mota', 'fp', 'fn', 'switch'])
    x_t_1, y_t_1 = mh_fix1.get_metric_along_line('mota', lambda x: x[1] == 0.0, 0, ret_type='avg') # translation
    x_r_1, y_r_1 = mh_fix1.get_metric_along_line('mota', lambda x: x[0] == 0.0, 1, ret_type='avg') # rotation
    x_c_1, y_c_1 = mh_fix1.get_metric_along_line('mota', lambda x: abs(x[1] - x[0] * 8.1712) < .01, 1, ret_type='avg') # combined

if num_methods_to_plot > 1:
    mh_fix2 = parse_metric_file(fix2_file, ['mota', 'fp', 'fn', 'switch'])
    x_t_2, y_t_2 = mh_fix2.get_metric_along_line('mota', lambda x: x[1] == 0.0, 0, ret_type='avg') # translation
    x_r_2, y_r_2 = mh_fix2.get_metric_along_line('mota', lambda x: x[0] == 0.0, 1, ret_type='avg') # rotation
    x_c_2, y_c_2 = mh_fix2.get_metric_along_line('mota', lambda x: abs(x[1] - x[0] * 8.1712) < .01, 1, ret_type='avg') # combined

print(mh_fix1)
print(mh_fix2)
print(mh_og)

width = 3.487
height = width / 1.618 * 1.
plt.style.use('/home/masonbp/computer/python/matplotlib/publication.mplstyle')
fig, ax1 = plt.subplots(figsize=(width, height), dpi=200)
lns = []
line_kwargs = {'linestyle': '-', 'marker': 's', 'markersize': 1.5, 'linewidth': 1.25}

if num_methods_to_plot == 0:
    lbs = ['Translation & Heading Error',
       'MOTA = 0.5 ref', 'MOTA = 0.0 ref']
elif num_methods_to_plot > 1:
    lbs = ['Casao', r'MOTLEE (constant $\tau_{gate}$)', r'MOTLEE (reactive $\tau_{gate}$)', 
       'MOTA = 0.5 ref', 'MOTA = 0.0 ref']
else:
    lbs = ['Translation & Heading Error', 'WLS Realignment',
        'MOTA = 0.5 ref', 'MOTA = 0.0 ref']

color = 'tab:blue'
ax1.set_xlabel('Translation Error Standard Deviation (m)', color=color)
ax1.set_ylabel('MOTA')
ax1.tick_params(axis='x', labelcolor=color)
if plot_type == 'trans':
    lns += ax1.plot(x_t, y_t, '--o', color=color)
    lns += ax1.plot(x_t_1, y_t_1, '--o', color='orange')
    if num_methods_to_plot > 1:
        lns += ax1.plot(x_t_2, y_t_2, '--o')
ax1.grid(True)

ax2 = ax1.twiny()

color = 'tab:purple'
ax2.set_xlabel('Heading Error Standard Deviation (deg)', color=color)
if plot_type == 'rot':
    lns += ax2.plot(x_r, y_r, '--o', color=color)
    lns += ax2.plot(x_r_1, y_r_1, '--o', color='orange')
    if num_methods_to_plot > 1:
        lns += ax2.plot(x_r_2, y_r_2, '--o', color=color)
ax2.tick_params(axis='x', labelcolor=color)
if plot_type == 'rot_trans':
    lns += ax2.plot(x_c, y_c, **line_kwargs, color='navy')
    if num_methods_to_plot > 0:
        lns += ax2.plot(x_c_1, y_c_1, **line_kwargs, color='#FA86C4')
    if num_methods_to_plot > 1:
        lns += ax2.plot(x_c_2, y_c_2, **line_kwargs, color='green')

# lns += ax1.plot([min(x_t), max(x_t)], [0.5, 0.5], '--', color='orange')
# lns += ax1.plot([min(x_t), max(x_t)], [0.0, 0.0], 'r--')
# lns += 

# plt.title(f'Static Test Results')

ax1.legend(lns, lbs)
fig.subplots_adjust(
                top=0.85,
                bottom=0.15,
                left=0.11,
                right=0.98)

if args.output is not None:
    if args.publish:
        ax1.plot([-0.5, 2.5], [0.5, 0.5], '-', color='orange')
        ax1.plot([-0.5, 2.5], [0.0, 0.0], '-', color='red')
        ax1.set_xlim([-.1, 2.1])
        ax1.set_ylim([-.08, 1.05])
    plt.savefig(args.output)
else:
    plt.show()
