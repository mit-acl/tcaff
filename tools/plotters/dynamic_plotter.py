import numpy as np
import matplotlib.pyplot as plt
import argparse
import json

def smooth_scores(scores):
    scores = np.array(scores)
    num_samples_per_window = int(args.window_len / args.sample_len)
    half_window_size = int(num_samples_per_window/2)
    scores_smoothed = np.copy(scores)
    for i in range(scores.shape[0]):
        if np.isnan(scores[i, 0]):
            scores[i, 0] = -1.0
        scores[i, 0] = max(-1.0, scores[i, 0])
    for i in range(scores.shape[0]):
        if i < num_samples_per_window / 2:
            scores_smoothed[i, 0] = np.mean(scores[:i+half_window_size+1, 0])
        elif i > scores.shape[0] - num_samples_per_window / 2:
            scores_smoothed[i, 0] = np.mean(scores[i-half_window_size+1:, 0])
        else:
            scores_smoothed[i, 0] = np.mean(scores[i-half_window_size+1:i+half_window_size+1, 0])
    scores = scores_smoothed
    return scores

def get_float_val(line, identifier, is_list=False):
    try:
        if not is_list:
            return float(line.split(identifier)[1].strip().split()[0])
        else:
            list_str = line.split(identifier)[1].strip()
            list_floats = [float(sub_str.strip()) for sub_str in list_str.split('[')[1].split(']')[0].strip().split(',')]
            return list_floats
    except:
        print('offending line:')
        print(line)
        print(identifier)
        assert False

def read_scores(metric_file, args):
    target_metric = args.metric
    scores = []
    T_diffs = []
    with open(metric_file, 'r') as f:
        new_score = [None, None, None, None]
        for line in f.readlines():
            metric_list = [target_metric, 'T_mag', 'psi_diff', 't_diff']
            metric_idx = [0, 1, 2, 3]
            metric_is_list = [args.metric_is_list, False, False, False]
            for metric, idx, is_list in zip(metric_list, metric_idx, metric_is_list):
                if f'{metric}:' in line:
                    assert new_score[idx] is None, f'{metric}'
                    new_score[idx] = get_float_val(line, f'{metric}:', is_list=is_list)
                    if None not in new_score: 
                        scores.append(new_score)
                        new_score = [None, None, None, None]
    return scores

def smooth(scalars, weight): # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        if np.isnan(last):
            last = point
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        if not np.isnan(smoothed_val):
            last = smoothed_val # if not np.isnan(smoothed_val) else last                             # Anchor the last smoothed value
        
    return smoothed

############################################
########### Args Setup #####################
############################################

parser = argparse.ArgumentParser(description='Annotate video sequence with ground truth')
parser
parser.add_argument('metric_file',
        nargs=argparse.REMAINDER,
        type=str,
        default=None,
        help='Metric file to plot')
parser.add_argument('--sample-len',
                    default=0.5,
                    type=float)
parser.add_argument('--window-len',
                    default=5.0,
                    type=float)
parser.add_argument('--num-lines', '-n', default=None, type=int)
parser.add_argument('--metric', '-m', default='mota', type=str)
parser.add_argument('--metric-only', action='store_true')
parser.add_argument('--metric-is-list', action='store_true')
parser.add_argument('--metric-plot-avg', action='store_true')
parser.add_argument('--output', '-o', default=None, type=str)
parser.add_argument('--legend', '-l', type=str, default=None,
                    help='legend formatted as python list')
parser.add_argument('--dpi', type=int, default=240)
parser.add_argument('--publish', '-p', action='store_true')
args = parser.parse_args()

if not args.metric_file:
    args.metric_file = ['/home/masonbp/ford/data/mot_dynamic/dynamic_motlee_iros/results/iros/mobile0_vicon.yaml', 
                '/home/masonbp/ford/data/mot_dynamic/dynamic_motlee_iros/results/iros/mobile1_casao.yaml',
                '/home/masonbp/ford/data/mot_dynamic/dynamic_motlee_iros/results/iros/mobile2_motlee_no_uncertainty.yaml',
                '/home/masonbp/ford/data/mot_dynamic/dynamic_motlee_iros/results/iros/mobile3_motlee_uncertainty.yaml']
metric = args.metric
if args.num_lines is None:
    num_lines = len(args.metric_file)
else:
    num_lines = args.num_lines
if args.legend is None:
    legend = [f'run {i}' for i in range(num_lines)]
else:
    legend = [element.strip() for element in args.legend.strip().split('[')[1].split(']')[0].split(',')]
    
plt.style.use('/home/masonbp/computer/python/matplotlib/publication.mplstyle')
if args.publish:
    legend = ["Perfect localization", "Casao", "MOTLEE, no frame alignment uncertainty", "MOTLEE, frame alignment uncertainty"]
    


############################################
############## Parsing  ####################
############################################

score_list = []
for metric_file in args.metric_file[:num_lines]:
    scores = read_scores(metric_file, args)
    if 'old_mes' in metric_file:
        score_list.append(smooth_scores(scores))
    elif not args.metric_is_list:
        score_list.append(np.array(scores))
    else:
        score_list.append(scores)

############################################
############## Plotting  ###################
############################################

colors = ['goldenrod', 'navy', '#FA86C4', 'green']
line_kwargs = {'linestyle': '-', 'linewidth': 1.25}

if not args.metric_only:
    width = 3.487
    height = width * 1.
    f, (ax) = plt.subplots(3, 1, sharex=True, figsize=(width, height))

    # top plot MOTA
    ts = list()
    for i, s in enumerate(score_list):
        t = [*range(s.shape[0])]
        t = np.array(t) * args.sample_len
        ax[0].plot(t, s[:, 0], color=colors[i], **line_kwargs)
    ax[0].set_ylabel('MOTA')
    ax[0].set_ylim([0, 1])
    ax[0].set_yticks([0., .2, .4, .6, .8, 1.], minor=False)
    

    minor_ticks = np.arange(0, t[-1], 5)

    # mid plot T_diff
    for i, s in enumerate(score_list):
        t = [*range(s.shape[0])]
        t = np.array(t) * args.sample_len
        ax[1].plot(t, s[:, 2], color=colors[i],  **line_kwargs)
        ax[2].plot(t, s[:, 3], color=colors[i],  **line_kwargs)
    ax[1].set_ylabel('Heading Error (deg)')
    ax[1].grid(True)
    ax[2].set_ylabel('Translation Error (m)')
    ax[2].set_xlabel('Time (s)')
    
    ax[1].set_ylim([-.9, 17])
    ax[2].set_ylim([-.15, 3.1])

    for axi in ax:
        axi.grid(True)

else:
    f, ax = plt.subplots()
    ax.grid(True)
    ts = list()
    for i, s in enumerate(score_list):
        t = [*range(len(s))]
        t = np.array(t) * args.sample_len
        if not args.metric_is_list:
            ax.plot(t, s[:, 0], color=colors[i],  **line_kwargs)
        elif args.metric_plot_avg:
            avgs = []
            for j, step in enumerate(s):
                avgs.append(np.mean(step[0]))
            ax.plot(t, smooth(avgs, 0.9), color=colors[i],  **line_kwargs)
        else:
            for j, step in enumerate(s):
                data_pts = step[0]
                ax.scatter(np.ones(len(data_pts))*t[j], data_pts, color=colors[i])
    ax.set_ylabel(metric)
    ax.set_xlabel('time (s)')
    ax.legend(legend)
    if metric == 'mota':
        ax.set_ylim([0, 1])

f.set_dpi(args.dpi)
ax[0].legend(legend, loc='upper left', ncols=2, bbox_to_anchor=(0.0,1.4), frameon=False)

if args.publish:
    f.subplots_adjust(
                top=0.91,
                bottom=0.1,
                left=0.11,
                right=0.98)

if args.output is not None:
    plt.savefig(args.output)
else:
    plt.show()