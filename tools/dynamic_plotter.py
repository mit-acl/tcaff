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
            json.loads(line.split(identifier)[1].strip())
            # import ipdb; ipdb.set_trace()
            return json.loads(line.split(identifier)[1].strip())
    except:
        print('offending line:')
        print(line)
        print(identifier)
        assert False

def read_scores(metric_file, args):
    metric = args.metric
    scores = []
    T_diffs = []
    with open(metric_file, 'r') as f:
        new_score = [None, None, None, None]
        for line in f.readlines():
            if metric in line:
                assert new_score[0] is None
                new_score[0] = get_float_val(line, f'{metric}:', is_list=args.metric_is_list)
                if None not in new_score: 
                    scores.append(new_score)
                    new_score = [None, None, None, None]
            elif 'T_mag' in line:
                assert new_score[1] is None
                new_score[1] = get_float_val(line, 'T_mag:')
                if None not in new_score: 
                    scores.append(new_score)
                    new_score = [None, None, None, None]
            elif 'psi_diff:' in line:
                assert new_score[2] is None
                new_score[2] = get_float_val(line, 'psi_diff:')
                if None not in new_score: 
                    scores.append(new_score)
                    new_score = [None, None, None, None]
            elif 't_diff:' in line:
                assert new_score[3] is None
                new_score[3] = get_float_val(line, 't_diff:')
                if None not in new_score: 
                    scores.append(new_score)
                    new_score = [None, None, None, None]
    return scores

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
args = parser.parse_args()

if not args.metric_file:
    args.metric_file = ['/home/masonbp/ford-project/data/mot_metrics/dynamic-final/3_rovers/vicon.yaml', 
                '/home/masonbp/ford-project/data/mot_metrics/dynamic-final/3_rovers/no_fix.yaml',
                '/home/masonbp/ford-project/data/mot_metrics/dynamic-final/3_rovers/wls_22_1s.yaml',
                '/home/masonbp/ford-project/data/mot_metrics/dynamic-final/3_rovers/smart_R_2_1_Rfagain.yaml']
metric = args.metric

############################################
############## Parsing  ####################
############################################

score_list = []
if args.num_lines is None:
    num_lines = len(args.metric_file)
else:
    num_lines = args.num_lines
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

colors = ['tab:red', 'navy', 'tab:orange', 'tab:green']

if not args.metric_only:
    f, (ax) = plt.subplots(3, 1, sharex=True, figsize=[8, 4.8])

    # top plot MOTA
    ts = list()
    for i, s in enumerate(score_list):
        t = [*range(s.shape[0])]
        t = np.array(t) * args.sample_len
        ax[0].plot(t, s[:, 0], color=colors[i])
    ax[0].set_ylabel('MOTA')
    ax[0].set_ylim([0, 1])

    minor_ticks = np.arange(0, t[-1], 5)

    # mid plot T_diff
    for i, s in enumerate(score_list):
        t = [*range(s.shape[0])]
        t = np.array(t) * args.sample_len
        ax[1].plot(t, s[:, 2], color=colors[i])
        ax[2].plot(t, s[:, 3], color=colors[i])
    ax[1].set_ylabel('Heading error (deg)')
    ax[1].grid(True)
    ax[2].set_ylabel('Translation error (m)')
    ax[2].set_xlabel('Time (s)')
    ax[1].legend(['Perfect localization', 'Casao', 'MOTLEE (realignment)', 'MOTLEE (realignment +\n uncertainty incorporation)'], loc=(0.01,0.3))

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
            ax.plot(t, s[:, 0], color=colors[i])
        elif not args.metric_plot_avg:
            for j, step in enumerate(s):
                data_pts = step[0]
                ax.scatter(np.ones(len(data_pts))*t[j], data_pts, color=colors[i])
    ax.set_ylabel(metric)
    if metric == 'mota':
        ax.set_ylim([0, 1])



# f.set_dpi(240)

if args.output is not None:
    with open(args.output, 'w') as fh:
        plt.savefig(args.output, format='png')
else:
    plt.show()