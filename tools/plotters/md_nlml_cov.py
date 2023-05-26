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

def read_scores(metric_file, target_metric, metric_is_list):
    scores = []
    T_diffs = []
    with open(metric_file, 'r') as f:
        new_score = [None, None, None, None]
        for line in f.readlines():
            metric_list = [target_metric, 'T_mag', 'psi_diff', 't_diff']
            metric_idx = [0, 1, 2, 3]
            metric_is_list_list = [metric_is_list, False, False, False]
            for metric, idx, is_list in zip(metric_list, metric_idx, metric_is_list_list):
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
parser.add_argument('--num-lines', '-n', default=2, type=int)
parser.add_argument('--metric', '-m', default='det(V)', type=str)
parser.add_argument('--output', '-o', default=None, type=str)
parser.add_argument('--thresh', '-t', type=float, default=25.0, help='plot line when metric >= tresh')
args = parser.parse_args()

if not args.metric_file:
    args.metric_file = ['/home/masonbp/ford-project/data/mot_metrics/dynamic-final/3_rovers/run45_orig_md_covs.json', 
                '/home/masonbp/ford-project/data/mot_metrics/dynamic-final/3_rovers/run43_more_covs.json',
                '/home/masonbp/ford-project/data/mot_metrics/dynamic-final/3_rovers/vicon.yaml']
metric = args.metric
num_lines = args.num_lines
assert num_lines == 2 or num_lines == 3
thresh = args.thresh
# legend = ['MD', 'NLML', f'{metric} >= {thresh}', 'Vicon']
legend = ['MD', 'NLML', f'covariance > {thresh}', 'Vicon']


############################################
############## Parsing  ####################
############################################

score_list = []
for metric_file in args.metric_file[:num_lines]:
    scores = read_scores(metric_file, target_metric='mota', metric_is_list=False)
    score_list.append(np.array(scores))
    
# Add on cov info
scores = read_scores(args.metric_file[0], target_metric=args.metric, metric_is_list=True)
score_list.append(scores)

############################################
############## Plotting  ###################
############################################

colors = ['tab:red', 'navy', 'tab:green', 'tab:orange',]

f, (ax) = plt.subplots(2,1)
for a in ax:
    a.grid(True)

ts = list()
for i, s in enumerate(score_list[:-1]):
    t = [*range(len(s))]
    t = np.array(t) * args.sample_len
    ax[0].plot(t, s[:, 0], color=colors[i])
    
avgs = []
for j, step in enumerate(score_list[-1]):
    if len(step[0]) > 0:
        avgs.append(np.mean(step[0]))
    else:
        avgs.append(np.nan)
smoothed_avgs = smooth(avgs, 0.5)
ax[1].plot(t, smoothed_avgs, color=colors[3])

cov_over_thresh = np.zeros(len(score_list[-1]))
# import ipdb; ipdb.set_trace()
for i, a in enumerate(smoothed_avgs):
    if a > thresh:
        cov_over_thresh[i] = 0.95
    else:
        cov_over_thresh[i] = np.nan
ax[0].plot(t, cov_over_thresh, '.', color=colors[3], markersize=2)
ax[1].plot([t[0], t[-1]], np.ones(2)*thresh, '--', color='k')
    
ax[0].set_ylabel('MOTA')
ax[0].set_ylim([0, 1])

ax[1].set_ylabel('det(innovation covariance)')
ax[1].set_xlabel('time (s)')
ax[0].legend(legend)



f.set_dpi(240)

if args.output is not None:
    f.set_dpi(1000)
    with open(args.output, 'w') as fh:
        plt.savefig(args.output, format='png')
else:
    plt.show()