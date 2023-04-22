import numpy as np
import matplotlib.pyplot as plt
import argparse

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

def get_float_val(line, identifier):
    try:
        return float(line.split(identifier)[1].strip().split()[0])
    except:
        print('offending line:')
        print(line)
        print(identifier)
        assert False

def read_scores(metric_file, metric):
    scores = []
    T_diffs = []
    with open(metric_file, 'r') as f:
        for line in f.readlines():
            if metric in line:
                new_score = [[], [], [], []]
                new_score[0] = get_float_val(line, f'{metric}:')
            elif 'T_mag' in line:
                new_score[1] = get_float_val(line, 'T_mag:')
            elif 'psi_diff:' in line:
                new_score[2] = get_float_val(line, 'psi_diff:')
            elif 't_diff:' in line:
                new_score[3] = get_float_val(line, 't_diff:')
                scores.append(new_score)
                new_score = None
            
            if 'T_diff_list:' in line:
                pass
    return scores

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
args = parser.parse_args()

if not args.metric_file:
    args.metric_file = ['/home/masonbp/ford-project/data/mot_metrics/dynamic-final/3_rovers/vicon.yaml', 
                '/home/masonbp/ford-project/data/mot_metrics/dynamic-final/3_rovers/no_fix.yaml',
                '/home/masonbp/ford-project/data/mot_metrics/dynamic-final/3_rovers/wls_22_1s.yaml',
                '/home/masonbp/ford-project/data/mot_metrics/dynamic-final/3_rovers/smart_R_2_1_Rfagain.yaml']

score_list = []
if args.num_lines is None:
    num_lines = len(args.metric_file)
else:
    num_lines = args.num_lines
for metric_file in args.metric_file[:num_lines]:
    scores = read_scores(metric_file, args.metric)
    if 'old_mes' in metric_file:
        score_list.append(smooth_scores(scores))
    else:
        score_list.append(np.array(scores))
            




f, (ax) = plt.subplots(3, 1, sharex=True, figsize=[8, 4.8])

colors = ['tab:red', 'navy', 'tab:orange', 'tab:green']
# top plot MOTA
ts = list()
for i, s in enumerate(score_list):
    print(s.shape)
    t = [*range(s.shape[0])]
    t = np.array(t) * args.sample_len
    # print(t.shape)
    # print(s[:, 0].shape)
    ax[0].plot(t, s[:, 0], color=colors[i])
# ax1.plot(t, score_list[1][:, 0] - score_list[0][:,  0])
ax[0].set_ylabel('MOTA')
ax[0].set_ylim([0, 1])
# ax[0].legend(['VICON odometry', 'WLS realignment', 'original realignment'])
# ax[0].legend(['VICON odometry', 'WLS realignment', 'L515 detections'])
# ax[0].legend(['t265 w/ realignment', 't265 no realignment'])

minor_ticks = np.arange(0, t[-1], 5)
# ax[0].set_xticks(minor_ticks, minor=True)
# ax[0].grid(which='both')

# mid plot T_diff
for i, s in enumerate(score_list):
    print(s.shape)
    t = [*range(s.shape[0])]
    t = np.array(t) * args.sample_len
    ax[1].plot(t, s[:, 2], color=colors[i])
    ax[2].plot(t, s[:, 3], color=colors[i])
ax[1].set_ylabel('Heading error (deg)')
ax[1].grid(True)
# ax[1].set_xticks(minor_ticks, minor=True)
# ax[1].grid(which='both')
ax[2].set_ylabel('Translation error (m)')
ax[2].set_xlabel('Time (s)')
ax[1].legend(['Perfect localization', 'Casao', 'MOTLEE (realignment)', 'MOTLEE (realignment +\n uncertainty incorporation)'], loc=(0.01,0.3))
# ax[2].set_xticks(minor_ticks, minor=True)
# ax[2].grid(which='both')

ax[1].set_ylim([-.9, 17])
ax[2].set_ylim([-.15, 3.1])


for axi in ax:
    axi.grid(True)

# # bottom plot T_mag
# ax[-1].plot(t, score_list[-1][:, 1])
# ax[-1].set_xlabel('t (s)')
# ax[-1].set_ylabel('T_mag (m)')
# ax[-1].grid(True)

f.set_dpi(240)
# plt.show()

# with open('/home/masonbp/fig.png', 'w') as fh:
# plt.savefig('/home/masonbp/fig.png', format='png')
plt.show()