import numpy as np
import matplotlib.pyplot as plt
import argparse

from utils import get_list_val

parser = argparse.ArgumentParser(description='Annotate video sequence with ground truth')
parser
parser.add_argument('-m', '--metric_file',
        # nargs=argparse.REMAINDER,
        type=str,
        default='/home/masonbp/ford/data/mot_dynamic/dynamic_motlee_iros/results/iros/mobile3_motlee_uncertainty.yaml',
        # default='/home/masonbp/ford-project/data/dynamic-final',
        help='Metric file to plot')
parser.add_argument('--sample-len',
                    default=0.5,
                    type=float)
parser.add_argument('--window-len',
                    default=5.0,
                    type=float)
parser.add_argument('-o', '--output', type=str, default=None)
parser.add_argument('-p', '--publish', action='store_true')
args = parser.parse_args()

def read_scores(metric_file):
    psi_diffs = []
    t_diffs = []
    with open(metric_file, 'r') as f:
        realign = False
        for line in f.readlines():
            # realign=True
            # if 'mota' in line:
            #     new_score = [[], [], [], []]
            #     new_score[0] = get_float_val(line, 'mota:')
            if 'REALIGN' in line:
                realign = True
            if 'psi_diff_list:' in line and realign:
                psi_diff_list = get_list_val(line, 'psi_diff_list:')
                psi_diffs.append(psi_diff_list)
            elif 't_diff_list:' in line and realign:
                t_diff_list = get_list_val(line, 't_diff_list:')
                t_diffs.append(t_diff_list)
                realign = False
    return psi_diffs, t_diffs

psi_diffs, t_diffs = [], []
# for metric_file in args.metric_file:
pd, td = read_scores(args.metric_file)
psi_diffs.append(np.array(pd).reshape(-1))
t_diffs.append(np.array(td).reshape(-1))



# mid plot T_diff
# for pd, td in zip(psi_diffs, t_diffs):
#     t = [*range(td.shape[0])]
#     t = np.array(t) * args.sample_len
#     for i in range(pd.shape[1]):
#         ax[0].plot(t, pd[:, i])
#         ax[1].plot(t, td[:, i])

print(f"heading mean: {np.mean(psi_diffs)}")
print(f"heading median: {np.median(psi_diffs)}")
print(f"translation mean: {np.mean(t_diffs)}")
print(f"translation median: {np.median(t_diffs)}")

if args.publish:
    plt.style.use('/home/masonbp/computer/python/matplotlib/publication.mplstyle')
    
width = 3.487
height = width / 1.618 * 1.

f, (ax) = plt.subplots(1, 2, sharey=True, figsize=(width, height), dpi=200)

psi_diffs[0] = sorted(psi_diffs[0].tolist())[1::2]
t_diffs[0] = sorted(t_diffs[0].tolist())[1::2]
# f.suptitle(r'$\hat{\mathbf{T}}_{ij}$ error')
ax[0].hist(psi_diffs[0], 15, ec='C0', lw=0.5)
ax[0].set_xlabel('Heading Error (deg)')
ax[1].hist(t_diffs[0], 15, ec='C0', lw=0.5)
ax[1].set_xlabel('Translation Error (m)')
f.set_dpi(240) 

if args.publish:
    f.subplots_adjust(
                top=0.96,
                bottom=0.16,
                left=0.08,
                right=0.97,
                wspace=.1)
    
if args.output is None:
    plt.show()
else:
    plt.savefig(args.output)