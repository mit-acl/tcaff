import argparse
import matplotlib.pyplot as plt
import sys
import numpy as np

sys.path.append('..')
from rover_output_parser import RoverOutputParser

parser = argparse.ArgumentParser()
parser.add_argument('input_file', 
                    type=str,
                    default='/home/masonbp/ford/data/mot_dynamic/dynamic_motlee_iros/results/3_rovers/001_realign_baseline.yaml',
                    help='input file')

args=parser.parse_args()

rover_parser = RoverOutputParser(args.input_file)
rover_parser.add_metric('psi_diff_list', has_multiple_vals=True)
rover_parser.add_metric('t_diff_list', has_multiple_vals=True)
rover_parser.parse(format=dict, skip_none_lines=True)

num_pairs = rover_parser.get_array('t_diff_list').shape[1]

fig, ax = plt.subplots(num_pairs, 2)

for i in range(num_pairs):
    for j, metric in enumerate(['psi_diff_list', 't_diff_list']):
        ax[i,j].plot(np.arange(rover_parser.num_entries()), rover_parser.get_array(metric)[:,i])

# for entry in results:
#     for t_diff, resid, num_cones in zip(entry['t_diff_list'], entry['residuals'], entry['num_cones']):
#         ax.plot([resid/num_cones], [t_diff], 'o', color='red', markersize=2)

# ax.set_xlabel('residual')
# ax.set_ylabel('t_diff (m)')
plt.show()