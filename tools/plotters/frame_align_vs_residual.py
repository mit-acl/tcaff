import argparse
import matplotlib.pyplot as plt
import sys

sys.path.append('..')
from rover_output_parser import RoverOutputParser

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', 
                    type=str,
                    default='/home/masonbp/ford/data/mot_dynamic/dynamic_motlee_iros/results/3_rovers/001_realign_baseline.yaml',
                    help='input file')

args=parser.parse_args()

rover_parser = RoverOutputParser(args.file)
rover_parser.add_metric('psi_diff_list', has_multiple_vals=True)
rover_parser.add_metric('t_diff_list', has_multiple_vals=True)
rover_parser.add_metric('residuals', has_multiple_vals=True)
rover_parser.add_metric('num_cones', has_multiple_vals=True)
results = rover_parser.parse(format=dict, skip_none_lines=True)


fig, ax = plt.subplots()
for entry in results:
    for t_diff, resid, num_cones in zip(entry['t_diff_list'], entry['residuals'], entry['num_cones']):
        ax.plot([resid/num_cones], [t_diff], 'o', color='red', markersize=2)

ax.set_xlabel('residual')
ax.set_ylabel('t_diff (m)')
plt.show()