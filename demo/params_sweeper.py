import argparse
import motlee_runner

parser = argparse.ArgumentParser()
parser.add_argument("--params", "-p", type=str)
parser.add_argument("--viz", "-v", action="store_true")
parser.add_argument("--save-aligns", type=str, default=None, help="Directory to save alignment results to")
parser.add_argument("--save-pickle", "-s", type=str, default=None, help="Save data to pickle file for faster data loading")
parser.add_argument("--load-pickle", "-l", type=str, default=None, help="Load data from pickle file for faster data loading")
parser.add_argument("--output", "-o", type=str, default=None, help="Output file for plotting results")
parser.add_argument("--override-params", "-x", type=str, default=None, help="Override parameters in the yaml file")
parser.add_argument("--terse", "-t", action="store_true", help="Print only the average translation and rotation error")

args = parser.parse_args()

# args.params = "/home/masonbp/ford/motlee/demo/params/kmd2_new.yaml"
args.params = "/home/masonbp/ford/motlee/demo/params/kmd2_new.yaml"
args.terse = True
args.output = "/home/masonbp/ford/motlee/demo/tmp.png"
args.override_params = """
{
    "tcaff": {
        "main_tree_obj_req": 8.0, 
        "max_opt_fraction": 0.0,
        "clipper_sigma": 0.15
    }
}
"""

    # "tcaff": {
    #     "main_tree_obj_req": 8.0, 
    #     "max_opt_fraction": 0.5,
    #     "clipper_epsilon": 0.3,
    #     "clipper_sigma": 0.15,
    #     "wh_scale_diff": 1.5,
    #     "h_diff": 0.1
    # }
    # "mapping": {
    #     "Q_el":  0.2,
    #     "R_el":  0.2,
    #     "P0_el": 0.2,
    #     "tau": 10.0,
    #     "zmax": 8.0,
    #     "zmin": 1.5
    # }

motlee_runner.main(args)