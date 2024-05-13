![motlee](./media/banner.png)

![experiment](./media/experiment.gif)

## MOTLEE: <ins>M</ins>ulti-<ins>O</ins>bject <ins>T</ins>racking with <ins>L</ins>ocalization <ins>E</ins>rror <ins>E</ins>limination

MOTLEE is an algorithm used for performing collaborative multi-object tracking (MOT) in read-world environments (i.e. in the presence of imperfect or unknown relative robot localization). 
This involves performing data association of object measurements with tracked targets, sharing information across a distributed network, and filtering shared information for an accurate estimate. 
Importantly, MOTLEE involves performing frame alignment between pairs of robot without any initial information about relative robot localization using our Temporally Consistent Alignment of Frames Filter (TCAFF).

This repo contains code for performing multi-object tracking (useful for both collaborative or single-agent cases) and frame alignment between robots.

# Citation

If you use this code in your research, please cite our paper:

Mason B. Peterson, Parker C. Lusk, Antonio Avila, and Jonathan P. How. "MOTLEE: Collaborative Multi-Object Tracking Using Temporal Consistency for Neighboring Robot Frame Alignment." arXiv preprint arXiv:2405.05210 (2024).

```
@article{peterson2024motlee,
  title={MOTLEE: Collaborative Multi-Object Tracking Using Temporal Consistency for Neighboring Robot Frame Alignment},
  author={Peterson, Mason B and Lusk, Parker C and Avila, Antonio and How, Jonathan P},
  journal={arXiv preprint arXiv:2405.05210},
  year={2024}
}
```

# Installation

The `motlee` Python package can be installed with:

```
cd <this rep>
pip install .
```

Note: to install open3d 0.17.0, you may need to run:

```
pip install --upgrade pip
```

Additionally, the steps for installing [CLIPPER](https://github.com/mit-acl/clipper) should be followed:

```
git clone git@github.com:mit-acl/clipper.git
cd clipper
mkdir build
cd build
cmake ..
make
make pip-install
```

# Running

After downloading the MOTLEE dataset, the demo can be run with the following commands (after `cd`ing into this repo):

```
mkdir results
python3 ./demo/demo.py --params ./demo/params/motlee_full.yaml --output ./results/motlee_full.png
```

# Dataset

The MOTLEE dataset of four robots/six pedestrian object tracking experiment associated with the MOTLEE paper can be obtained by filling out [this Google Form](https://forms.gle/aKoQqBDXJVYe38mK9).

# Coming Soon

`motlee_ros`, a ROS wrapper for MOTLEE code will soon be released.

# Acknowledgements

MOTLEE was supported by the Ford Motor Company and by ARL DCIST.


