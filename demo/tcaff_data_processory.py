import numpy as np
from typing import List

try:
    from .robot import Robot
except:
    from robot import Robot

class TCAFFDataProcessor():

    def __init__(
            self, 
            robots: List[Robot], 
            mapping_ts: float = .1,
            frame_align_ts: float = 1.0,
            perform_mot: bool = True,
            mot_ts: float = None,
        ) -> None:

        self.robots = robots
        self.mapping_ts = mapping_ts
        self.frame_align_ts = frame_align_ts
        self.mot_ts = mot_ts
        self.perform_mot = perform_mot

        assert self.mot_ts is not None or not self.perform_mot, "Must specify MOT ts if performing MOT"

        self.last_mot_t = -np.inf
        self.last_mapping_t = -np.inf
        self.last_fa_t = -np.inf
        self.fa_updated = False

    def update(self, t: float):
        run_mot = self.perform_mot and np.round(t - self.last_mot_t, 4) >= self.mot_ts
        run_mapping = np.round(t - self.last_mapping_t, 4) >= self.mapping_ts
        run_frame_align = np.round(t - self.last_fa_t, 4) >= self.frame_align_ts

        # Update each robots local map
        if run_mapping:
            self.last_mapping_t = t
            for robot in self.robots:
                robot.update_mapping(t)

        # Run frame alignment
        if run_frame_align:
            self.fa_updated = True
            self.last_fa_t = t
            # exchange map data
            for r1 in self.robots:
                for r2 in self.robots:
                    r1.set_neighbor_map(r2.name, r2.get_map())

            for robot in self.robots:
                robot.update_frame_alignments()
        else:
            self.fa_updated = False
            
        # Run Multi-object tracking
        if run_mot:
            self.last_mot_t = t
            for robot in self.robots:
                robot.update_mot_local_info(t)
            
            # exchange MOT info
            for r1 in self.robots:
                for r2 in self.robots:
                    r1.mot_info[r2.name] = r2.get_mot_info()

            # update trackers with exchanged data
            for robot in self.robots:
                robot.update_mot_global_info()
            