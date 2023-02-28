class ObservationMsg():

    def __init__(self, tracker_id, mapped_ids, xbar, ell, a=None):
        self.tracker_id = tracker_id
        self.mapped_ids = mapped_ids
        self.xbar = xbar
        self.ell = ell
        self.z = None
        self.R = None
        self.H = None
        self.a = None
        self.destination = None
        self.has_measurement_info = False
        self.has_appearance_info = False
        
    def add_appearance(self, a):
        self.has_appearance_info = True
        self.a = a
        
    def add_measurement(self, z, R, H):
        self.has_measurement_info = True
        self.z = z
        self.R = R
        self.H = H
        
    def add_destination(self, dest):
        self.destination = dest
        
    def __str__(self):
        return f'observation from: {self.tracker_id}'