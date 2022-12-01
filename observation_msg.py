class ObservationMsg():

    def __init__(self, tracker_id, mapped_ids, xbar, ell, a=None):
        self.tracker_id = tracker_id
        self.mapped_ids = mapped_ids
        self.xbar = xbar
        self.ell = ell
        self.u = None
        self.U = None
        self.a = None
        self.has_measurement_info = False
        self.has_appearance_info = False
        
    def add_appearance(self, a):
        self.has_appearance_info = True
        self.a = a
        
    def add_measurement(self, u, U):
        self.has_measurement_info = True
        self.u = u
        self.U = U
        
    def __str__(self):
        return f'observation from: {self.tracker_id}'