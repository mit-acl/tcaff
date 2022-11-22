class ObservationMsg():

    def __init__(self, tracker_id, xbar, u, U, ell, a=None):
        self.tracker_id = tracker_id
        self.xbar = xbar
        self.u = u
        self.U = U
        self.ell = ell
        
        if a is None:
            self.has_appearance_info = False
        else:
            self.has_appearance_info = True
        self.a = a
        
    def __str__(self):
        return f'observation from: {self.tracker_id}'