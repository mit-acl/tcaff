class ObservationMsg():

    def __init__(self, tracker_id, xbar, u, U, ell):
        self.tracker_id = tracker_id
        self.xbar = xbar
        self.u = u
        self.U = U
        self.ell = ell