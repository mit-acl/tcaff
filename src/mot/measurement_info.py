class MeasurementInfo():

    def __init__(self, track_id, mapped_ids, xbar, ell):
        self.track_id = track_id
        self.mapped_ids = mapped_ids
        self.xbar = xbar
        self.ell = ell
        self.zs = None
        self.u = None
        self.U = None
        self.destination = None
        self.has_measurement_info = False
        
    def add_measurements(self, zs, u, U):
        self.has_measurement_info = True
        self.zs = zs
        self.u = u
        self.U = U
        
    def add_destination(self, dest):
        self.destination = dest
        
    def __str__(self):
        return f'observation from: {self.track_id}'