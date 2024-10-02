import numpy as np
# from numpy.linalg import inv, det
from copy import deepcopy

def inv(A):
    return np.diag(1/np.diag(A))

def det(A):
    return np.prod(np.diag(A))

class Node():
    
    def __init__(self, xhat, P, z=None, R=None, parent=None, objective=0):
        self.xhat = xhat
        self.P = P
        self.n = self.xhat.shape[0]
        self.z = z
        self.R = R
        
        self.parent = parent
        self.children = []
        self.objective = objective
        if self.parent is None:
            self.objective_sum = objective 
            self.measurements_in_hyp = 0 if z is None else 1
        else: 
            # Note: Not sure this objective_sum is working right now. Need to check before going back (right now using cumulative sum function)
            self.objective_sum = objective + parent.objective
            self.measurements_in_hyp = parent.measurements_in_hyp + (0 if z is None else 1)
        
    def add_model(self, A, H, Q):
        self.A = A
        self.H = H
        self.Q = Q
        
    def mahalanobis_diff(self, z, R):
        xhat_plus = self.A @ self.xhat
        # print(f'z: {z.reshape(-1).tolist()}, xhat_plus: {xhat_plus.reshape(-1).tolist()}')
        S = self.H @ self.P  @ self.H.T + R
        mahalanobis_diff = (z - self.H @ xhat_plus).T @ inv(S) @ (z - self.H @ xhat_plus)
        return mahalanobis_diff.item(0)
    
    def nlml(self, z, R):
        p = z.shape[0]
        # import ipdb; ipdb.set_trace()
        S = self.H @ self.P  @ self.H.T + R
        nlml = .5 * (self.mahalanobis_diff(z, R)**2 + np.log(det(S)) + p*np.log(2*np.pi))
        # print(f'nlml: {nlml}')
        return nlml
    
    def create_children(self, zs, Rs, objectives):
        new_children = []
        for z, R, obj in zip(zs, Rs, objectives):
            new_children.append(self.create_child(z, R, obj))
        return new_children
            
    def create_child(self, z, R, objective):
        xhat_plus = self.A @ self.xhat
        P_plus = self.A@self.P@self.A.T + self.Q

        L = P_plus @ self.H.T @ inv(self.H@P_plus@self.H.T + R)
        xhat = xhat_plus + L @ (z - self.H@xhat_plus)
        P = (np.eye(self.n) - L@self.H) @ P_plus
                
        self.children.append(Node(xhat, P, z=z, R=R, parent=self, objective=objective))
        self.children[-1].add_model(A=self.A, H=self.H, Q=self.Q)
        return self.children[-1]
    
    def create_measurementless_child(self, objective):
        xhat = self.A @ self.xhat
        P = self.A@self.P@self.A.T + self.Q
        
        new_child = Node(xhat, P, z=None, R=None, parent=self, objective=objective)
        new_child.add_model(A=self.A, H=self.H, Q=self.Q)
        self.children.append(new_child)
        
        return new_child
        
    def add_child(self, child):
        self.children.append(child)
        
    def cumulative_objective(self):
        n = self
        obj = 0.
        while n is not None:
            obj += n.objective
            n = n.parent
        return obj
    
    def __lt__(self, obj):
        assert type(obj) == Node
        return self.cumulative_objective() < obj.cumulative_objective()
    
    def __str__(self):
        return f'({self.xhat.reshape(-1).tolist()}, {self.z.reshape(-1).tolist() if self.z is not None else None}, {self.cumulative_objective()})'
