from collections import deque
from typing import Optional
import torch


class BFGS:  # to manage BFGS related attributes
    def __init__(self, k_max=10):
        self.k_max = k_max
        self.s = deque(maxlen=k_max) # store s_i in form of ndarray!
        self.y = deque(maxlen=k_max)
        self.k_size = 0  # current size
        self.sigma = Optional[float]  # sigma_k
        self.SY = None # stores all s_i y_i.T, hence has shape (d, m, m)
        self.SS = None # stores all s_i s_i.T, hence has shape (d, m, m)
        self.YY = None # stores all y_i y_i.T, hence has shape (d, m, m)
        self.U = None
        self.J = None
        self.A = None # A = U.T @ U - J
        self.big_SY = None # same as U
        self.big_SYT = None # big_SY.T
        self.K_0 = None # J/(sigma - alpha)
        self.K_k = None # A/(sigma - alpha)
 
def LBFGS(z, bfgs):  # bfgs is a realization of class BFGS
    # realization of inverse Hessian-vector product with H generated from Algo 2 (Hessian Update)
    q = z
    alpha = []
    m = bfgs.k_size
    for i in range(1, m + 1):
        rho = 1 / torch.dot(bfgs.y[-i], bfgs.s[-i])
        alpha.append(rho * torch.matmul(bfgs.s[-i].T, q))
        q = q - alpha[-1] * bfgs.y[-i]
    r = 1 / bfgs.sigma * q
    alpha.reverse()
    for j in range(m, 0, -1):
        rho = 1 / torch.dot(bfgs.y[-j], bfgs.s[-j])
        beta = rho * torch.dot(bfgs.y[-j].T, r)
        r = r + (alpha[-j] - beta) * bfgs.s[-j]
    return r