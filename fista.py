import torch
import numpy as np
from logistic_setup import prox_h
import math


def fista(grad, L, lam, w_init, options={}):
    '''FISTA for solving a general minimization problem
        problem setup:
            grad: gradient function, take one parameter w
            L: smoothness constant of the smooth part f
            lam: coefficient of L1-norm
        algorithm setup:
            w_init: initial point, should be chosen as w_k!!!
            options: a dict containing keys 'max_iter', 'stopping' and 'threshold'    
    '''
    # initialization
    options.setdefault('max_iter', 1000)
    options.setdefault('threshold', 1e-8)
    options.setdefault('store_seq', False)
    
    max_iter = options['max_iter']
    d = len(w_init)
    iter_num = 0
    x_old = w_init.clone()
    y_old = w_init.clone()
    t_old = 1

    if options['store_seq']:
        x_sequence = torch.zeros((max_iter, d))

    # main loop
    def error(w): # first-order optimal condition 
        return w - prox_h(w - grad(w), lam)

    while iter_num < options['max_iter']:
        if options['store_seq']:
            x_sequence[iter_num] = x_old
        x_new = prox_h(y_old - (1 / L) * grad(y_old), lam / L)
        t_new = 0.5 * (1 + math.sqrt(1 + 4 * t_old ** 2))
        y_new = x_new + (t_old - 1) / t_new * (x_new - x_old)

        e = torch.norm(error(x_new), p=2) / x_new.numel()
        if e < options['threshold']:
            break
        x_old, y_old, t_old = x_new.clone(), y_new.clone(), t_new
        iter_num += 1
    if options['store_seq']:
        return x_sequence, iter_num
    else:
        return x_new, iter_num