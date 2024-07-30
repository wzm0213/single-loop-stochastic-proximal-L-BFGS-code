import time

import numpy as np
import torch
from tqdm import tqdm

from bfgs_class import BFGS, LBFGS
from logistic_setup import f_grad, f_hess_z, F, prox_h
from other_inner_solver import scaled_prox_h
from ssn_method import ssn

torch.manual_seed(0)

def splbfgs(X, y, b, bH, M = 10, L=10, alpha=0.01, prob=0.01, w=None, n_epochs=100, lam=0.01, mu=0.01, optimizer='fista', window_size=1000):
    r"""Stochastic Proximal L-BFGS for logistic regression with L1 regularization

    Args:
        X: training data
        y: training label
        b: batch size for gradient
        bH: batch size for Hessian
        M: maximum memory size
        L: update frequency for correction pairs
        alpha: (initial) learning rate
        prob: probability to update reference point
        w: initial guess
        n_epochs: number of iterations
        lam: L_1 regularization parameter
        mu: L_2 regularization parameter
        optimizer: inner solver, available ones include 'fista', 'ista' and 'ssn'
        window_size: window size for learning rate decay (optional)

    Returns:
        loss: loss history
        w_sequence: w history
        info: additional information
    """
    n, d = X.shape
    bfgs = BFGS(k_max=M)
    if w is None:
        w = 0.01 * torch.ones(d)
    w_hat = w.clone()  # center
    t = 0
    w_bar_old = torch.zeros(d)
    w_bar_new = torch.zeros(d)

    loss = torch.zeros(n_epochs)
    w_sequence = torch.zeros((n_epochs, d))

    center_grad = f_grad(w_hat, X, y, mu)
    lr = alpha

    # only for ssn
    acc_time = 0
    max_inner_iter = 0
    ave_inner_iter = 0
    max_ave_search_len = 0

    store_time = 0 # for testing the time of storing correction pairs

    for k in tqdm(range(n_epochs)):
        if k % window_size == 0 and k != 0:
            lr /= 2
        w_old = w.clone()  # store x_k
        p = torch.randperm(n) # shuffle data

        # store current loss
        loss[k] = F(w, X, y, lam, mu)
        w_sequence[k] = w.clone()

        # update correction pairs bfgs.s and bfgs.y
        if k % L == 0 and k > 0:
            t += 1
            w_bar_new /= L
            if t > 0:
                S_H = p[:bH]  # sample for Hessian
                s_new = w_bar_new - w_bar_old
                y_new = f_hess_z(w_bar_new, X[p[S_H]], s_new, mu)
                w_bar_old = w_bar_new.clone()
                w_bar_new = torch.zeros(d)

                s_new = s_new.numpy()
                y_new = y_new.numpy()

                if s_new.dot(y_new) > 1e-15:
                    if bfgs.k_size < M:
                        bfgs.s.append(s_new)
                        bfgs.y.append(y_new)
                        bfgs.k_size += 1
                        S = torch.tensor(np.array(bfgs.s).T, dtype=torch.float32)
                        Y = torch.tensor(np.array(bfgs.y).T, dtype=torch.float32)
                        s_new = torch.from_numpy(s_new)
                        y_new = torch.from_numpy(y_new)
                        # update bfgs.SS, bfgs.SY, bfgs.YY
                        if bfgs.k_size == 1:
                            bfgs.SS = (s_new ** 2).view(-1, 1, 1)
                            bfgs.SY = (s_new * y_new).view(-1, 1, 1)
                            bfgs.YY = (y_new ** 2).view(-1, 1, 1)
                        else:
                            app_SS = s_new.view(-1, 1) * S
                            app_SY = s_new.view(-1, 1) * Y
                            app_YS = y_new.view(-1, 1) * S
                            app_YY = y_new.view(-1, 1) * Y
                            bfgs.SS = torch.cat((torch.cat((bfgs.SS, app_SS.view(d, 1, -1)[:, :, :-1]), dim=1), app_SS.view(d, -1, 1)), dim=2)
                            bfgs.SY = torch.cat((torch.cat((bfgs.SY, app_SY.view(d, 1, -1)[:, :, :-1]), dim=1), app_YS.view(d, -1, 1)), dim=2)
                            bfgs.YY = torch.cat((torch.cat((bfgs.YY, app_YY.view(d, 1, -1)[:, :, :-1]), dim=1), app_YY.view(d, -1, 1)), dim=2)
                    else:
                        t1 = time.time()
                        bfgs.s.append(s_new) # automatically popleft
                        bfgs.y.append(y_new)
                        S = torch.tensor(np.array(bfgs.s).T, dtype=torch.float32)
                        Y = torch.tensor(np.array(bfgs.y).T, dtype=torch.float32)
                        s_new = torch.from_numpy(s_new)
                        y_new = torch.from_numpy(y_new)
                        app_SS = s_new.view(-1, 1) * S
                        app_SY = s_new.view(-1, 1) * Y
                        app_YS = y_new.view(-1, 1) * S
                        app_YY = y_new.view(-1, 1) * Y
                        bfgs.SS = torch.cat((torch.cat((bfgs.SS[:, 1:, 1:], app_SS.view(d, 1, -1)[:, :, :-1]), dim=1), app_SS.view(d, -1, 1)), dim=2)
                        bfgs.SY = torch.cat((torch.cat((bfgs.SY[:, 1:, 1:], app_SY.view(d, 1, -1)[:, :, :-1]), dim=1), app_YS.view(d, -1, 1)), dim=2)
                        bfgs.YY = torch.cat((torch.cat((bfgs.YY[:, 1:, 1:], app_YY.view(d, 1, -1)[:, :, :-1]), dim=1), app_YY.view(d, -1, 1)), dim=2)
                        t2 = time.time()
                        store_time += t2 - t1 # for testing the time of storing correction pairs

        # compute variance reduced stochastic gradient
        grad = f_grad(w, X[p[:b]], y[p[:b]], mu) - f_grad(w_hat, X[p[:b]], y[p[:b]], mu) + center_grad

        w_bar_new += w

        # update w
        if k < L:
            z = w - lr * grad
            w = prox_h(z, lr * lam)

        else:
            # update bfgs information
            if bfgs.k_size == 0:
                bfgs.sigma = 1.0
            else:
                bfgs.sigma = bfgs.y[-1] @ bfgs.y[-1] / (bfgs.s[-1] @ bfgs.y[-1])
        
            if optimizer == 'fista' or optimizer == 'ista':
                z = w - lr * (LBFGS(grad, bfgs))
                w, iter = scaled_prox_h(z, bfgs, lr * lam, optimizer=optimizer)
                ave_inner_iter += iter
                max_inner_iter = max(max_inner_iter, iter)

            elif optimizer == 'ssn':
                w, info = ssn(grad, bfgs, w, lr, lam)
                acc_time += info[0] # warm start time
                max_ave_search_len = max(max_ave_search_len, info[1])
                ave_inner_iter += info[2]
                max_inner_iter = max(max_inner_iter, info[2])

            else:
                raise NotImplementedError('optimizer {} is not implemented'.format(optimizer))

        # update w_k with small prob
        if np.random.random() < prob:
            w_hat = w_old.clone()
            center_grad = f_grad(w_hat, X, y, mu)
            
    if optimizer == 'ssn':
        ave_inner_iter /= n_epochs - 2 * L
        ssn_info = [acc_time, ave_inner_iter, max_inner_iter, max_ave_search_len, store_time] # add store_time
        return loss, w_sequence, ssn_info
    else: 
        ave_inner_iter /= n_epochs - 2 * L
        info = [ave_inner_iter, max_inner_iter]
        return loss, w_sequence, info