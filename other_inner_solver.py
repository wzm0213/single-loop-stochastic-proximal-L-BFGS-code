import numpy as np
import torch
from logistic_setup import prox_h
import math


# solving subproblem x^T B x + g^T x + h(x)
def fista_backtracking(calc_f, grad, calc_F, prox, Xinit, opts, terminate='error', backtracking=True):
    opts.setdefault('max_iter', 500)
    opts.setdefault('regul', 'l1')
    opts.setdefault('pos', False)
    opts.setdefault('tol', 1e-8)
    opts.setdefault('verbose', False)
    opts.setdefault('L0', 1)
    opts.setdefault('eta', 2)

    lam = opts['lambda']

    def calc_Q(x, y, L):
        return calc_f(y) + torch.dot((x - y).view(-1), grad(y).view(-1)) + (L / 2) * torch.norm(x - y, p=2) ** 2 + torch.sum(
            torch.abs(lam * x))

    x_old = Xinit.clone()
    y_old = Xinit.clone()
    t_old = 1
    iter = 0
    L = opts['L0']

    if terminate == 'x':
        while iter < opts['max_iter']: 
            iter += 1
            if backtracking:
                Lbar = L
                inner_iter = 0
                while inner_iter < 100:
                    inner_iter += 1
                    zk = prox(y_old - (1 / Lbar) * grad(y_old), lam / Lbar)
                    F = calc_F(zk)
                    Q = calc_Q(zk, y_old, Lbar)
                    if F <= Q:
                        break
                    Lbar = Lbar * opts['eta']
                L = Lbar
            x_new = prox(y_old - (1 / L) * grad(y_old), lam / L)
            t_new = 0.5 * (1 + math.sqrt(1 + 4 * t_old ** 2))
            y_new = x_new + (t_old - 1) / t_new * (x_new - x_old)

            e = torch.norm(x_new - x_old, p=1) / x_new.numel()
            if e < opts['tol']:
                break
            x_old, y_old, t_old = x_new.clone(), y_new.clone(), t_new
    elif terminate == 'loss':
        while iter < opts['max_iter']:  # back tracking
            iter += 1
            if backtracking:
                Lbar = L
                inner_iter = 0
                while inner_iter < 100:
                    inner_iter += 1
                    zk = prox(y_old - (1 / Lbar) * grad(y_old), lam / Lbar)
                    F = calc_F(zk)
                    Q = calc_Q(zk, y_old, Lbar)
                    if F <= Q:
                        break
                    Lbar = Lbar * opts['eta']
                L = Lbar
            x_new = prox(y_old - (1 / L) * grad(y_old), lam / L)
            t_new = 0.5 * (1 + math.sqrt(1 + 4 * t_old ** 2))
            y_new = x_new + (t_old - 1) / t_new * (x_new - x_old)

            e = torch.abs(calc_F(x_new) - calc_F(x_old))
            if e < opts['tol']:
                break
            x_old, y_old, t_old = x_new.clone(), y_new.clone(), t_new
    elif terminate == 'error':  # optimal condition
        def error(w):
            return w - prox(w - grad(w), lam)

        e0 = torch.norm(error(Xinit), p=2)
        while iter < opts['max_iter']:  # back tracking
            iter += 1
            if backtracking:
                Lbar = L
                inner_iter = 0
                while inner_iter < 100:
                    inner_iter += 1
                    zk = prox(y_old - (1 / Lbar) * grad(y_old), lam / Lbar)
                    F = calc_F(zk)
                    Q = calc_Q(zk, y_old, Lbar)
                    if F <= Q:
                        break
                    Lbar = Lbar * opts['eta']
                L = Lbar
            x_new = prox(y_old - (1 / L) * grad(y_old), lam / L)
            t_new = 0.5 * (1 + math.sqrt(1 + 4 * t_old ** 2))
            y_new = x_new + (t_old - 1) / t_new * (x_new - x_old)

            e = torch.norm(error(x_new), p=2) / e0
            if e < opts['tol']:
                break
            x_old, y_old, t_old = x_new.clone(), y_new.clone(), t_new
    else:
        raise ValueError(terminate)
    return x_new, iter, e


def ista_backtracking(calc_f, grad, calc_F, prox, Xinit, opts, terminate='x', backtracking=True):
    opts.setdefault('max_iter', 500)
    opts.setdefault('regul', 'l1')
    opts.setdefault('pos', False)
    opts.setdefault('tol', 1e-8)
    opts.setdefault('verbose', False)
    opts.setdefault('L0', 1)
    opts.setdefault('eta', 2)

    lam = opts['lambda']

    def calc_Q(x, y, L):
        return calc_f(y) + torch.dot((x - y).view(-1), grad(y).view(-1)) + (L / 2) * torch.norm(x - y, p=2) ** 2 + torch.sum(
            torch.abs(lam * x))

    x_old = Xinit.clone()
    y_old = Xinit.clone()
    iter = 0
    L = opts['L0']

    if terminate == 'x':
        while iter < opts['max_iter']: 
            iter += 1
            if backtracking:
                Lbar = L
                inner_iter = 0
                while inner_iter < 100:
                    inner_iter += 1
                    zk = prox(y_old - (1 / Lbar) * grad(y_old), lam / Lbar)
                    F = calc_F(zk)
                    Q = calc_Q(zk, y_old, Lbar)
                    if F <= Q:
                        break
                    Lbar = Lbar * opts['eta']
                L = Lbar
            x_new = prox(x_old - (1 / L) * grad(x_old), lam / L)

            e = torch.norm(x_new - x_old, p=1) / x_new.numel()
            if e < opts['tol']:
                break
            x_old = x_new.clone()
    elif terminate == 'loss':
        while iter < opts['max_iter']:  # back tracking
            iter += 1
            if backtracking:
                Lbar = L
                inner_iter = 0
                while inner_iter < 100:
                    inner_iter += 1
                    zk = prox(y_old - (1 / Lbar) * grad(y_old), lam / Lbar)
                    F = calc_F(zk)
                    Q = calc_Q(zk, y_old, Lbar)
                    if F <= Q:
                        break
                    Lbar = Lbar * opts['eta']
                L = Lbar
            x_new = prox(x_old - (1 / L) * grad(x_old), lam / L)

            e = torch.abs(calc_F(x_new) - calc_F(x_old))
            if e < opts['tol']:
                break
            x_old = x_new.clone()
    elif terminate == 'error':  # optimal condition
        def error(w):
            return w - prox(w - grad(w), lam)

        e0 = torch.norm(error(Xinit), p=2)
        while iter < opts['max_iter']:  # back tracking
            iter += 1
            if backtracking:
                Lbar = L
                inner_iter = 0
                while inner_iter < 100:
                    inner_iter += 1
                    zk = prox(y_old - (1 / Lbar) * grad(y_old), lam / Lbar)
                    F = calc_F(zk)
                    Q = calc_Q(zk, y_old, Lbar)
                    if F <= Q:
                        break
                    Lbar = Lbar * opts['eta']
                L = Lbar
            x_new = prox(x_old - (1 / L) * grad(x_old), lam / L)

            e = torch.norm(error(x_new), p=2) / e0
            if e < opts['tol']:
                break
            x_old = x_new.clone()
    else:
        raise ValueError(terminate)
    return x_new, iter, e


def scaled_prox_h(v, bfgs, lam=0.01, tol=1e-8, max_iter=1000, optimizer='fista'):  # use FISTA to solve the scaled proximal operator
    # only require bfgs.sigma, s, y, SS, SY, YY to compute U, J, A
    S = torch.from_numpy(np.array(bfgs.s).T)
    Y = torch.from_numpy(np.array(bfgs.y).T)
    bfgs.U = torch.cat((bfgs.sigma * S, Y), dim=1)
    S_S = torch.sum(bfgs.SS, dim=0)  # recall that bfgs.SS has shape (d, m, m)
    S_Y = torch.sum(bfgs.SY, dim=0)
    L_SY = torch.tril(S_Y, diagonal=-1)
    D = torch.diag(torch.diag(S_Y))
    bfgs.K_0 = torch.cat((torch.cat((bfgs.sigma * S_S, L_SY), dim=1),
                        torch.cat((L_SY.t(), -D), dim=1)), dim=0)


    if optimizer == 'fista':
        K_0_inv = torch.linalg.inv(bfgs.K_0)
        K_0_inv_SYT = K_0_inv @ bfgs.U.T
        kappa = bfgs.sigma

        def f(w):
            x = w - v
            z = K_0_inv_SYT @ x
            res = kappa * x - bfgs.U.dot(z)
            result = 0.5 * x @ res
            return result

        def grad(w):
            x = w - v
            z = K_0_inv_SYT @ x
            result = kappa * x - bfgs.U @ z
            return result

        def cal_F(w):
            x = w - v
            z = K_0_inv_SYT @ x
            res = kappa * x - bfgs.U @ z
            result = 0.5 * x @ res
            result += lam * torch.norm(w, p=1)
            return result

        opts = {'lambda': lam, 'tol': tol, 'max_iter': max_iter}
        x_init = 0.01 * torch.ones_like(v)
        x, iter, _ = fista_backtracking(f, grad, cal_F, prox_h, x_init, opts)

    elif optimizer == 'ista':
        K_0_inv = torch.linalg.inv(bfgs.K_0)
        K_0_inv_SYT = K_0_inv @ bfgs.U.T
        kappa = bfgs.sigma

        def f(w):
            x = w - v
            z = K_0_inv_SYT @ x
            res = kappa * x - bfgs.U @ z
            result = 0.5 * x @ res
            return result

        def grad(w):
            x = w - v
            z = K_0_inv_SYT @ x
            result = kappa * x - bfgs.U @ z
            return result

        def cal_F(w):
            x = w - v
            z = K_0_inv_SYT @ x
            res = kappa * x - bfgs.U @ z
            result = 0.5 * x @ res
            result += lam * torch.norm(w, p=1)
            return result

        opts = {'lambda': lam, 'tol': tol, 'max_iter': max_iter}
        x_init = 0.01 * torch.ones_like(v)
        x, iter, _ = ista_backtracking(f, grad, cal_F, prox_h, x_init, opts)
    
    else:
        raise ValueError('Unknown optimizer')
    
    if optimizer == 'fista' or optimizer == 'ista':
        return x, iter
    
    else:
        return x
