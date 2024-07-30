# solving subproblem \ref{eq: sub_sim}
# min g^\top x + \frac{1}{2} x^\top B x + \theta(x)

import torch
import numpy as np  # noqa: F401
import time

from logistic_setup import prox_h
from ssn_auxiliary_fun import B_alpha_z, B_alpha_inv_z, B_alpha_inv_D_inv_z, initialize_lamb


def ssn(v, bfgs, x_current, eta, lam=0.01, tol=1e-10, max_iter=100):
    # choose alpha
    alpha_inv = 1 / bfgs.sigma
    for i in range(bfgs.k_size):
        alpha_inv += bfgs.s[i] @ bfgs.s[i] / (bfgs.y[i] @ bfgs.s[i])
    alpha = 0.5 / alpha_inv

    # t1 = time.time()

    # only require bfgs.sigma, s, y, SS, SY, YY to compute U, J, A
    S = torch.from_numpy(np.array(bfgs.s).T)
    Y = torch.from_numpy(np.array(bfgs.y).T)
    bfgs.U = torch.cat((bfgs.sigma * S, Y), dim=1)
    S_S = torch.sum(bfgs.SS, dim=0)  # recall that bfgs.SS has shape (d, m, m)
    S_Y = torch.sum(bfgs.SY, dim=0)
    Y_Y = torch.sum(bfgs.YY, dim=0)
    # S_S = S.T @ S  # pretty fast
    # S_Y = S.T @ Y
    # Y_Y = Y.T @ Y
    L_SY = torch.tril(S_Y, diagonal=-1)
    D = torch.diag(torch.diag(S_Y))
    bfgs.J = (bfgs.sigma - alpha) * torch.cat((torch.cat((bfgs.sigma * S_S, L_SY), dim=1),
                        torch.cat((L_SY.t(), -D), dim=1)), dim=0)
    block1 = bfgs.sigma * alpha * S_S
    block2 = bfgs.sigma * S_Y - (bfgs.sigma - alpha) * L_SY
    block3 = Y_Y + (bfgs.sigma - alpha) * D
    bfgs.A = torch.cat((torch.cat((block1, block2), dim=1),
                        torch.cat((block2.t(), block3), dim=1)), dim=0)
    
    # t2 = time.time()
    # print('time for computing U, J, A:', t2 - t1)

    # initialization
    g = eta * v - B_alpha_z(x_current, alpha, bfgs) - alpha * x_current
    reg = eta * lam
    lamb0 = 0.01 * torch.ones_like(g)
    start_time = time.time()
    lamb = initialize_lamb(lamb0, g, alpha, reg, bfgs, ratio=1e-2)[0]
    end_time = time.time()

    acc_time = end_time - start_time
    ave_search_length = 0

    # t3 = time.time()
    # print('time for initializing lambda:', t3 - t2)
    
    # inner loop
    for iter in range(max_iter):
        x = B_alpha_inv_z(lamb - g, alpha, bfgs)
        z = prox_h(-lamb / alpha, reg / alpha)
        grad_Lamb = x - z
        d = B_alpha_inv_D_inv_z(grad_Lamb, lamb, alpha, reg, bfgs)
        if torch.norm(d @ grad_Lamb, p=2) < tol:
            break
        rho_test = 1

        # need to compute \Lambda(lamb - rho * d) - \Lambda(lamb), refer to formula, store some terms
        B_alpha_inv_d = B_alpha_inv_z(d, alpha, bfgs)
        term1 = d @ B_alpha_inv_d / 2 + d @ d / (2 * alpha)  # need to scaled by rho^2
        term2 = - (lamb - g) @ B_alpha_inv_d - lamb @ d / alpha  # need to scaled by rho

        temp11 = - lamb / alpha
        temp12 = torch.sign(temp11) * torch.max(torch.abs(temp11) - reg / alpha, torch.zeros_like(temp11))
        Moreau1 = torch.norm(temp12 - temp11, p=2)**2 / 2 + reg / alpha * torch.norm(temp12, p=1)
        temp21 = - lamb / alpha
        temp22 = torch.sign(temp21) * torch.max(torch.abs(temp21) - reg / alpha, torch.zeros_like(temp21))
        Moreau2 = torch.norm(temp22 - temp21, p=2)**2 / 2 + reg / alpha * torch.norm(temp22, p=1)
        # line search
        for it in range(20):
            lamb_test = lamb - rho_test * d
            Lamb_gap = (rho_test ** 2) * term1 + rho_test * term2 - alpha * (Moreau1 - Moreau2)
            RHS = - 1e-4 * rho_test * d @ grad_Lamb

            if Lamb_gap < RHS:
                break
            else:
                rho_test *= 0.9
        lamb = lamb_test.clone()
        ave_search_length += it
    ave_search_length /= iter

    # t4 = time.time()
    # print('time for main loop:', t4 - t3)

    return x, [acc_time, ave_search_length, iter]


# testing the ssn_direct
if __name__ == "__main__":
    from bfgs_class import BFGS
    # f = 0
    d = 10000
    torch.manual_seed(0)
    x = torch.randn(d)
    v = torch.randn(d)
    x_current = torch.randn(d)
    bfgs = BFGS()
    for i in range(10):
        x_old = x.clone()
        x -= 0.1 * x
        bfgs.s.append((x - x_old).detach().numpy())
        bfgs.y.append((x - x_old).detach().numpy())
        bfgs.k_size += 1
    bfgs.sigma = bfgs.y[-1] @ bfgs.y[-1] / (bfgs.s[-1] @ bfgs.y[-1])
    # compute SS, SY, YY
    bfgs.SS = torch.zeros((d, bfgs.k_size, bfgs.k_size))
    bfgs.SY = torch.zeros((d, bfgs.k_size, bfgs.k_size))
    bfgs.YY = torch.zeros((d, bfgs.k_size, bfgs.k_size))
    S_T = np.array(bfgs.s)
    S_T = torch.from_numpy(S_T)
    Y_T = np.array(bfgs.y)
    Y_T = torch.from_numpy(Y_T)
    for i in range(d):
        bfgs.SS[i] = S_T[:, i].view(-1, 1) @ S_T[:, i].view(1, -1)
        bfgs.SY[i] = S_T[:, i].view(-1, 1) @ Y_T[:, i].view(1, -1)
        bfgs.YY[i] = Y_T[:, i].view(-1, 1) @ Y_T[:, i].view(1, -1)
    tic = time.time()
    output = ssn(v, bfgs, x_current, 0.1, tol=1e-10)
    toc = time.time()
    print('time:', toc - tic)
    print('output:', output[0])
