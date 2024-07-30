import torch
import numpy as np  # noqa: F401
from logistic_setup import prox_h

# auxiliary functions for direct method
def B_alpha_z(z, alpha, bfgs):
    '''calculate Hessian vector product B_{\alpha} z'''
    res = (bfgs.sigma - alpha) * z
    temp_0 = bfgs.U.T @ z
    temp = torch.linalg.solve(bfgs.J, temp_0)
    res -=  (bfgs.sigma - alpha) * bfgs.U @ temp
    return res


def B_alpha_inv_z(z, alpha, bfgs):
    '''calculate inverse Hessian vector product B_{\alpha}^{-1} z'''
    res =  z / (bfgs.sigma - alpha)
    temp_0 = bfgs.U.T @ z
    temp = torch.linalg.solve(bfgs.A, temp_0)
    res -=  bfgs.U @ temp / (bfgs.sigma - alpha)
    return res


def C_alpha_z(z, lamb, alpha, reg, bfgs):
    '''calculate C_{\alpha, j} z'''
    comparison = torch.abs(lamb) > reg
    index = torch.where(comparison)[0]
    res = (bfgs.sigma - alpha) * z
    if z.dim() == 1:
        res[index] *= alpha / bfgs.sigma
    elif z.dim() == 2:
        for j in range(z.shape[1]):
            res[:, j][index] *= alpha / bfgs.sigma
    else:
        raise ValueError('too large dimension of z')
    return res


def B_alpha_inv_D_inv_z(z, lamb, alpha, reg, bfgs):
    '''calculate (B_{\alpha}^{-1} + D_j)^{-1} z'''
    res = C_alpha_z(z, lamb, alpha, reg, bfgs)
    temp1 = bfgs.U.T @ res
    UCU = bfgs.U.T @ (C_alpha_z(bfgs.U, lamb, alpha, reg, bfgs)) # compute U.T C U
    inv = UCU - (bfgs.sigma - alpha) * bfgs.A # U.T C U - (\sigma_0 - \alpha) * A
    temp2 = torch.linalg.solve(inv, temp1)
    temp3 = bfgs.U @ temp2
    res -= C_alpha_z(temp3, lamb, alpha, reg, bfgs)
    return res


# for warm start
def Lambda(lamb, g, alpha, reg, bfgs):
    '''calculate the objective function value \Lambda of the dual problem'''
    temp1 = lamb - g
    temp2 = -lamb / alpha
    temp3 = torch.sign(temp2) * torch.max(torch.abs(temp2) - reg / alpha, torch.zeros_like(temp2))
    res1 = temp1 @ B_alpha_inv_z(temp1, alpha, bfgs) / 2 + lamb @ lamb / (2 * alpha)
    res2 = torch.norm(temp3 - temp2, p=2)**2 / 2 + reg / alpha * torch.norm(temp3, p=1)
    res = res1 - alpha * res2
    return res


def initialize_lamb(lamb, g, alpha, reg, bfgs, ratio=1e-2, max_iter=1000):
    '''initialize the dual variable lambda'''
    alpha_inv = 1 / bfgs.sigma
    for i in range(bfgs.k_size):
        alpha_inv += bfgs.s[i] @ bfgs.s[i] / (bfgs.y[i] @ bfgs.s[i])
    alpha_bar = 1 / alpha_inv
    L = 1 / (alpha_bar - alpha) + 1 / alpha

    obj_old = Lambda(lamb, g, alpha, reg, bfgs)
    grad_lamb = B_alpha_inv_z(lamb - g, alpha, bfgs) - prox_h(-lamb / alpha, reg / alpha)
    res = lamb - 1 / L * grad_lamb
    obj_new = Lambda(res, g, alpha, reg, bfgs)
    gap_0 = torch.abs(obj_new - obj_old)

    for iter in range(max_iter):
        obj_old = obj_new.clone()
        grad_lamb = B_alpha_inv_z(res - g, alpha, bfgs) - prox_h(-res / alpha, reg / alpha)
        res -= 1 / L * grad_lamb
        obj_new = Lambda(res, g, alpha, reg, bfgs)
        gap = torch.abs(obj_new - obj_old)
        if gap/gap_0 < ratio:
            break
    return res, iter
