import torch
import numpy as np  # noqa: F401
from scipy.special import expit
from scipy.sparse import issparse


def f(w, X, y, mu=0.01):  # full f
    n = X.shape[0]
    z = X @ w
    if isinstance(z, np.ndarray):
        z = torch.from_numpy(z)
    sigmoid = torch.sigmoid(z)
    loss = -y * torch.log(sigmoid) - (1 - y) * torch.log1p(-sigmoid)
    result = torch.sum(loss) / n + mu / 2 * torch.norm(w, p=2)**2
    return result


def h(w, lam=0.01):  # L1-regularization
    return lam * torch.norm(w, p=1)


def F(w, X, y, lam=0.01, mu=0.01):  # total loss
    return f(w, X, y, mu) + h(w, lam)


def f_grad(w, X, y, mu=0.01):  # batch gradient
    n = X.shape[0]
    z = X @ w
    if isinstance(z, np.ndarray):
        z = torch.from_numpy(z)
    c = torch.sigmoid(z)

    # Calculate the gradient
    error = c - y
    if issparse(X):
        error = error.numpy()
        w = w.numpy()
        grad = X.T.dot(error) / n + mu * w
        grad = torch.from_numpy(grad)
    else:
        grad = X.T @ error / n + mu * w
    return grad


def f_hess_z(w, X, z, mu=0.01):  # batch hessian vector product
    n = X.shape[0]
    v = X @ w
    if isinstance(v, np.ndarray):
        v = torch.from_numpy(v)
    c = torch.sigmoid(v)
    sigmoid_derivative = c * (1 - c)
    X_dot_z = X @ z
    if issparse(X): # sparse matrix
        hess_product = X.multiply(X_dot_z[:, None]).multiply(sigmoid_derivative[:, None])
        hess = hess_product.sum(axis=0) / n
        hess = np.asarray(hess).reshape(-1)
        hess = torch.from_numpy(hess)
    else: # tensor
        hess_product = sigmoid_derivative[:, None] * X * X_dot_z[:, None]
        hess = hess_product.sum(axis=0) / n
    hess = hess + mu * z
    return hess

def f_hess(w, X, z, mu=0.01):  # only for test purpose, do not call
    n = X.shape[0]
    v = X.dot(w)
    c = expit(v)
    sigmoid_derivative = c * (1 - c)
    X_dot_z = X.dot(z)
    if isinstance(X, np.ndarray):
        hess_product = sigmoid_derivative[:, np.newaxis] * X * X_dot_z[:, np.newaxis]
        hess = hess_product.sum(axis=0)/n + mu * z
    else: # csr_sparse matrix
        hess_product = X.multiply(X_dot_z[:, np.newaxis]).multiply(sigmoid_derivative[:, np.newaxis]) # return a coo_matrix
        hess = hess_product.sum(axis=0)/n # return an array
        hess = np.asarray(hess).reshape(-1) + mu * z
    return hess


def prox_h(w, lam=0.01):  # proximal operator of lam * L1_norm
    return torch.sign(w) * torch.max(torch.abs(w) - lam, torch.zeros_like(w))


if __name__ == '__main__':
    from scipy.sparse import rand

    n = 1000
    d = 100
    X = rand(n, d, density=0.1, format='csr')
    y = torch.randint(0, 2, (n,)).float()
    w = torch.zeros(d)
    # test f_hess_z()
    z = torch.randn(d)
    hess = f_hess_z(w, X, z)
    w = w.numpy()
    z = z.numpy()
    hess_2 = f_hess(w, X, z)
    print(hess, hess_2) # correct