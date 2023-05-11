from typing import Tuple, Callable
import numpy as np


def demean_matrix(M: np.ndarray) -> np.ndarray:
    m, n = M.shape
    M1 = M.sum(0).repeat(m).reshape(n,m).T / m
    M2 = M.sum(1).repeat(n).reshape(m,n) / n
    M3 = np.sum(M) / (m*n)
    M_p = M - M1 - M2 + M3

    return M_p


def get_eigendecomposition(M:   np.ndarray,
                           eps: float     = 1e-12
                           ) -> Tuple[np.ndarray, np.ndarray]:
    
    L, U = np.linalg.eigh(M)
    L = L[::-1]
    U = U[:,::-1]
    L[L < eps] = 0

    return L, U


def get_inverse(M:    np.ndarray,
                func: Callable  = lambda x: x,
                eps:  float     = 1e-9) -> np.ndarray:
    L, U = get_eigendecomposition(M, eps)

    j = np.where(L > 0)[0][-1]
    M_inv = U[:,:j+1] @ np.diag(1 / func(L[:j+1])) @ U[:,:j+1].T

    return M_inv


def get_tail_sums(L: np.ndarray,
                  d: int       = None) -> np.ndarray:
    if d is None:
        d = len(L)

    L_sums = (L.sum() - L.cumsum())[:d]

    return L_sums


def get_kappa(kappa:  np.ndarray,
              K:      np.ndarray,
              demean: bool      = True) -> np.ndarray:
    if not demean:
        kappa_p = kappa

    else:
        m, n = K.shape
        n_new = kappa.shape[1]

        kappa_mean = np.tile(kappa.sum(0), (m, 1)) / m

        n_mean = np.tile(K.sum(1)[:, np.newaxis], n_new) / n

        nm_mean = np.tile(K.sum() / (n * m), (m, n_new))

        kappa_p = kappa - kappa_mean - n_mean + nm_mean

    return kappa_p


def to_column_vector(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        y = y[:, np.newaxis]

    if y.ndim == 2 and y.shape[0] == 1:
        y = y.T

    return y


def flip_dimensions(scores:     np.ndarray,
                    components: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flip = (scores.min(0) + scores.max(0)) / 2 < 0
    flip_matrix = np.diag(1 - 2 * flip)

    scores_flipped     = scores     @ flip_matrix
    components_flipped = components @ flip_matrix

    return scores_flipped, components_flipped


class IdentityScaler:

    @classmethod
    def fit_transform(cls, X: np.ndarray) -> np.ndarray:
        return X

    @classmethod
    def transform(cls, X: np.ndarray) -> np.ndarray:
        return X

