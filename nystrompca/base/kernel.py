from itertools import product
import numpy as np
from nystrompca.utils import demean_matrix


class Kernel:
    def __init__(self, kernel:    str   = 'rbf',
                       sigma:     float = 3,
                       degree:    int   = 2,
                       coef0:     float = 1,
                       normalize: bool  = True ):

        self.sigma        = sigma
        self.degree       = degree
        self.coef0        = coef0
        self.normalize    = normalize

        self.name = kernel

        bound: float

        if kernel == 'rbf':
            kernel_fcn = lambda x, y: rbf(x, y, degree, sigma)
            bound = 1

        elif kernel == 'poly':
            kernel_fcn = lambda x, y: poly(x, y, degree, coef0, normalize)
            if normalize:
                bound = 1
            else:
                bound = np.inf

        elif kernel == 'linear':
            kernel_fcn = lambda x, y: poly(x, y, 1, 0, normalize=False)
            bound = np.inf

        elif kernel == 'laplace':
            kernel_fcn = lambda x, y: rbf(x, y, 2, sigma)
            bound = 1

        elif kernel == 'cauchy':
            kernel_fcn = lambda x, y: cauchy(x, y, sigma)
            bound = 1

        elif kernel == 'cosine':
            kernel_fcn = lambda x, y: poly(x, y, 1, 0, normalize=True)
            bound = 1

        self.kernel_fcn = kernel_fcn

        self.bound = bound


    def __call__(self, x, y):
        return self.kernel_fcn(x, y)


    def get_name(self) -> str:
        return self.name


    def get_bound(self) -> float:
        return self.bound


    def calc_sigma(self, X: np.ndarray):
        assert X.ndim == 2, "Two-dimensional input data expected"

        n = X.shape[0]

        pairwise_diff = np.tile(X, (n,1)) - X.repeat(n, axis=0)
        sigma = np.median(np.sqrt((pairwise_diff**2).sum(1)))
        self.__init__(self.name, sigma, self.degree, 
                      self.coef0, self.normalize)


    def matrix(self, X1:     np.ndarray,
                     X2:     np.ndarray = None,
                     demean: bool       = True) -> np.ndarray:
        
        n = X1.shape[0]

        if X2 is None:
            K = np.zeros((n,n))
            for i in range(n):
                K[i, i] = self(X1[i], X1[i]) / 2
                for j in range(i+1, n):
                    K[i,j] = self(X1[i], X1[j])
            K = K + K.T

        else:
            m = X2.shape[0]
            K = np.zeros((n,m))
            for i, j in product(range(n), range(m)):
                K[i,j] = self(X1[i],X2[j])

        if demean:
            K = demean_matrix(K)

        return K


def rbf(x:      np.ndarray,
        y:      np.ndarray,
        degree: int,
        sigma:  float     ) -> float:
    return np.exp(-np.sqrt((x - y) @ (x - y))**degree / (sigma**2))


def poly(x:         np.ndarray,
         y:         np.ndarray,
         degree:    int,
         coef0:     float,
         normalize: bool      = True) -> float:
    value = (x @ y + coef0)**degree
    if normalize:
        value /= ((x @ x + coef0) * (y @ y + coef0))**(degree/2)

    return value


def cauchy(x:     np.ndarray,
           y:     np.ndarray,
           sigma: float     ) -> float:
    return 1 / (1 + (x - y) @ (x - y) / sigma**2)

