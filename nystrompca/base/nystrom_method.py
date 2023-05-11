import numpy as np


class NystromMethod:

    def __init__(self, m_subset: int = 10,
                       seed:     int = None,
                       **kwargs            ):

        super().__init__(**kwargs) 

        self.m = m_subset

        self.subset: np.ndarray = None

        self.seed = seed


    def create_subset(self, n: int) -> None:
        rng = np.random.default_rng(self.seed)

        subset = rng.choice(range(n), self.m, replace=False)

        self.subset = np.sort(subset) 
