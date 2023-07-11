import numpy as np
from abc import ABC, abstractmethod
from typing import Union
import logging
import sys


logger = logging.getLogger(__name__)


class FilterBase(ABC):

    @abstractmethod
    def __init__(
        self,
        name='FP filter'
    ) -> None:
        
        self.name = name

    @abstractmethod
    def __call__(self, x) -> float:
        pass


class IQRFilter(FilterBase):

    def __init__(
            self, 
            q1: Union[int, float] = 25,
            q2: Union[int, float] = 75,
            name='IQR filter'
    ) -> None:
        super().__init__(name)

        self.q1 = q1
        self.q2 = q2

        logger.info(f'{self.name}: \t initialized')


    def __call__(self, x) -> float:

        p1, p2 = np.percentile(x, [self.q1, self.q2])
        iqr = p2 - p1
        ub = p2 + 1.5 * iqr
        logger.debug(f'{self.name}:\t upper bound = {ub}')

        # filter out false matches due to missing detection boxes
        # import matplotlib
        # import matplotlib.pyplot as plt
        # matplotlib.use('agg')
        # plt.figure()
        # plt.hist(x.flatten(), bins=42)
        # plt.plot([ub, ub], plt.ylim())
        # plt.savefig('img.png')

        return ub
    

class GMMFilter(FilterBase):

    def __init__(
            self,
            n_components: int,
            std_coef: float = 3.0,
            name='GMM filter'
    ) -> None:
        super().__init__(name)

        self.n_components = n_components
        self.std_coef = std_coef
        
        logger.info(f'{self.name}: \t initialized')
    
    
    def __call__(self, x) -> float:

        assert len(x.shape) == 2 and x.shape[1] == 1, 'Invalid shape for x, expect (N, 1)'
        
        np.random.seed(42)
        from sklearn.mixture import GaussianMixture
        gmm_error_handled = False
        reg_covar = 1e-6
        while not gmm_error_handled:
            try:
                logger.debug(f'{self.name}:\t trying reg_covar = {reg_covar}')
                gm = GaussianMixture(n_components=self.n_components, covariance_type='diag', reg_covar=reg_covar).fit(x)
                gmm_error_handled = True
            except:
                logger.warning(f'{self.name}:\t reg_covar failed!')
                reg_covar *= 10
        smaller_component = np.argmin(gm.means_)                # type: ignore
        ub = gm.means_[smaller_component] + self.std_coef * np.sqrt(gm.covariances_[smaller_component])     # type: ignore
        logger.debug(f'{self.name}:\t smaller component has mean = {min(gm.means_)} and std = {np.sqrt(gm.covariances_[smaller_component])}')  # type: ignore
        logger.debug(f'{self.name}:\t upper bound = {ub}')

        # filter out false matches due to missing detection boxes
        # import matplotlib
        # import matplotlib.pyplot as plt
        # matplotlib.use('agg')
        # plt.figure()
        # plt.hist(x.flatten(), bins=42)
        # plt.plot([ub, ub], plt.ylim())
        # plt.savefig('img.png')

        return ub
