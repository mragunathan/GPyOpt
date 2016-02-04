from .base import AcquisitionBase
from ..util.general import get_quantiles

class AcquisitionEI(AcquisitionBase):
    """
    Class for Expected improvement acquisition functions.
    """
    def __init__(self, model, space, optimizer=None, cost = None, jitter=0.01):
        optimizer = optimizer
        super(AcquisitionEI, self).__init__(model, space, optimizer)
        self.jitter = jitter
        if cost == None: 
            self.cost = lambda x: 1
        else:
            self.cost = cost

    def acquisition_function(self,x):
        """
        Expected Improvement
        """
        m, s = self.model.predict(x)
        fmin = self.model.get_fmin()
        phi, Phi, _ = get_quantiles(self.jitter, fmin, m, s)    
        f_acqu = (fmin - m + self.jitter) * Phi + s * phi
        return -f_acqu/self.cost(x)  # note: returns negative value for posterior minimization 

    def acquisition_function_withGradients(self, x):
        """
        Derivative of the Expected Improvement (has a very easy derivative!)
        """
        m, s, dmdx, dsdx = self.model.predict_withGradients(x)
        fmin = self.model.get_fmin()
        phi, Phi, _ = get_quantiles(self.jitter, fmin, m, s)    
        f_acqu = (fmin - m + self.jitter) * Phi + s * phi        
        df_acqu = dsdx * phi - Phi * dmdx
        return -f_acqu/self.cost(x), -df_acqu/self.cost(x)
    
    