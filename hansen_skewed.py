import numpy as np
from scipy.special import gamma
from scipy.stats import t, uniform

class SkewStudent(object):
    def __init__(self, dof, skew):
        self.dof = dof
        self.skew = skew
        
    def __const_a(self):
        return 4*self.skew*self.__const_c()*(self.dof-2)/(self.dof-1)

    def __const_b(self):
        return (1 + 3*self.skew**2 - self.__const_a()**2)**.5

    def __const_c(self):
        return gamma((self.dof+1)/2)/((np.pi*(self.dof-2))**.5*gamma(self.dof/2))
            
    def ppf(self, arg):
        arg = np.atleast_1d(arg)
        a = self.__const_a()
        b = self.__const_b()
        cond = arg < (1-self.skew)/2
        ppf1 = t.ppf(arg / (1-self.skew), self.dof)
        ppf2 = t.ppf(.5 + (arg - (1-self.skew)/2) / (1+self.skew), self.dof)
        ppf = np.nan_to_num(ppf1)*cond+np.nan_to_num(ppf2)*np.logical_not(cond)
        ppf = (ppf*(1+np.sign(arg-(1-self.skew)/2)*self.skew)*(1-2/self.dof)**.5-a)/b
            
        if ppf.shape == (1, ):
            return float(ppf)
        else:
            return ppf
        
    def rvs(self, size=1):
        return self.ppf(uniform.rvs(size=size))