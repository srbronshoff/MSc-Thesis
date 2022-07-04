import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import warnings
import pandas as pd
import scipy.special as sp
from scipy.stats import norm
from hansen_skewed import SkewStudent
import statsmodels.api as sm
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.ar_model import AutoReg
from scipy import integrate
import math
warnings.filterwarnings("ignore")
plt.rcParams['figure.dpi']= 100
#############################
def diebold_mariano_table_generator(RW125, RW250, RW500, GN, Gskt, GEDF, FZ1F, FZ2F, FZG, FZhybrid):
    '''
    This function constructs the diebold-mariano table in the single-step ahead forecasting setting.

    Parameters
    ----------
    RW125 : array
        contains the loss contributions of the rolling window estimator with sample size 125.
    RW250 : array
        contains the loss contributions of the rolling window estimator with sample size 250.
    RW500 : array
        contains the loss contributions of the rolling window estimator with sample size 500.
    GN : array
        contains the loss contributions of the normal GARCH.
    Gskt : array
        contains the loss contributions of the skewed student T GARCH.
    GEDF : array
        contains the loss contributions of the EDF GARCH.
    FZ1F : array
        contains the loss contributions of the FZ GAS-1F.
    FZ2F : array
        contains the loss contributions of the FZ GAS-2F.
    FZG : array
        contains the loss contributions of the FZ GARCH.
    FZhybrid : array
        contains the loss contributions of the FZ hybrid.

    Returns
    -------
    df : dataframe
        contains the diebold-mariano table

    '''
    losses= [RW125, RW250, RW500, GN, Gskt, GEDF, FZ2F, FZ1F, FZG, FZhybrid]
    names= ['RW125', 'RW250', 'RW500', 'G-N', 'G-Skt', 'G-EDF', 'FZ-1F', 'FZ-2F', 'FZ-G', 'FZ-hybrid']
    df= pd.DataFrame(columns= names, index= names)
    for i in range(len(names)):
        temp= []
        collumn= losses[i]
        for j in range(len(names)):
            if i == j:
                temp.append(np.nan)
            else:
                row= losses[j]
                temp.append(diebold_mariano(row, collumn))
        df[names[i]]= temp
    return df

def diebold_mariano(y_loss, x_loss):
    '''
    calcutes the coefficients in the diebold-mariano table.

    Parameters
    ----------
    y_loss : array
        loss contributions of the row model.
    x_loss : array
        loss contributions of the collumn model.

    Returns
    -------
    double
        coefficient of a regression on a constant between the collumn - and row loss contributions.

    '''
    lags= int(math.ceil(4*(len(y_loss)/100)**(2/9)))
    loss_diff= y_loss-x_loss
    X= np.ones(len(loss_diff))
    est= sm.OLS(loss_diff, X).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    return est.tvalues[0]
    
def loss_contributer(returns, var, es, alpha):
    '''
    function calculates the loss contributions on every time point.

    Parameters
    ----------
    returns : array
        contains the return data.
    var : array
        contains the estimated value-at-risk on every time point.
    es : array
        contains the estimated expected shortfall on every time point.
    alpha : float
        quantile level.

    Returns
    -------
    loss_contributions : array
        contains the loss contributions at every time point.

    '''
    T= len(returns)
    loss_contributions= np.zeros(T)
    for i in range(T):
        loss_contributions[i]= FZ_loss_function(returns[i], var[i], es[i], alpha, 0, False)
    return loss_contributions

def goodness_of_fit(returns, var, es, alpha):
    '''
    function performs a goodness of fit test of the VaR and ES seperately using HAC SE.

    Parameters
    ----------
    returns : array
        contains the return data.
    var : array
        contains the estimated VaR.
    es : array
        contains the estimated ES.
    alpha : float
        quantile level.

    Returns
    -------
    GoF_VaR_pvalue : float
        p value of the VaR goodness of fit test.
    GoF_ES_pvalue : float
        p value of the ES goodness of fit test.

    '''
    T= len(returns)
    lags= int(math.ceil(4*(T/100)**(2/9)))
    aLambda_v= np.zeros(T)
    aLambda_e= np.zeros(T)
    for i in range(T):
        term= indicator_function(returns[i], var[i])
        aLambda_v[i]= term-alpha
        aLambda_e[i]= (1/alpha)*term*(returns[i]/es[i])-1
    y_v= aLambda_v[1:]
    x1_v= aLambda_v[:-1]
    x2_v= var[1:]
    y_e= aLambda_e[1:]
    x1_e= aLambda_e[:-1]
    x2_e= var[1:]
    data= {'x1_v':x1_v, 'x2_v':x2_v}
    X= pd.DataFrame(data)
    X= X[['x1_v', 'x2_v']]
    X= sm.add_constant(X)
    est= sm.OLS(y_v, X).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    GoF_VaR_pvalue= est.f_pvalue
    data= {'x1_e':x1_e, 'x2_e':x2_e}
    X= pd.DataFrame(data)
    X= X[['x1_e', 'x2_e']]
    X= sm.add_constant(X)
    est= sm.OLS(y_e, X).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    GoF_ES_pvalue= est.f_pvalue
    return GoF_VaR_pvalue, GoF_ES_pvalue

def ML_VaR_maker(aSigma2, alpha, dist, par):
    '''
    function estimates the VaR of a gaussian or skewed student T distribution.

    Parameters
    ----------
    aSigma2 : array
        contains the estimated volatility.
    alpha : float
        quantile level.
    dist : string
        gaussian- or skewed student T distribution.
    par : array
        contains the estimated parameters of the model.

    Returns
    -------
    aVaR : array
        contains the estimated VaR at every time point.

    '''
    T= len(aSigma2)
    aVaR= np.zeros(T)
    if dist == 'gaussian':
        a= norm.ppf(alpha)
        for i in range(T):
            aVaR[i]= np.sqrt(aSigma2[i])*a
    elif dist == 'skewed student': 
        dof= par[3]
        skew= par[4]
        a= cdf_value_skewed_student(dof, 1, skew, alpha)
        for i in range(T):
            aVaR[i]= np.sqrt(aSigma2[i])*a
    return aVaR

def cdf_value_skewed_student(v, h, xi, alpha):
    '''
    function estimates the VaR using on a grid search of the specified 
    skewed student T distribution.

    Parameters
    ----------
    v : float
        estimated degrees of freedom.
    h : float
        estimated variance.
    xi : float
        estimated skewness.
    alpha : float
        quantile level.

    Returns
    -------
    float
        estimated VaR.

    '''
    pdf_scaled, x= cdf_factor_skewed_student(v, h, xi)
    cdf= np.zeros(len(pdf_scaled))
    cdf[0]= pdf_scaled[0]
    for i in range(1, len(pdf_scaled)):
        cdf[i]= cdf[i-1]+pdf_scaled[i]
    index= np.where(cdf > alpha)[0][0]
    return x[index]

def cdf_factor_skewed_student(v, h, xi):
    '''
    function performes a grid search of a specified skewed student T distribution.

    Parameters
    ----------
    v : float
        estimated degrees of freedom.
    h : float
        estimated variance.
    xi : float
        estimated skewness.

    '''
    n= 10000
    test= np.zeros(n)
    x= np.linspace(-20, 20, n)
    for i in range(n):
        test[i]= skewedStudent_func(x[i], v, h, xi)
    total= np.sum(test)
    return (test/total), x

def skewedStudent_func(r_t, v, h_t, xi):
    '''
    skewed student T distribution.

    Parameters
    ----------
    r_t : float
        return.
    v : float
        estimated degrees of freedom.
    h_t : float
        variance.
    xi : float
        estimated skewness.

    Returns
    -------
    float
        PDF value.

    '''
    m= (sp.gamma((v - 1) / 2) / sp.gamma(v / 2)) * np.sqrt((v - 2) / np.pi) * (xi - (1 / xi))
    s= np.sqrt((xi**2 + 1 / (xi**2) - 1) - m**2)
    I_t= I_t_maker(s, r_t, h_t, m)
    term1= sp.gamma((v + 1) / 2) / sp.gamma(v / 2)
    term2= 1 / np.sqrt((v - 2) * np.pi * h_t)
    term3= 2 / (xi + (1 / xi))
    term4= (1 + ((((((s * r_t) / np.sqrt(h_t)) + m)**2) / (v - 2)) * (xi**(-2 * I_t))))**(-.5 * (v + 1))
    return term1 * term2 * s * term3 * term4

def skewedStudent_ES_func(x, v, h_t, xi):
    '''
    function used to estimate the ES for a skewed student T distribution.

    '''
    return x*skewedStudent_func(x, v, h_t, xi)
    
def I_t_maker(s, r, h, m):
    '''
    function calculates I_t of the skewed student T distribution.
    
    '''
    term= (s * r) / np.sqrt(h)
    if term >= 0:
        return 1
    else:
        return -1
    
def gaussian_func(x, sigma):
    '''
    function calculates gaussian pdf.
    '''
    return (1/(np.sqrt(2*np.pi*sigma**2)))*np.exp(-0.5 * (x/sigma)**2)

def gaussian_ES_func(x, sigma):
    '''
    function used to estimate the ES for a gaussian distribution.
    
    '''
    return x*gaussian_func(x, sigma)
 
def ML_ES_maker(aSigma2, aVaR, returns, alpha, dist, par):
    '''
    function estimates the ES for a gaussian- or skewed student distribution.

    Parameters
    ----------
    aSigma2 : array
        estimated variance.
    aVaR : array
        estimated VaR.
    returns : array
        return data.
    alpha : float
        quantile level.
    dist : string
        gaussian or skewed student T distribution.
    par : array
        contains the estiamted parameter values.

    Returns
    -------
    aES : array
        estimated ES.

    '''
    T= len(aSigma2)
    aES= np.zeros(T)
    if dist == 'gaussian':
        a= integrate.quad(gaussian_ES_func, -np.inf, norm.ppf(alpha), (1))[0]
        b= integrate.quad(gaussian_func, -np.inf, norm.ppf(alpha), (1))[0]
        c= a/b
        for i in range(T):
            aES[i]= c*np.sqrt(aSigma2[i])
    elif dist == 'skewed student':
        dof= par[3]
        skew= par[4]
        standard_var= cdf_value_skewed_student(dof, 1, skew, alpha)
        a= integrate.quad(skewedStudent_ES_func, -np.inf, standard_var, (dof, 1, skew))[0]
        b= integrate.quad(skewedStudent_func, -np.inf, standard_var, (dof, 1, skew))[0]
        c= a/b
        for i in range(T):
               aES[i]= c*np.sqrt(aSigma2[i])
    return aES

def ML_VaR_ES_getter(returns, dist, alpha, par):
    '''
    function obtains the VaR and ES estimates for a gaussian- or skewed student T GARCH model.

    Parameters
    ----------
    returns : array
        contains the return data.
    dist : string
        gaussian- or skewed student T distribution.
    alpha : float
        quantile level.
    par : array
        contains the estimated parameters.

    Returns
    -------
    aVaR : array
        contains the estimated VaR.
    aES : array
        contains the estimated ES.

    '''
    aSigma2= ML_sigma2_getter(par, returns)
    aVaR= ML_VaR_maker(aSigma2, alpha, dist, par)
    aES= ML_ES_maker(aSigma2, aVaR, returns, alpha, dist, par)
    return aVaR, aES

def ML_sigma2_getter(par, returns):
    '''
    function estimates the variance based on a (1, 1) GARCH model.

    Parameters
    ----------
    par : array
        contains the estimated parameters.
    returns : array
        return data.

    Returns
    -------
    aSigma2 : array
        contains the estimated variance.

    '''
    omega= par[0]
    beta= par[1]
    gamma= par[2]
    T= len(returns)
    aSigma2= np.zeros(T)
    aSigma2[0]= np.var(returns)
    for i in range(T-1):
        aSigma2[i+1]= omega+beta*aSigma2[i]+gamma*returns[i]**2
    return aSigma2

def logLL_gaussian_GARCH(par, returns):
    '''
    calculates the log likelihood of a gaussian GARCH model.

    Parameters
    ----------
    par : array
        contains parameter estimates.
    returns : array
        return data.

    Returns
    -------
    float
        log likelihood value of a gaussian GARCH model.

    '''
    omega= par[0]
    beta= par[1]
    gamma= par[2]
    h= np.zeros(len(returns))
    h[0]= np.var(returns)
    for i in range(len(returns) -1):
        h[i+1]= omega+beta*h[i]+gamma*returns[i]**2
    LL= np.zeros(len(h))
    for i in range(len(h)):
        LL[i]= (np.log(2*np.pi)+np.log(h[i])+returns[i]**2/h[i])
    return -(-.5*np.sum(LL)/len(h))

def ML_gaussian_GARCH_parameter_optimizer(returns):
    '''
    function optimizes a gaussian GARCH model using MLE.

    Parameters
    ----------
    returns : array
        return data.

    Returns
    -------
    array
        estimated parameters.

    '''
    par= opt.minimize(logLL_gaussian_GARCH, [0.6, 0.5, 0.1], (returns), 'SLSQP')
    if par.success == False:
        print('ML GAUSSIAN ERROR FOUND DURING OPTIMISATION')
    return par.x

def logLL_skewedStudent_GARCH(par, returns):
    '''
    calculates the log likelihood of a skewed student T GARCH model

    Parameters
    ----------
    par : array
        contains the estimated parameters.
    returns : array
        return data.

    Returns
    -------
    float
        log likelihood value of a skewed student T GARCH model.

    '''
    omega= par[0]
    beta= par[1]
    gamma= par[2]
    dof= par[3]
    xi= par[4]
    h= np.zeros(len(returns))
    h[0]= np.var(returns)
    for i in range(len(returns) -1):
        h[i+1]= omega+beta*h[i]+gamma*returns[i]**2
    m= sp.gamma((dof-1)/2)/sp.gamma(dof/2)*np.sqrt((dof-2)/np.pi)*(xi-1/xi)
    s= np.sqrt((xi**2+1/(xi**2)-1)-m**2)
    LL= np.zeros(len(h))
    for i in range(len(h)):
        if s*returns[i]/np.sqrt(h[i])+m >= 0:
            I_t= 1
        else:
            I_t= -1
        term0= sp.loggamma((dof+1)/2)-sp.loggamma(dof/2)-.5*np.log((dof-2)*np.pi*h[i])
        term1= np.log(s)+np.log(2/(xi+1/xi))
        term2= (-(dof+1)/2)*np.log(1+(s*returns[i]/np.sqrt(h[i])+m)**2/(dof-2)*(xi)**(-2*I_t))
        LL[i]= term0+term1+term2
    return -(np.sum(LL))/len(h)

def ML_skewedStudent_GARCH_parameter_optimizer(returns):
    '''
    function optimizes a skewed student T GARCH model using MLE.

    Parameters
    ----------
    returns : array
        return data.

    Returns
    -------
    array
        estimated parameter values.

    '''
    par= opt.minimize(logLL_skewedStudent_GARCH, [0.6, 0.4, 0.2, 8, 1], (returns), 'SLSQP')
    if par.success == False:
        print('ML SKEWED STUDENT ERROR FOUND DURING OPTIMISATION')
    return par.x

def sim_gaussian_GARCH(omega, beta, gamma, T):
    '''
    function simulates return observations according to a gaussian (1, 1) GARCH model.

    Parameters
    ----------
    omega : float
        omega parameter.
    beta : float
        beta parameter.
    gamma : float
        gamma parameter.
    T : float
        desired amount of time periods.

    Returns
    -------
    array
        simulated return data.
    array
        simulated variance data.

    '''
    fBurn= 100
    T= T+fBurn
    aSigmas= np.zeros(T)
    aReturns= np.zeros(T)
    aErrors= np.random.normal(0, 1, T)
    for i in range(T-1):
        aReturns[i]= np.sqrt(aSigmas[i])*aErrors[i]
        aSigmas[i+1]= omega+beta*aSigmas[i]+gamma*aReturns[i]**2
    aReturns[-1]= np.sqrt(aSigmas[-1])*aErrors[-1]
    return aReturns[100:T], aSigmas[100:T]

def sim_skewedStudent_GARCH(omega, beta, gamma, dof, xi, T):
    '''
    function simulates return observations according to a skewed student T (1, 1) GARCH model.

    Parameters
    ----------
    omega : float
        omega parameter.
    beta : float
        beta parameter.
    gamma : float
        gamma parameter.
    dof : float
        degrees of freedom parameter.
    xi : float
        skewness parameter.
    T : float
        desired time period.

    Returns
    -------
    array
        simulated return data.
    array
        simulated variance data.

    '''
    fBurn= 100
    T= T+fBurn
    aSigmas= np.zeros(T)
    aReturns= np.zeros(T)
    t= SkewStudent(dof, xi)
    aErrors= t.rvs(T)
    for i in range(T-1):
        aReturns[i]= np.sqrt(aSigmas[i])*aErrors[i]
        aSigmas[i+1]= omega+beta*aSigmas[i]+gamma*aReturns[i]**2
    aReturns[-1]= np.sqrt(aSigmas[-1])*aErrors[-1]
    return aReturns[100:T], aSigmas[100:T]

def data_simulator(distribution, par, T):
    '''
    function simulates gaussian or skewed student t distribution returns

    Parameters
    ----------
    distribution : string
        gaussian or skewed student T.
    par : array
        parameters.
    T : float
        desired time periods.

    Returns
    -------
    array, array
        simulated returns, simulated variances.

    '''
    omega, beta, gamma, dof, skew= par[0], par[1], par[2], par[3], par[4]
    if distribution == 'gaussian':
        return sim_gaussian_GARCH(omega, beta, gamma, T)
    elif distribution == 'skewed student':
        return sim_skewedStudent_GARCH(omega, beta, gamma, dof, skew, T)
    
def simulation_study():
    '''
    carries out the Monte Carlo simulation study. First data is simulated, and afterwards 
    the parameters are estimated. When finished, all information is saved to an excel file.

    '''
    number_of_simulations= 1000
    length_of_simulation= 2500
    alpha_list= [0.01, 0.025, 0.05, 0.10]
    omega, beta, gamma= 0.05, 0.9, 0.05
    dof, skew= 5, -0.5
    par= (omega, beta, gamma, dof, skew)
    writer= pd.ExcelWriter('FZ_GARCH_skewedStudent_2500.xlsx')
    for i in range(len(alpha_list)):
        a_list= []
        b_list= []
        beta_list= []
        gamma_list= []
        for j in range(number_of_simulations):
            print(i+1, j+1)
            aReturns, aSigmas= data_simulator('skewed student', par, length_of_simulation)
            FZ_est_par= FZ_GARCH_parameter_optimizer(aReturns, omega, alpha_list[i])
            a, b, beta, gamma= FZ_est_par[0], FZ_est_par[1], FZ_est_par[2], FZ_est_par[3]
            a_list.append(a)
            b_list.append(b)
            beta_list.append(beta)
            gamma_list.append(gamma)
        dict= {'a':a_list, 'b':b_list, 'beta':beta_list, 'gamma':gamma_list}
        df= pd.DataFrame(dict)
        df.to_excel(writer, sheet_name= f'{alpha_list[i]}', index= False)
    writer.save()
    
def RW_VaR(returns, length, alpha):
    '''
    function estimates the VaR of the rolling window estimator model. 

    Parameters
    ----------
    returns : array
        return data.
    length : foat
        length of the rolling window sample size.
    alpha : float
        quantile level.

    Returns
    -------
    aRW_VaR : array
        contains the estimated VaR for the rolling window estimator model.

    '''
    T= len(returns)
    aRW_VaR= np.zeros(T-length)
    for i in range(T-length):
        sample= returns[i:i+length-1]
        aRW_VaR[i]= np.quantile(sample, alpha)
    return aRW_VaR

def RW_ES(returns, length, alpha, aRW_VaR):
    '''
    function estimates the ES of the rolling window estimator model.

    Parameters
    ----------
    returns : array
        return data.
    length : float
        length of the rolling window sample size.
    alpha : float
        quantile level.
    aRW_VaR : array
        estimated VaR.

    Returns
    -------
    aRW_ES : array
        contains the estimated ES.

    '''
    T= len(returns)
    aRW_ES= np.zeros(T-length)
    for i in range(T-length):
        sample= returns[i:i+length-1]
        total= 0
        var_t= aRW_VaR[i]
        for j in range(len(sample)):
            if sample[j] <= var_t:
                total+= sample[j]
        aRW_ES[i]= (1/(alpha*length))*total
    return aRW_ES

def rolling_window_getter(returns, alpha, length):
    '''
    function obtains the estimates of the VaR and ES of the rolling window model.

    Parameters
    ----------
    returns : array
        return data.
    alpha : float
        quantile level.
    length : float
        length of the rolling window sample size.

    Returns
    -------
    var : array
        rolling window estimates of the VaR.
    es : TYPE
        rolling window estimates of the ES.

    '''
    var= RW_VaR(returns, length, alpha)
    es= RW_ES(returns, length, alpha, var)
    return var, es

def FZ_GARCH_ES_VaR_getter(par, returns, omega):
    '''
    function estimates the VaR and ES according to an FZ GARCH model

    Parameters
    ----------
    par : array
        contains the parameter estimates.
    returns : array
        return data.
    omega : float
        omega parameter.

    Returns
    -------
    lVaR : array
        contains the estimated VaR.
    lES : array
        contains the estimated ES.

    '''
    a= par[0]
    b= par[1]
    beta= par[2]
    gamma= par[3]
    T= len(returns)
    lSig= np.zeros(T)
    lVaR= np.zeros(T)
    lES= np.zeros(T)
    lSig[0]= np.var(returns)
    lVaR[0]= a*np.sqrt(lSig[0])
    lES[0]= b*np.sqrt(lSig[0])  
    for i in range(T-1):
        lSig[i+1]= omega+beta*lSig[i]+gamma*(returns[i]**2)
        lVaR[i+1]= a*np.sqrt(lSig[i+1])
        lES[i+1]= b*np.sqrt(lSig[i+1])
    return lVaR, lES

def FZ_GARCH_fun(par, returns, omega, alpha, tau, smooth):
    '''
    FZ GARCH function that is maximised in order to estimate the parameter values.

    Parameters
    ----------
    par : array
        parameters.
    returns : array
        return data.
    omega : float
        omega parameter.
    alpha : float
        quantile level.
    tau : float
        tau parameter.
    smooth : boolean
        lets the function know if the smoothed version of the loss function needs to be used.

    Returns
    -------
    float
        mean FZ loss function value.

    '''
    lVaR, lES= FZ_GARCH_ES_VaR_getter(par, returns, omega)
    T= len(returns)
    total= 0
    for i in range(T):
        total+= FZ_loss_function(returns[i], lVaR[i], lES[i], alpha, tau, smooth)
    return (1/T)*total

def FZ_GARCH_parameter_optimizer(returns, omega, alpha, initial_guess):
    '''
    function estimates the parameters of an FZ GARCH model

    Parameters
    ----------
    returns : array
        return data.
    omega : float
        omega parameter.
    alpha : float
        quantile level.
    initial_guess : aray
        initial guess for the parameter values.

    Returns
    -------
    array
        parameter estimates.

    '''
    theta_05= opt.minimize(FZ_GARCH_fun, initial_guess, (returns, omega, alpha, 5, True), 'SLSQP')
    if theta_05.success == False:
        # print('FZ GARCH ERROR FOUND DURING OPTIMISATION WITH THETA_05')
        theta_05.x= initial_guess
    theta_20= opt.minimize(FZ_GARCH_fun, theta_05.x, (returns, omega, alpha, 20, True), 'SLSQP')
    if theta_20.success == False:
        # print('FZ GARCH ERROR FOUND DURING OPTIMISATION WITH THETA_20')
        theta_20.x= theta_05.x
    theta_final= opt.minimize(FZ_GARCH_fun, theta_20.x, (returns, omega, alpha, 0, False), 'SLSQP')
    if theta_final.success == False:
        print('FZ GARCH ERROR FOUND DURING OPTIMISATION WITH THETA_FINAL')
    return theta_final.x

def FZ_GAS1F_ES_VAR_getter(par, returns, omega, alpha, tau, smooth):
    '''
    function estimates the VaR and ES according to an FZ GAS-1F model.

    Parameters
    ----------
    par : array
        parameter estimates.
    returns : array
        return data.
    omega : float
        omega parameter.
    alpha : float
        quantile level.
    tau : float
        tau parameter.
    smooth : float
        lets the function know if the smoothed version of the loss function needs to be used.

    Returns
    -------
    lVaR : array
        contains the estimated VaR.
    lES : array
        contains the estimated ES.

    '''
    a= par[0]
    b= par[1]
    beta= par[2]
    gamma= par[3]
    T= len(returns)
    lK= np.zeros(T)
    lVaR= np.zeros(T)
    lES= np.zeros(T)
    lK[0]= np.var(returns)
    lVaR[0]= a*np.exp(lK[0])
    lES[0]= b*np.exp(lK[0])  
    for i in range(T-1):
        if smooth == True:
            term1= capital_gamma_function(returns[i], lVaR[i], tau)
        else:
            term1= indicator_function(returns[i], lVaR[i])
        term2= (1/alpha)*term1*returns[i]-lES[i]
        lK[i+1]= omega+beta*lK[i]+gamma*(1/(lES[i]))*term2
        lVaR[i+1]= a*np.exp(lK[i+1])
        lES[i+1]= b*np.exp(lK[i+1])
    return lVaR, lES

def FZ_GAS1F_fun(par, returns, omega, alpha, tau, smooth):
    '''
    FZ GAS-1F function that is maximised in order to estimate the parameter values.

    Parameters
    ----------
    par : array
        parameters.
    returns : array
        return data.
    omega : float
        omega parameter.
    alpha : float
        quantile level.
    tau : float
        tau parameter.
    smooth : float
        lets the function know if the smoothed version of the loss function needs to be used..

    Returns
    -------
    float
        mean FZ loss function value.

    '''
    lVaR, lES= FZ_GAS1F_ES_VAR_getter(par, returns, omega, alpha, tau, smooth)
    T= len(returns)
    total= 0
    for i in range(T):
        total+= FZ_loss_function(returns[i], lVaR[i], lES[i], alpha, tau, smooth)
    return (1/T)*total

def FZ_GAS1F_parameter_optimizer(returns, omega, alpha, initial_guess):
    '''
    function estimates the parameters of an FZ GAS-1F model

    Parameters
    ----------
    returns : array
        return data.
    omega : float
        omega parameter.
    alpha : float
        quantile level.
    initial_guess : array
        initial guess of the parameters.

    Returns
    -------
    array
        parameter estimates.

    '''
    theta_05= opt.minimize(FZ_GAS1F_fun, initial_guess, (returns, omega, alpha, 5, True), 'SLSQP')
    if theta_05.success == False:
        # print('FZ GAS1F ERROR FOUND DURING OPTIMISATION WITH THETA_05')
        theta_05.x= initial_guess
    theta_20= opt.minimize(FZ_GAS1F_fun, theta_05.x, (returns, omega, alpha, 20, True), 'SLSQP')
    if theta_20.success == False:
        # print('FZ GAS1F ERROR FOUND DURING OPTIMISATION WITH THETA_20')
        theta_20.x= theta_05.x
    theta_final= opt.minimize(FZ_GAS1F_fun, theta_20.x, (returns, omega, alpha, 0, False), 'SLSQP')
    if theta_final.success == False:
        print('FZ GAS1F ERROR FOUND DURING OPTIMISATION WITH THETA_FINAL')
    return theta_final.x

def indicator_function(statement1, statement2):
    '''
    indicator function returning 1 if statement 2 is bigger (or equal) to statement 1.

    '''
    if statement1 <= statement2:
        return 1
    else:
        return 0
    
def capital_gamma_function(Y, v, tau):
    '''
    gamma function that is used when the smoothed version of the loss function is used.

    Parameters
    ----------
    Y : float
        return.
    v : float
        VaR.
    tau : float
        tau parameter.

    Returns
    -------
    float
        output of the gamma function.

    '''
    return 1/(1+np.exp(tau*(Y-v)))
    
def FZ_loss_function(Y, v, e, alpha, tau, smooth):
    '''
    calculates the FZ loss function value.

    Parameters
    ----------
    Y : float
        return.
    v : float
        VaR.
    e : float
        ES.
    alpha : float
        quantile level.
    tau : float
        tau parameter.
    smooth : boolean
        lets the function know if the smoothed version needs to be used.

    Returns
    -------
    float
        FZ loss function value.

    '''
    if smooth == True:
        term= capital_gamma_function(Y, v, tau)
    else:
        term= indicator_function(Y, v)
    return (-1/(alpha*e))*term*(v-Y)+(v/e)+np.log(-e)-1

def average_FZ_loss(returns, var, es, alpha):
    '''
    calculates the mean FZ loss function value

    Parameters
    ----------
    returns : array
        return data.
    var : array
        estimates for the VaR.
    es : array
        estimates for the ES.
    alpha : float
        quantile level.

    Returns
    -------
    float
        average FZ loss function value.

    '''
    total= 0
    T= len(returns)
    for i in range(T):
        total+= FZ_loss_function(returns[i], var[i], es[i], alpha, 0, False)
    return total/T

def FZ_GAS2F_ES_VAR_getter(par, returns, alpha, tau, smooth):
    '''
    function estimates the VaR and ES according to a FZ GAS-2F model

    Parameters
    ----------
    par : array
        parameter values.
    returns : array
        return data.
    alpha : float
        quantile level.
    tau : float
        tau parameter.
    smooth : boolean
        lets the function know if the smoothed version of the FZ loss function needs to be used.

    Returns
    -------
    aVaR : array
        contains the estimates of the VaR.
    aES : array
        contains the estimates of the ES.

    '''
    w_v= par[0]
    w_e= par[1]
    b_v= par[2]
    b_e= par[3]
    a_vv= par[4]
    a_ve= par[5]
    a_ev= par[6]
    a_ee= par[7]
    T= len(returns)
    aVaR= np.zeros(T)
    aES= np.zeros(T)
    aVaR[0], aES[0]= -1, -1
    for i in range(T-1):
        if smooth == True:
            term= capital_gamma_function(returns[i], aVaR[i], tau)
        else:
            term= indicator_function(returns[i], aVaR[i])
        lambda_vt= -aVaR[i]*(term-alpha)
        lambda_et= (1/alpha)*term*returns[i]-aES[i]
        aVaR[i+1]= w_v+b_v*aVaR[i]+a_vv*lambda_vt+a_ve*lambda_et
        aES[i+1]= w_e+b_e*aES[i]+a_ev*lambda_vt+a_ee*lambda_et
    return aVaR, aES

def FZ_GAS2F_fun(par, returns, alpha, tau, smooth):
    '''
    function that is used to estimate the parameter values of an FZ GAS-2F model.

    Parameters
    ----------
    par : array
        parameters.
    returns : array
        return data.
    alpha : float
        quantile level.
    tau : float
        tau parameter.
    smooth : boolean
        lets the function know if the smoothed version of the FZ loss function needs to be used.

    Returns
    -------
    float
        mean FZ loss function value.

    '''
    aVaR, aES= FZ_GAS2F_ES_VAR_getter(par, returns, alpha, tau, smooth)
    T= len(returns)
    total= 0
    for i in range(T):
        total+= FZ_loss_function(returns[i], aVaR[i], aES[i], alpha, tau, smooth)
    return (1/T)*total
    
def FZ_GAS2F_parameter_optimizer(returns, alpha, initial_guess):
    '''
    function estimates the parameter values of an FZ GAS-2F model.

    Parameters
    ----------
    returns : array
        return data.
    alpha : float
        quantile level.
    initial_guess : array
        initial guess of the parameter values.

    Returns
    -------
    array
        estimates of the parameter values.

    '''
    theta_05= opt.minimize(FZ_GAS2F_fun, initial_guess, (returns, alpha, 5, True), 'SLSQP')
    if theta_05.success == False:
        # print('FZ GAS2F ERROR FOUND DURING OPTIMISATION WITH THETA_05')
        theta_05.x= initial_guess    
    theta_20= opt.minimize(FZ_GAS2F_fun, theta_05.x, (returns, alpha, 20, True), 'SLSQP')
    if theta_20.success == False:
        # print('FZ GAS2F ERROR FOUND DURING OPTIMISATION WITH THETA_20')
        theta_20.x= theta_05.x
    theta_final= opt.minimize(FZ_GAS2F_fun, theta_20.x, (returns, alpha, 0, False), 'SLSQP')
    if theta_final.success == False:
        print('FZ GAS2F ERROR FOUND DURING OPTIMISATION WITH THETA_FINAL')
    return theta_final.x
    
def hybrid_VaR_ES_getter(par, returns, alpha, omega, tau, smooth):
    '''
    function obtains estimates for the VaR and ES according to an FZ hybrid model.

    Parameters
    ----------
    par : array
        parameters.
    returns : array
        return data.
    alpha : float
        quantile level.
    omega : float
        omega parameter.
    tau : float
        tau parameter.
    smooth : boolean
        lets the function know if a smoothed version of the FZ loss function needs to be used.

    Returns
    -------
    aVaR : array
        estimates of the VaR.
    aES : array
        estimates of the ES.

    '''
    a= par[0]
    b= par[1]
    beta= par[2]
    gamma= par[3]
    delta= par[4]
    T= len(returns)
    aVaR= np.zeros(T)
    aES= np.zeros(T)
    aK= np.zeros(T)
    aK[0]= np.var(returns)
    aVaR[0]= a*np.exp(aK[0])
    aES[0]= b*np.exp(aK[0])
    for i in range(T-1):
        if smooth == True:
            term1= capital_gamma_function(returns[i], aVaR[i], tau)
        else:
            term1= indicator_function(returns[i], aVaR[i])
        term2= (1/aES[i])*((1/alpha)*term1*returns[i]-aES[i])
        if np.abs(returns[i]) <= 0.0000001:
            term3= 0
        else:
            term3= np.log(np.abs(returns[i]))
        aK[i+1]= omega+beta*aK[i]+gamma*term2+delta*term3
        aVaR[i+1]= a*np.exp(aK[i+1])
        aES[i+1]= b*np.exp(aK[i+1])
    return aVaR, aES

def hybrid_fun(par, returns, omega, alpha, tau, smooth):
    '''
    function that is used to estimate the parameter values of an FZ hybrid model.

    Parameters
    ----------
    par : array
        parameters.
    returns : array
        return data.
    omega : float
        omega parameter.
    alpha : float
        quantile level.
    tau : float
        tau parameter.
    smooth : boolean
        lets the function know if a smoothed version of the FZ loss funciton needs to be used.

    Returns
    -------
    float
        average FZ loss function value.

    '''
    aVaR, aES= hybrid_VaR_ES_getter(par, returns, alpha, omega, tau, smooth)
    total= 0
    for i in range(len(returns)):
        temp= FZ_loss_function(returns[i], aVaR[i], aES[i], alpha, tau, smooth)
        total+= temp
    return total/len(returns)

def hybrid_parameter_optimizer(returns, alpha, omega, initial_guess):
    '''
    function estimates parameters of an FZ hybrid model .

    Parameters
    ----------
    returns : array
        return data.
    alpha : float
        quantile level.
    omega : float
        omega parameter.
    initial_guess : array
        initial guess of the parameters.

    Returns
    -------
    array
        estimates of the parameters.

    '''
    theta_05= opt.minimize(hybrid_fun, initial_guess, (returns, omega, alpha, 5, True), 'SLSQP')
    if theta_05.success == False:
        # print('FZ HYBRID ERROR FOUND DURING OPTIMISATION WITH THETA_05')
        theta_05.x= initial_guess
    theta_20= opt.minimize(hybrid_fun, theta_05.x, (returns, omega, alpha, 20, True), 'SLSQP')
    if theta_20.success == False:
        # print('FZ HYBRID ERROR FOUND DURING OPTIMISATION WITH THETA_20')
        theta_20.x= theta_05.x
    theta_final= opt.minimize(hybrid_fun, theta_20.x, (returns, omega, alpha, 0, False), 'SLSQP')
    if theta_final.success == False:
        print('FZ HYBRID ERROR FOUND DURING OPTIMISATION WITH THETA_FINAL')
    return theta_final.x

def RW_adder(df, alpha):
    '''
    adds the RW VaR and ES estimates to an existing dataframe

    Parameters
    ----------
    df : dataframe
        dataframe contain the returns.
    alpha : float
        quantile level.

    Returns
    -------
    df : datafarme
        same datadrame as before, but now with rolling window estimates of the VaR and ES.

    '''
    RW_length= 125
    RW_VaR, RW_ES= rolling_window_getter(df['returns'].tolist(), alpha, RW_length)
    RW_125_aVaR, RW_125_aES= [], []
    for i in range(RW_length):
        RW_125_aVaR.append(np.nan)
        RW_125_aES.append(np.nan)
    for i in range(len(RW_VaR)):
        RW_125_aVaR.append(RW_VaR[i])
        RW_125_aES.append(RW_ES[i])
    df['RW_125_VaR']= RW_125_aVaR
    df['RW_125_ES']= RW_125_aES
    RW_length= 250
    RW_VaR, RW_ES= rolling_window_getter(df['returns'].tolist(), alpha, RW_length)
    RW_250_aVaR, RW_250_aES= [], []
    for i in range(RW_length):
        RW_250_aVaR.append(np.nan)
        RW_250_aES.append(np.nan)
    for i in range(len(RW_VaR)):
        RW_250_aVaR.append(RW_VaR[i])
        RW_250_aES.append(RW_ES[i])
    df['RW_250_VaR']= RW_250_aVaR
    df['RW_250_ES']= RW_250_aES
    RW_length= 500
    RW_VaR, RW_ES= rolling_window_getter(df['returns'].tolist(), alpha, RW_length)
    RW_500_aVaR, RW_500_aES= [], []
    for i in range(RW_length):
        RW_500_aVaR.append(np.nan)
        RW_500_aES.append(np.nan)
    for i in range(len(RW_VaR)):
        RW_500_aVaR.append(RW_VaR[i])
        RW_500_aES.append(RW_ES[i])
    df['RW_500_VaR']= RW_500_aVaR
    df['RW_500_ES']= RW_500_aES
    return df

def data_splitter(df, startTrain, endTrain, startValid, endValid, alpha):
    '''
    splits the dataframe into an in-sample dataframe and out-of-sample dataframe
    '''
    dfTraining= df.loc[(df['date']>=startTrain)&(df['date']<=endTrain)]
    dfTraining= dfTraining.reset_index(drop = True, inplace = False)
    dfValidation= df.loc[(df['date']>startValid)&(df['date']<=endValid)]
    dfValidation= dfValidation.reset_index(drop = True, inplace = False)
    return dfTraining, dfValidation

def EDF_GARCH_VaR_ES_getter(returns, alpha, par):
    '''
    function estimates the VaR and ES according to an EDF GARCH model

    Parameters
    ----------
    returns : array
        return data.
    alpha : float
        quantile level.
    par : array
        parameters.

    Returns
    -------
    aVaR : array
        estimates of the VaR.
    aES : array
        estimates of the ES.

    '''
    T= len(returns)
    a= np.quantile(returns, alpha)
    total= 0
    counter= 0
    for i in range(T):
        if returns[i]< a:
            total+= returns[i]
            counter+= 1
    b= total/counter
    omega= par[0]
    beta= par[1]
    gamma= par[2]
    aSig= np.zeros(T)
    aVaR= np.zeros(T)
    aES= np.zeros(T)
    aSig[0]= np.var(returns)
    aVaR[0]= a*np.sqrt(aSig[0])
    aES[0]= b*np.sqrt(aSig[0])  
    for i in range(T-1):
        aSig[i+1]= omega+beta*aSig[i]+gamma*(returns[i]**2)
        aVaR[i+1]= a*np.sqrt(aSig[i+1])
        aES[i+1]= b*np.sqrt(aSig[i+1])
    return aVaR, aES
    
def inSample_parameter_estimator(dfTraining, alpha, asset_name, burn_in, time_frame, dictFZ_guesses):
    '''
    function obtains estimates for all models discussed in this thesis.

    Parameters
    ----------
    dfTraining : datafarme
        dataframe containg in-sample data.
    alpha : float
        quantile level.
    asset_name : string
        name of the asset (BTC, ETH, etc.).
    burn_in : float
        ignores the first few observations since the FZ models can give really large values for
        the first few observations.
    time_frame : string
        mentions the timeframe in the plots.
    dictFZ_guesses : dictionary
        dictionary which contains initial guesses for all the models discussed.

    Returns
    -------
    dict_parameters : dictionary
        dictionary which contains all parameter estimates of the models discussed.

    '''
    FZ_GAS1F_initial_guess= dictFZ_guesses['FZ_GAS1F_guess']
    FZ_GAS2F_initial_guess= dictFZ_guesses['FZ_GAS2F_guess']
    FZ_GARCH_initial_guess= dictFZ_guesses['FZ_GARCH_guess']
    FZ_hybrid_initial_guess= dictFZ_guesses['FZ_hybrid_guess']
    
    returns= dfTraining['returns'].tolist()
    
    FZ_GAS1F_est_par= FZ_GAS1F_parameter_optimizer(returns, 0, alpha, FZ_GAS1F_initial_guess)
    FZ_GAS1F_aVaR, FZ_GAS1F_aES= FZ_GAS1F_ES_VAR_getter(FZ_GAS1F_est_par, returns, 0, alpha, 0, False)
    
    FZ_GAS2F_est_par= FZ_GAS2F_parameter_optimizer(returns, alpha, FZ_GAS2F_initial_guess)
    # FZ_GAS2F_aVaR, FZ_GAS2F_aES= FZ_GAS2F_ES_VAR_getter(FZ_GAS2F_est_par, returns, alpha, 0, False)

    FZ_GARCH_est_par= FZ_GARCH_parameter_optimizer(returns, 1, alpha, FZ_GARCH_initial_guess)
    # FZ_GARCH_aVaR, FZ_GARCH_aES= FZ_GARCH_ES_VaR_getter(FZ_GARCH_est_par, returns, 1)
    
    FZ_hybrid_est_par= hybrid_parameter_optimizer(returns, alpha, 0, FZ_hybrid_initial_guess)
    # FZ_hybrid_aVaR, FZ_hybrid_aES= hybrid_VaR_ES_getter(FZ_hybrid_est_par, returns, alpha, 0, 0, False)
    
    ML_gauss_GARCH_est_par= ML_gaussian_GARCH_parameter_optimizer(returns)
    # ML_gauss_GARCH_aVaR, ML_gauss_GARCH_aES= ML_VaR_ES_getter(returns, 'gaussian', alpha, ML_gauss_GARCH_est_par)
    
    ML_skewT_GARCH_est_par= ML_skewedStudent_GARCH_parameter_optimizer(returns)
    # ML_skewT_GARCH_aVaR, ML_skewT_GARCH_aES= ML_VaR_ES_getter(returns, 'skewed student', alpha, ML_skewT_GARCH_est_par)
    
    EDF_GARCH_aVaR, EDF_GARCH_aES= EDF_GARCH_VaR_ES_getter(returns, alpha, ML_gauss_GARCH_est_par)
    
    start= dfTraining['date'].iloc[0]
    end= dfTraining['date'].iloc[-1]
    times= pd.date_range(start= start, end= end, periods= len(dfTraining))
    
    plt.figure(figsize= (15, 10))
    plt.title(f'{asset_name} Value-at-Risk In-Sample on {time_frame} time frame')
    plt.plot(times[burn_in+500:1000], FZ_GAS1F_aVaR[burn_in+500:1000], label= 'GAS1F')
    # plt.plot(times[burn_in:], FZ_GAS2F_aVaR[burn_in:], label= 'GAS2F')
    # plt.plot(times[burn_in:], FZ_GARCH_aVaR[burn_in:], label= 'FZ_GARCH')
    # plt.plot(times[burn_in:], FZ_hybrid_aVaR[burn_in:], label= 'hybrid')
    plt.plot(times[burn_in+500:1000], dfTraining['RW_125_VaR'][burn_in+500:1000], label= 'RW_125')
    plt.plot(times[burn_in+500:1000], EDF_GARCH_aVaR[burn_in+500:1000], label= 'EDF_GARCH')
    # plt.plot(times[burn_in:], ML_gauss_GARCH_aVaR[burn_in:], label= 'gauss_GARCH')
    # plt.plot(times[burn_in:], ML_skewT_GARCH_aVaR[burn_in:], label= 'skewT_GARCH')
    plt.ylabel('VaR')
    plt.xlabel('Time')
    plt.legend()
    plt.show()
    
    plt.figure(figsize= (15, 10))
    plt.title(f'{asset_name} Expected Shortfall In-Sample on {time_frame} time frame')
    plt.plot(times[burn_in+500:1000], FZ_GAS1F_aES[burn_in+500:1000], label= 'GAS1F')
    # plt.plot(times[burn_in:], FZ_GAS2F_aES[burn_in:], label= 'GAS2F')
    # plt.plot(times[burn_in:], FZ_GARCH_aES[burn_in:], label= 'GARCH')
    # plt.plot(times[burn_in:], FZ_hybrid_aES[burn_in:], label= 'hybrid')
    plt.plot(times[burn_in+500:1000], dfTraining['RW_125_ES'][burn_in+500:1000], label= 'RW_125')
    plt.plot(times[burn_in+500:1000], EDF_GARCH_aES[burn_in+500:1000], label= 'EDF_GARCH')
    # plt.plot(times[burn_in:], ML_gauss_GARCH_aES[burn_in:], label= 'gauss_GARCH')
    # plt.plot(times[burn_in:], ML_skewT_GARCH_aES[burn_in:], label= 'skewT_GARCH') 
    plt.xlabel('Time')
    plt.ylabel('ES')
    plt.legend()
    plt.show()
    
    dict_parameters= {'FZ_GAS1F_est_par': FZ_GAS1F_est_par,
                      'FZ_GAS2F_est_par': FZ_GAS2F_est_par,
                      'FZ_GARCH_est_par': FZ_GARCH_est_par,
                      'FZ_hybrid_est_par': FZ_hybrid_est_par,
                      'ML_gauss_est_par': ML_gauss_GARCH_est_par,
                      'ML_skewT_est_par': ML_skewT_GARCH_est_par}

    return dict_parameters

def outSample(dfValidation, dict_parameters, asset_name, burn_in, time_frame):
    '''
    first obtains the VaR and ES estimates for all models discussed. secondly calculates the
    average losses of all models and performs the goodness of fit tests. lastly, the loss contributions
    are calculated and the diebold-mariano table is constructed. note that this is for the single-step
    ahead setting.

    Parameters
    ----------
    dfValidation : dataframe
        dataframe containing the out-of-sample data.
    dict_parameters : dictionary
        contains parameter estimates of all models discussed.
    asset_name : string
        name of the asset (BTC, ETH, etc.).
    burn_in : float
        burn in to delete the first few observations.
    time_frame : string
        mentions that time frame in the plots.

    Returns
    -------
    diebold_mariano_table : dataframe
        contains the diebold-mariano table.

    '''
    FZ_GAS1F_est_par= dict_parameters['FZ_GAS1F_est_par']
    FZ_GAS2F_est_par= dict_parameters['FZ_GAS2F_est_par']
    FZ_GARCH_est_par= dict_parameters['FZ_GARCH_est_par']
    FZ_hybrid_est_par= dict_parameters['FZ_hybrid_est_par']
    ML_gauss_GARCH_est_par= dict_parameters['ML_gauss_est_par']
    ML_skewT_GARCH_est_par= dict_parameters['ML_skewT_est_par']
    
    returns= dfValidation['returns'].tolist()
    
    # obtain the VaR and ES of every model for the validation data
    FZ_GAS1F_aVaR, FZ_GAS1F_aES= FZ_GAS1F_ES_VAR_getter(FZ_GAS1F_est_par, returns, 0, alpha, 0, False)
    FZ_GAS1F_aVaR, FZ_GAS1F_aES= FZ_GAS1F_aVaR[burn_in:], FZ_GAS1F_aES[burn_in:]

    FZ_GAS2F_aVaR, FZ_GAS2F_aES= FZ_GAS2F_ES_VAR_getter(FZ_GAS2F_est_par, returns, alpha, 0, False)
    FZ_GAS2F_aVaR, FZ_GAS2F_aES= FZ_GAS2F_aVaR[burn_in:], FZ_GAS2F_aES[burn_in:]

    FZ_GARCH_aVaR, FZ_GARCH_aES= FZ_GARCH_ES_VaR_getter(FZ_GARCH_est_par, returns, 1)
    FZ_GARCH_aVaR, FZ_GARCH_aES= FZ_GARCH_aVaR[burn_in:], FZ_GARCH_aES[burn_in:]

    FZ_hybrid_aVaR, FZ_hybrid_aES= hybrid_VaR_ES_getter(FZ_hybrid_est_par, returns, alpha, 0, 0, False)
    FZ_hybrid_aVaR, FZ_hybrid_aES= FZ_hybrid_aVaR[burn_in:], FZ_hybrid_aES[burn_in:]
    
    ML_gaus_GARCH_aVaR, ML_gaus_GARCH_aES= ML_VaR_ES_getter(returns, 'gaussian', alpha, ML_gauss_GARCH_est_par)
    ML_gaus_GARCH_aVaR, ML_gaus_GARCH_aES= ML_gaus_GARCH_aVaR[burn_in:], ML_gaus_GARCH_aES[burn_in:]

    ML_skewT_GARCH_aVaR, ML_skewT_GARCH_aES= ML_VaR_ES_getter(returns, 'skewed student', alpha, ML_skewT_GARCH_est_par)
    ML_skewT_GARCH_aVaR, ML_skewT_GARCH_aES= ML_skewT_GARCH_aVaR[burn_in:], ML_skewT_GARCH_aES[burn_in:]
    
    EDF_GARCH_aVaR, EDF_GARCH_aES= EDF_GARCH_VaR_ES_getter(returns, alpha, ML_gauss_GARCH_est_par)
    EDF_GARCH_aVaR, EDF_GARCH_aES= EDF_GARCH_aVaR[burn_in:], EDF_GARCH_aES[burn_in:]
    
    RW_125_aVaR, RW_125_aES= dfValidation['RW_125_VaR'].tolist(), dfValidation['RW_125_ES'].tolist()
    RW_125_aVaR, RW_125_aES=  RW_125_aVaR[burn_in:], RW_125_aES[burn_in:]
    RW_250_aVaR, RW_250_aES= dfValidation['RW_250_VaR'].tolist(), dfValidation['RW_250_ES'].tolist()
    RW_250_aVaR, RW_250_aES= RW_250_aVaR[burn_in:], RW_250_aES[burn_in:]
    RW_500_aVaR, RW_500_aES= dfValidation['RW_500_VaR'].tolist(), dfValidation['RW_500_ES'].tolist()
    RW_500_aVaR, RW_500_aES= RW_500_aVaR[burn_in:], RW_500_aES[burn_in:]
    
    # obtain the average losses of every model for the validation data
    returns= returns[burn_in:]
    
    FZ_GAS1F_avg_loss= average_FZ_loss(returns, FZ_GAS1F_aVaR, FZ_GAS1F_aES, alpha)
    FZ_GAS2F_avg_loss= average_FZ_loss(returns, FZ_GAS2F_aVaR, FZ_GAS2F_aES, alpha)
    FZ_GARCH_avg_loss= average_FZ_loss(returns, FZ_GARCH_aVaR, FZ_GARCH_aES, alpha)
    FZ_hybrid_avg_loss= average_FZ_loss(returns, FZ_hybrid_aVaR, FZ_hybrid_aES, alpha)
    
    ML_gauss_avg_loss= average_FZ_loss(returns, ML_gaus_GARCH_aVaR, ML_gaus_GARCH_aES, alpha)
    ML_skewT_avg_loss= average_FZ_loss(returns, ML_skewT_GARCH_aVaR, ML_skewT_GARCH_aES, alpha)
    
    EDF_GARCH_avg_loss= average_FZ_loss(returns, EDF_GARCH_aVaR, EDF_GARCH_aES, alpha)
    
    RW_125_avg_loss= average_FZ_loss(returns, RW_125_aVaR, RW_125_aES, alpha)
    RW_250_avg_loss= average_FZ_loss(returns, RW_250_aVaR, RW_250_aES, alpha)
    RW_500_avg_loss= average_FZ_loss(returns, RW_500_aVaR, RW_500_aES, alpha)
    
    rounding_decimals= 3
    
    print('FZ GAS1F average loss:', round(FZ_GAS1F_avg_loss, rounding_decimals))
    print('FZ GAS2F average loss:', round(FZ_GAS2F_avg_loss, rounding_decimals))
    print('FZ GARCH average loss:', round(FZ_GARCH_avg_loss, rounding_decimals))
    print('FZ hybrid average loss:', round(FZ_hybrid_avg_loss, rounding_decimals))
    
    print('ML gaussian GARCH average loss:', round(ML_gauss_avg_loss, rounding_decimals))
    print('ML skewed student GARCH average loss:', round(ML_skewT_avg_loss, rounding_decimals))
    
    print('EDF GARCH average loss:', round(EDF_GARCH_avg_loss, rounding_decimals))
    
    print('RW_125 average loss:', round(RW_125_avg_loss, rounding_decimals))
    print('RW_250 average loss:', round(RW_250_avg_loss, rounding_decimals))
    print('RW_500 average loss:', round(RW_500_avg_loss, rounding_decimals))
    
    # plot the out-of-sample VaR and ES for every model
    start= dfValidation['date'].iloc[0]
    end= dfValidation['date'].iloc[-1]
    times= pd.date_range(start= start, end= end, periods= len(dfValidation))
    
    plt.figure(figsize= (20, 10))
    plt.title(f'{asset_name} Value-at-Risk Out-of-Sample on {time_frame} time frame')
    plt.plot(times[burn_in:], FZ_GAS1F_aVaR, label= 'GAS1F')
    # plt.plot(times[burn_in:], FZ_GAS2F_aVaR, label= 'GAS2F')
    # plt.plot(times[burn_in:], FZ_GARCH_aVaR, label= 'GARCH')
    # plt.plot(times[burn_in:], FZ_hybrid_aVaR, label= 'hybrid')
    plt.plot(times[burn_in:], RW_125_aVaR, label= 'RW_125')
    # plt.plot(times[burn_in:], ML_gaus_GARCH_aVaR, label= 'gauss_GARCH')
    plt.plot(times[burn_in:], ML_skewT_GARCH_aVaR, label= 'skewT_GARCH')
    plt.plot(times[burn_in:], EDF_GARCH_aVaR, label= 'EDF_GARCH')
    plt.legend()
    plt.show()
    
    plt.figure(figsize= (20, 10))
    plt.title(f'{asset_name} Expected Shortfall Out-of-Sample {time_frame} time frame')
    plt.plot(times[burn_in:], FZ_GAS1F_aES, label= 'GAS1F')
    # plt.plot(times[burn_in:], FZ_GAS2F_aES, label= 'GAS2F')
    # plt.plot(times[burn_in:], FZ_GARCH_aES, label= 'GARCH')
    # plt.plot(times[burn_in:], FZ_hybrid_aES, label= 'hybrid')
    plt.plot(times[burn_in:], RW_125_aES, label= 'RW_125')
    # plt.plot(times[burn_in:], ML_gaus_GARCH_aES, label= 'gauss_GARCH')
    plt.plot(times[burn_in:], ML_skewT_GARCH_aES, label= 'skewT_GARCH')
    plt.plot(times[burn_in:], EDF_GARCH_aES, label= 'EDF_GARCH')
    plt.legend()
    plt.show()
    
    # calculate the goodness of fit test for the VaR and ES
    GoF_FZ_GAS1F_VaR_pvalue, GoF_FZ_GAS1F_ES_pvalue= goodness_of_fit(returns, FZ_GAS1F_aVaR, FZ_GAS1F_aES, alpha)
    GoF_FZ_GAS2F_VaR_pvalue, GoF_FZ_GAS2F_ES_pvalue= goodness_of_fit(returns, FZ_GAS2F_aVaR, FZ_GAS2F_aES, alpha)
    GoF_FZ_GARCH_VaR_pvalue, GoF_FZ_GARCH_ES_pvalue= goodness_of_fit(returns, FZ_GARCH_aVaR, FZ_GARCH_aES, alpha)
    GoF_FZ_hybrid_VaR_pvalue, GoF_FZ_hybrid_ES_pvalue= goodness_of_fit(returns, FZ_hybrid_aVaR, FZ_hybrid_aES, alpha)
    
    GoF_ML_gaus_GARCH_VaR_pvalue, GoF_ML_gaus_GARCH_ES_pvalue= goodness_of_fit(returns,ML_gaus_GARCH_aVaR,
                                                                               ML_gaus_GARCH_aES, alpha)
    GoF_ML_skewT_GARCH_VaR_pvalue, GoF_ML_skewT_GARCH_ES_pvalue= goodness_of_fit(returns, ML_skewT_GARCH_aVaR,
                                                                                 ML_skewT_GARCH_aES, alpha)
    
    GoF_EDF_GARCH_VaR_pvalue, GoF_EDF_GARCH_ES_pvalue= goodness_of_fit(returns, EDF_GARCH_aVaR, EDF_GARCH_aES, alpha)
    
    GoF_RW_125_VaR_pvalue, GoF_RW_125_ES_pvalue= goodness_of_fit(returns, RW_125_aVaR, RW_125_aES, alpha)
    GoF_RW_250_VaR_pvalue, GoF_RW_250_ES_pvalue= goodness_of_fit(returns, RW_250_aVaR, RW_250_aES, alpha)
    GoF_RW_500_VaR_pvalue, GoF_RW_500_ES_pvalue= goodness_of_fit(returns, RW_500_aVaR, RW_500_aES, alpha)
    
    print('FZ GAS1F GoF VaR:', round(GoF_FZ_GAS1F_VaR_pvalue, rounding_decimals))
    print('FZ GAS1F GoF ES:', round(GoF_FZ_GAS1F_ES_pvalue, rounding_decimals))
    
    print('FZ GAS2F GoF VaR:', round(GoF_FZ_GAS2F_VaR_pvalue, rounding_decimals))
    print('FZ GAS2F GoF ES:', round(GoF_FZ_GAS2F_ES_pvalue, rounding_decimals))
    
    print('FZ GARCH GoF VaR:', round(GoF_FZ_GARCH_VaR_pvalue, rounding_decimals))
    print('FZ GARCH GoF ES:', round(GoF_FZ_GARCH_ES_pvalue, rounding_decimals))
    
    print('FZ hybrid GoF VaR:', round(GoF_FZ_hybrid_VaR_pvalue, rounding_decimals))
    print('FZ hybrid GoF ES:', round(GoF_FZ_hybrid_ES_pvalue, rounding_decimals))
    
    print('ML gaussian GARCH GoF VaR:', round(GoF_ML_gaus_GARCH_VaR_pvalue, rounding_decimals))
    print('ML gaussian GARCH GoF ES:', round(GoF_ML_gaus_GARCH_ES_pvalue, rounding_decimals))
    
    print('ML skewed student GARCH GoF VaR:', round(GoF_ML_skewT_GARCH_VaR_pvalue, rounding_decimals))
    print('ML skewed student GARCH GoF ES:', round(GoF_ML_skewT_GARCH_ES_pvalue, rounding_decimals))
    
    print('EDF GARCH GoF VaR:', round(GoF_EDF_GARCH_VaR_pvalue, rounding_decimals))
    print('EDF GARCH GoF ES:', round(GoF_EDF_GARCH_ES_pvalue, rounding_decimals))
    
    print('RW_125 GoF VaR:', round(GoF_RW_125_VaR_pvalue, rounding_decimals))
    print('RW_125 GoF ES:', round(GoF_RW_125_ES_pvalue, rounding_decimals))
    
    print('RW_250 GoF VaR:', round(GoF_RW_250_VaR_pvalue, rounding_decimals))
    print('RW_250 GoF ES:', round(GoF_RW_250_ES_pvalue, rounding_decimals))
    
    print('RW_500 GoF VaR:', round(GoF_RW_500_VaR_pvalue, rounding_decimals))
    print('RW_500 GoF ES:', round(GoF_RW_500_ES_pvalue, rounding_decimals))
    
    # calculate the DM table
    FZ_GAS1F_loss_contributer= loss_contributer(returns, FZ_GAS1F_aVaR, FZ_GAS1F_aES, alpha)
    FZ_GAS2F_loss_contributer= loss_contributer(returns, FZ_GAS2F_aVaR, FZ_GAS2F_aES, alpha)
    FZ_GARCH_loss_contributer= loss_contributer(returns, FZ_GARCH_aVaR, FZ_GARCH_aES, alpha)
    FZ_hybrid_loss_contributer= loss_contributer(returns, FZ_hybrid_aVaR, FZ_hybrid_aES, alpha)
    
    ML_gaus_GARCH_loss_contributer= loss_contributer(returns, ML_gaus_GARCH_aVaR, ML_gaus_GARCH_aES, alpha)
    ML_skewT_GARCH_loss_contributer= loss_contributer(returns, ML_skewT_GARCH_aVaR, ML_skewT_GARCH_aES, alpha)
    
    EDF_GARCH_loss_contributer= loss_contributer(returns, EDF_GARCH_aVaR, EDF_GARCH_aES, alpha)
    
    RW_125_loss_contributer= loss_contributer(returns, RW_125_aVaR, RW_125_aES, alpha)
    RW_250_loss_contributer= loss_contributer(returns, RW_250_aVaR, RW_250_aES, alpha)
    RW_500_loss_contributer= loss_contributer(returns, RW_500_aVaR, RW_500_aES, alpha)
    
    diebold_mariano_table= diebold_mariano_table_generator(RW_125_loss_contributer, RW_250_loss_contributer,
                                                           RW_500_loss_contributer, ML_gaus_GARCH_loss_contributer,
                                                           ML_skewT_GARCH_loss_contributer, EDF_GARCH_loss_contributer,
                                                           FZ_GAS1F_loss_contributer, FZ_GAS2F_loss_contributer, 
                                                           FZ_GARCH_loss_contributer, FZ_hybrid_loss_contributer)
    print('#############################')
    return diebold_mariano_table

def extension(dfValidation, dict_parameters, asset_name, burn_in, seperate):
    '''
    first obtains the VaR and ES estimates for all models discussed. secondly calculates the
    average losses of all models and performs the goodness of fit tests. lastly, the loss contributions
    are calculated and the diebold-mariano table is constructed. note that this is for the multi-step
    ahead setting.

    Parameters
    ----------
    dfValidation : dataframe
        contains the out-of-sample data.
    dict_parameters : dictionary
        contains the parameter estimates of all models discussed.
    asset_name : string
        name of the asset.
    burn_in : float
        burn in to delete the first few observations.
    seperate : float
        this is to make sure that we get 8-step ahead predictions

    Returns
    -------
    array
        contains the loss contributions of every model.

    '''
    FZ_GAS1F_est_par= dict_parameters['FZ_GAS1F_est_par']
    FZ_GAS2F_est_par= dict_parameters['FZ_GAS2F_est_par']
    FZ_GARCH_est_par= dict_parameters['FZ_GARCH_est_par']
    FZ_hybrid_est_par= dict_parameters['FZ_hybrid_est_par']
    ML_gauss_GARCH_est_par= dict_parameters['ML_gauss_est_par']
    ML_skewT_GARCH_est_par= dict_parameters['ML_skewT_est_par']
    
    returns= dfValidation['returns'].tolist()
    
    # obtain the VaR and ES of every model for the validation data
    FZ_GAS1F_aVaR, FZ_GAS1F_aES= FZ_GAS1F_ES_VAR_getter(FZ_GAS1F_est_par, returns, 0, alpha, 0, False)
    FZ_GAS1F_aVaR, FZ_GAS1F_aES= FZ_GAS1F_aVaR[burn_in:], FZ_GAS1F_aES[burn_in:]
    FZ_GAS1F_aVaR, FZ_GAS1F_aES= FZ_GAS1F_aVaR[0::seperate], FZ_GAS1F_aES[0::seperate]

    FZ_GAS2F_aVaR, FZ_GAS2F_aES= FZ_GAS2F_ES_VAR_getter(FZ_GAS2F_est_par, returns, alpha, 0, False)
    FZ_GAS2F_aVaR, FZ_GAS2F_aES= FZ_GAS2F_aVaR[burn_in:], FZ_GAS2F_aES[burn_in:]
    FZ_GAS2F_aVaR, FZ_GAS2F_aES= FZ_GAS2F_aVaR[0::seperate], FZ_GAS2F_aES[0::seperate]

    FZ_GARCH_aVaR, FZ_GARCH_aES= FZ_GARCH_ES_VaR_getter(FZ_GARCH_est_par, returns, 1)
    FZ_GARCH_aVaR, FZ_GARCH_aES= FZ_GARCH_aVaR[burn_in:], FZ_GARCH_aES[burn_in:]
    FZ_GARCH_aVaR, FZ_GARCH_aES= FZ_GARCH_aVaR[0::seperate], FZ_GARCH_aES[0::seperate]

    FZ_hybrid_aVaR, FZ_hybrid_aES= hybrid_VaR_ES_getter(FZ_hybrid_est_par, returns, alpha, 0, 0, False)
    FZ_hybrid_aVaR, FZ_hybrid_aES= FZ_hybrid_aVaR[burn_in:], FZ_hybrid_aES[burn_in:]
    FZ_hybrid_aVaR, FZ_hybrid_aES= FZ_hybrid_aVaR[0::seperate], FZ_hybrid_aES[0::seperate]
    
    ML_gaus_GARCH_aVaR, ML_gaus_GARCH_aES= ML_VaR_ES_getter(returns, 'gaussian', alpha, ML_gauss_GARCH_est_par)
    ML_gaus_GARCH_aVaR, ML_gaus_GARCH_aES= ML_gaus_GARCH_aVaR[burn_in:], ML_gaus_GARCH_aES[burn_in:]
    ML_gaus_GARCH_aVaR, ML_gaus_GARCH_aES= ML_gaus_GARCH_aVaR[0::seperate], ML_gaus_GARCH_aES[0::seperate]

    ML_skewT_GARCH_aVaR, ML_skewT_GARCH_aES= ML_VaR_ES_getter(returns, 'skewed student', alpha, ML_skewT_GARCH_est_par)
    ML_skewT_GARCH_aVaR, ML_skewT_GARCH_aES= ML_skewT_GARCH_aVaR[burn_in:], ML_skewT_GARCH_aES[burn_in:]
    ML_skewT_GARCH_aVaR, ML_skewT_GARCH_aES= ML_skewT_GARCH_aVaR[0::seperate], ML_skewT_GARCH_aES[0::seperate]
    
    EDF_GARCH_aVaR, EDF_GARCH_aES= EDF_GARCH_VaR_ES_getter(returns, alpha, ML_gauss_GARCH_est_par)
    EDF_GARCH_aVaR, EDF_GARCH_aES= EDF_GARCH_aVaR[burn_in:], EDF_GARCH_aES[burn_in:]
    EDF_GARCH_aVaR, EDF_GARCH_aES= EDF_GARCH_aVaR[0::seperate], EDF_GARCH_aES[0::seperate]
    
    returns= dfValidation['returns'][burn_in:].tolist()[0::seperate]
    
    # calculating average loss
    FZ_GAS1F_avg_loss= average_FZ_loss(returns, FZ_GAS1F_aVaR, FZ_GAS1F_aES, alpha)
    FZ_GAS2F_avg_loss= average_FZ_loss(returns, FZ_GAS2F_aVaR, FZ_GAS2F_aES, alpha)
    FZ_GARCH_avg_loss= average_FZ_loss(returns, FZ_GARCH_aVaR, FZ_GARCH_aES, alpha)
    FZ_hybrid_avg_loss= average_FZ_loss(returns, FZ_hybrid_aVaR, FZ_hybrid_aES, alpha)
    
    ML_gauss_avg_loss= average_FZ_loss(returns, ML_gaus_GARCH_aVaR, ML_gaus_GARCH_aES, alpha)
    ML_skewT_avg_loss= average_FZ_loss(returns, ML_skewT_GARCH_aVaR, ML_skewT_GARCH_aES, alpha)
    
    EDF_GARCH_avg_loss= average_FZ_loss(returns, EDF_GARCH_aVaR, EDF_GARCH_aES, alpha)
    
    rounding_decimals= 3
    
    print('FZ GAS1F average loss:', round(FZ_GAS1F_avg_loss, rounding_decimals))
    print('FZ GAS2F average loss:', round(FZ_GAS2F_avg_loss, rounding_decimals))
    print('FZ GARCH average loss:', round(FZ_GARCH_avg_loss, rounding_decimals))
    print('FZ hybrid average loss:', round(FZ_hybrid_avg_loss, rounding_decimals))
    
    print('ML gaussian GARCH average loss:', round(ML_gauss_avg_loss, rounding_decimals))
    print('ML skewed student GARCH average loss:', round(ML_skewT_avg_loss, rounding_decimals))
    print('EDF GARCH average loss:', round(EDF_GARCH_avg_loss, rounding_decimals))
    
    # calculate the goodness of fit test for the VaR and ES
    GoF_FZ_GAS1F_VaR_pvalue, GoF_FZ_GAS1F_ES_pvalue= goodness_of_fit(returns, FZ_GAS1F_aVaR, FZ_GAS1F_aES, alpha)
    GoF_FZ_GAS2F_VaR_pvalue, GoF_FZ_GAS2F_ES_pvalue= goodness_of_fit(returns, FZ_GAS2F_aVaR, FZ_GAS2F_aES, alpha)
    GoF_FZ_GARCH_VaR_pvalue, GoF_FZ_GARCH_ES_pvalue= goodness_of_fit(returns, FZ_GARCH_aVaR, FZ_GARCH_aES, alpha)
    GoF_FZ_hybrid_VaR_pvalue, GoF_FZ_hybrid_ES_pvalue= goodness_of_fit(returns, FZ_hybrid_aVaR, FZ_hybrid_aES, alpha)
    
    GoF_ML_gaus_GARCH_VaR_pvalue, GoF_ML_gaus_GARCH_ES_pvalue= goodness_of_fit(returns,ML_gaus_GARCH_aVaR,
                                                                               ML_gaus_GARCH_aES, alpha)
    GoF_ML_skewT_GARCH_VaR_pvalue, GoF_ML_skewT_GARCH_ES_pvalue= goodness_of_fit(returns, ML_skewT_GARCH_aVaR,
                                                                                 ML_skewT_GARCH_aES, alpha)
    
    GoF_EDF_GARCH_VaR_pvalue, GoF_EDF_GARCH_ES_pvalue= goodness_of_fit(returns, EDF_GARCH_aVaR, EDF_GARCH_aES, alpha)
    
    print('FZ GAS1F GoF VaR:', round(GoF_FZ_GAS1F_VaR_pvalue, rounding_decimals))
    print('FZ GAS1F GoF ES:', round(GoF_FZ_GAS1F_ES_pvalue, rounding_decimals))
    
    print('FZ GAS2F GoF VaR:', round(GoF_FZ_GAS2F_VaR_pvalue, rounding_decimals))
    print('FZ GAS2F GoF ES:', round(GoF_FZ_GAS2F_ES_pvalue, rounding_decimals))
    
    print('FZ GARCH GoF VaR:', round(GoF_FZ_GARCH_VaR_pvalue, rounding_decimals))
    print('FZ GARCH GoF ES:', round(GoF_FZ_GARCH_ES_pvalue, rounding_decimals))
    
    print('FZ hybrid GoF VaR:', round(GoF_FZ_hybrid_VaR_pvalue, rounding_decimals))
    print('FZ hybrid GoF ES:', round(GoF_FZ_hybrid_ES_pvalue, rounding_decimals))
    
    print('ML gaussian GARCH GoF VaR:', round(GoF_ML_gaus_GARCH_VaR_pvalue, rounding_decimals))
    print('ML gaussian GARCH GoF ES:', round(GoF_ML_gaus_GARCH_ES_pvalue, rounding_decimals))
    
    print('ML skewed student GARCH GoF VaR:', round(GoF_ML_skewT_GARCH_VaR_pvalue, rounding_decimals))
    print('ML skewed student GARCH GoF ES:', round(GoF_ML_skewT_GARCH_ES_pvalue, rounding_decimals))
    
    print('EDF GARCH GoF VaR:', round(GoF_EDF_GARCH_VaR_pvalue, rounding_decimals))
    print('EDF GARCH GoF ES:', round(GoF_EDF_GARCH_ES_pvalue, rounding_decimals))
    
    # calculating the loss contributions
    FZ_GAS1F_loss_contributer= loss_contributer(returns, FZ_GAS1F_aVaR, FZ_GAS1F_aES, alpha)
    FZ_GAS2F_loss_contributer= loss_contributer(returns, FZ_GAS2F_aVaR, FZ_GAS2F_aES, alpha)
    FZ_GARCH_loss_contributer= loss_contributer(returns, FZ_GARCH_aVaR, FZ_GARCH_aES, alpha)
    FZ_hybrid_loss_contributer= loss_contributer(returns, FZ_hybrid_aVaR, FZ_hybrid_aES, alpha)
    
    ML_gaus_GARCH_loss_contributer= loss_contributer(returns, ML_gaus_GARCH_aVaR, ML_gaus_GARCH_aES, alpha)
    ML_skewT_GARCH_loss_contributer= loss_contributer(returns, ML_skewT_GARCH_aVaR, ML_skewT_GARCH_aES, alpha)
    EDF_GARCH_loss_contributer= loss_contributer(returns, EDF_GARCH_aVaR, EDF_GARCH_aES, alpha)
    
    return [FZ_GAS1F_loss_contributer, FZ_GAS2F_loss_contributer, FZ_GARCH_loss_contributer, FZ_hybrid_loss_contributer,
            ML_gaus_GARCH_loss_contributer, ML_skewT_GARCH_loss_contributer, EDF_GARCH_loss_contributer]

def extension_DM_table(iterated_hourly, iterated_4hourly, direct_8hourly):
    '''
    calculates the diebold-mariano table for the multi-step ahead forecasting setting

    Parameters
    ----------
    iterated_hourly : array
        contains the loss contributions of every model in the hourly iterated setting
    iterated_4hourly : array
        contains the loss contributions of every model in the 4 hourly iterated setting.
    direct_8hourly : array
        contains the loss contribution of every model in the 8 hourly direct setting.

    Returns
    -------
    df : dataframe
        diebold-mariano table.

    '''
    Ihourly_FZ_GAS1F= iterated_hourly[0]
    Ihourly_FZ_GAS2F= iterated_hourly[1]
    Ihourly_FZ_GARCH= iterated_hourly[2]
    Ihourly_FZ_hybrid= iterated_hourly[3]
    Ihourly_ML_gauss= iterated_hourly[4]
    Ihourly_ML_skewedT= iterated_hourly[5]
    Ihourly_EDF_GARCH= iterated_hourly[6]
    
    I4hourly_FZ_GAS1F= iterated_4hourly[0]
    I4hourly_FZ_GAS2F= iterated_4hourly[1]
    I4hourly_FZ_GARCH= iterated_4hourly[2]
    I4hourly_FZ_hybrid= iterated_4hourly[3]
    I4hourly_ML_gauss= iterated_4hourly[4]
    I4hourly_ML_skewedT= iterated_4hourly[5]
    I4hourly_EDF_GARCH= iterated_4hourly[6]
    
    D8hourly_FZ_GAS1F= direct_8hourly[0]
    D8hourly_FZ_GAS2F= direct_8hourly[1]
    D8hourly_FZ_GARCH= direct_8hourly[2]
    D8hourly_FZ_hybrid= direct_8hourly[3]
    D8hourly_ML_gauss= direct_8hourly[4]
    D8hourly_ML_skewedT= direct_8hourly[5]
    D8hourly_EDF_GARCH= direct_8hourly[6]
    
    losses= [Ihourly_FZ_GAS1F, Ihourly_FZ_GAS2F, Ihourly_FZ_GARCH, Ihourly_FZ_hybrid, Ihourly_ML_gauss, Ihourly_ML_skewedT,
             Ihourly_EDF_GARCH, I4hourly_FZ_GAS1F, I4hourly_FZ_GAS2F, I4hourly_FZ_GARCH, I4hourly_FZ_hybrid, I4hourly_ML_gauss,
             I4hourly_ML_skewedT, I4hourly_EDF_GARCH, D8hourly_FZ_GAS1F, D8hourly_FZ_GAS2F, D8hourly_FZ_GARCH, D8hourly_FZ_hybrid,
             D8hourly_ML_gauss, D8hourly_ML_skewedT, D8hourly_EDF_GARCH]
    
    names= ['I-H_FZ-1F', 'I-H_FZ-2F', 'I-H_FZ-G', 'I-H_FZ-hybrid', 'I-H_G-N', 'I-H_G-Skt', 'I-H_G-EDF', 'I-4H_FZ-1F', 'I-4H_FZ-2F',
            'I-4H_FZ-G', 'I-4H_FZ-hybrid', 'I-4H_G-N', 'I-4H_G-Skt', 'I-4H_G-EDF', 'D-8H_FZ-1F', 'D-8H_FZ-2F', 'D-8H_FZ-G', 
            'D-8H_FZ-hybrid', 'D-8H_G-N', 'D-8H_G-Skt', 'D-8H_G-EDF']
    
    df= pd.DataFrame(columns= names, index= names)
    for i in range(len(names)):
        temp= []
        collumn= losses[i]
        for j in range(len(names)):
            if i == j:
                temp.append(np.nan)
            else:
                row= losses[j]
                temp.append(round(diebold_mariano(row, collumn), 3))
        df[names[i]]= temp
    return df
        
###########################
# simulation_study()
alpha= 0.05

single_step= True
multi_step= True

if single_step == True:
    startTraining, endTraining= '2015-01-01', '2019-12-31'
    startValidation, endValidation= '2019-12-29', '2022-04-01'
    burn_in= 48
    
    
    df_BTC_hourly= pd.read_csv('empirical_data/Hourly/BTC.csv')
    df_BTC_hourly['date']= pd.to_datetime(df_BTC_hourly['date'], format= '%Y-%m-%d %H:%M:%S')
    df_BTC_hourly= RW_adder(df_BTC_hourly, alpha)
    
    df_ETH_hourly= pd.read_csv('empirical_data/Hourly/ETH.csv')
    df_ETH_hourly['date']= pd.to_datetime(df_ETH_hourly['date'], format= '%Y-%m-%d %H:%M:%S')
    df_ETH_hourly= RW_adder(df_ETH_hourly, alpha)
    
    df_LTC_hourly= pd.read_csv('empirical_data/Hourly/LTC.csv')
    df_LTC_hourly['date']= pd.to_datetime(df_LTC_hourly['date'], format= '%Y-%m-%d %H:%M:%S')
    df_LTC_hourly= RW_adder(df_LTC_hourly, alpha)
    
    df_XRP_hourly= pd.read_csv('empirical_data/Hourly/XRP.csv')
    df_XRP_hourly['date']= pd.to_datetime(df_XRP_hourly['date'], format= '%Y-%m-%d %H:%M:%S')
    df_XRP_hourly= RW_adder(df_XRP_hourly, alpha)
    
    
    df_Training_BTC_hourly, df_Validation_BTC_hourly= data_splitter(df_BTC_hourly, startTraining, endTraining, startValidation, endValidation, alpha)
    df_Training_ETH_hourly, df_Validation_ETH_hourly= data_splitter(df_ETH_hourly, startTraining, endTraining, startValidation, endValidation, alpha)
    df_Training_LTC_hourly, df_Validation_LTC_hourly= data_splitter(df_LTC_hourly, startTraining, endTraining, startValidation, endValidation, alpha)
    df_Training_XRP_hourly, df_Validation_XRP_hourly= data_splitter(df_XRP_hourly, startTraining, endTraining, startValidation, endValidation, alpha)
    
    
    dict_BTC_hourly_FZ_guesses= {'FZ_GAS1F_guess': [-1.164, -1.757, 0.995, 0.007],
                                 'FZ_GAS2F_guess': [-0.0813, -0.051, 0.886, 0.966, -0.339, 0.007, -0.564, 0.004],
                                 'FZ_GARCH_guess': [-1.955, -2.829, 0.994, 0.031],
                                 'FZ_hybrid_guess': [-2.320, -3.434, 0.974, 0.003, 0.017]}
    
    dict_ETH_hourly_FZ_guesses= {'FZ_GAS1F_guess': [-1.164, -1.757, 0.995, 0.007],
                                 'FZ_GAS2F_guess': [-0.6, -0.3, 0.5, 0.6, -0.05, 0.02, 0.6, 0.05],
                                 'FZ_GARCH_guess': [-1, -1.5, 0.9, 0.1],
                                 'FZ_hybrid_guess': [-2.320, -3.434, 0.974, 0.003, 0.017]}
    
    dict_LTC_hourly_FZ_guesses= {'FZ_GAS1F_guess': [-1, -2, 0.995, 0.007],
                                 'FZ_GAS2F_guess': [-0.0813, -0.051, 0.886, 0.966, -0.339, 0.007, -0.564, 0.004],
                                 'FZ_GARCH_guess': [-1, -1.5, 0.9, 0.1],
                                 'FZ_hybrid_guess': [-2.320, -3.434, 0.974, 0.003, 0.017]}
    
    dict_XRP_hourly_FZ_guesses= {'FZ_GAS1F_guess': [-1.164, -1.757, 0.995, 0.007],
                                 'FZ_GAS2F_guess': [-0.0813, -0.051, 0.886, 0.966, -0.339, 0.007, -0.564, 0.004],
                                 'FZ_GARCH_guess': [-1.955, -2.829, 0.994, 0.031],
                                 'FZ_hybrid_guess': [-2.320, -3.434, 0.974, 0.003, 0.017]}
    
    
    # dict_BTC_hourly_parameters= inSample_parameter_estimator(df_Training_BTC_hourly, alpha, 'BTC', burn_in, 'hourly', dict_BTC_hourly_FZ_guesses)
    # dict_ETH_hourly_parameters= inSample_parameter_estimator(df_Training_ETH_hourly, alpha, 'ETH', burn_in, 'hourly', dict_ETH_hourly_FZ_guesses)
    # dict_LTC_hourly_parameters= inSample_parameter_estimator(df_Training_LTC_hourly, alpha, 'LTC', burn_in, 'hourly', dict_LTC_hourly_FZ_guesses)
    # dict_XRP_hourly_parameters= inSample_parameter_estimator(df_Training_XRP_hourly, alpha, 'XRP', burn_in, 'hourly', dict_XRP_hourly_FZ_guesses)
    
    
    dict_BTC_hourly_parameters= {'FZ_GAS1F_est_par': [-0.6128, -1.1787, 0.9964, 0.0069],
                                 'FZ_GAS2F_est_par': [-0.0052, -0.0055, 0.9921, 0.9956, -0.4158, 0.0018, -0.5477, 0.0008],
                                 'FZ_GARCH_est_par': [-0.0693, -0.1357, 0.9121, 36.4026],
                                 'FZ_hybrid_est_par': [-2.6806, -5.3527, 0.9643, 0.0075, 0.0284],
                                 'ML_gauss_est_par': [0.0190, 0.8796, 0.1001],
                                 'ML_skewT_est_par': [0.0022, 0.9568, 0.0686, 2.3802, 0.9831]}
    
    dict_ETH_hourly_parameters= {'FZ_GAS1F_est_par': [-1.0257, -1.8968, 0.9847, 0.0125],
                                 'FZ_GAS2F_est_par': [-0.018, -0.0268, 0.9823, 0.9855, -0.2095, 0.0037, -0.2845, 0.0064],
                                 'FZ_GARCH_est_par': [-0.0505, -0.0917, 0.9374, 54.9041],
                                 'FZ_hybrid_est_par': [-2.133, -3.8977, 0.9448, 0.0118, 0.0314],
                                 'ML_gauss_est_par': [0.0156, 0.9324, 0.0546],
                                 'ML_skewT_est_par': [0.0063, 0.9408, 0.0687, 2.7647, 1.0148]}
    
    dict_LTC_hourly_parameters= {'FZ_GAS1F_est_par': [-1.267, -2.02, 0.9773, 0.0133],
                                 'FZ_GAS2F_est_par': [-0.0148, -0.0985, 0.9888, 0.9587, -0.0257, 0.0051, 0.4404, 0.0312],
                                 'FZ_GARCH_est_par': [-0.0933, -0.1637, 0.9427, 13.7242],
                                 'FZ_hybrid_est_par': [-2.3171, -4.0023, 0.941, 0.0104, 0.0362],
                                 'ML_gauss_est_par': [0.0225, 0.9382, 0.0422],
                                 'ML_skewT_est_par': [0.008, 0.9457, 0.0591, 2.8838, 1.0075]}
    
    dict_XRP_hourly_parameters= {'FZ_GAS1F_est_par': [-1.0679, -1.8514, 0.9774, 0.016],
                                 'FZ_GAS2F_est_par': [-0.0248, -0.042, 0.9771, 0.9776, -0.1382, 0.007, -0.4458, 0.0091],
                                 'FZ_GARCH_est_par': [-0.0669, -0.1153, 0.9098, 48.9622],
                                 'FZ_hybrid_est_par': [-2.3857, -4.0453, 0.9086, 0.0137, 0.0654],
                                 'ML_gauss_est_par': [0.0232, 0.884, 0.1041],
                                 'ML_skewT_est_par': [0.0097, 0.914, 0.0954, 2.9193, 1.0161]}
    
    
    print('Results for BTC on hourly timeframe:')
    BTC_hourly_DM_table= outSample(df_Validation_BTC_hourly, dict_BTC_hourly_parameters, 'BTC', burn_in, 'hourly')
    
    
    print('Results for ETH on hourly timeframe:')
    ETH_hourly_DM_table= outSample(df_Validation_ETH_hourly, dict_ETH_hourly_parameters, 'ETH', burn_in, 'hourly')
    
    
    print('Results for LTC on hourly timeframe:')
    LTC_hourly_DM_table= outSample(df_Validation_LTC_hourly, dict_LTC_hourly_parameters, 'LTC', burn_in, 'hourly')
    
    
    print('Results for XRP on hourly timeframe:')
    XRP_hourly_DM_table= outSample(df_Validation_XRP_hourly, dict_XRP_hourly_parameters, 'XRP', burn_in, 'hourly')


if multi_step == True:
    '''
    first the parameter values of all models are calculated in every model section (hourly iterated, 4 hourly iterated,
    8 hourly direct). then we load the out-of-sample data set which contains only return data points at every 8th step. 
    we then forecasat the missing return observations according to an AR model where the number of lags is based on the
    AIC. then we give the data set to the extension function to obtain loss contributions for every model. lastly the
    diebold-mariano table is calculated.
    
    
    '''
    print('Results for hourly iterated forecasts:')
    startTraining, endTraining= '2015-01-01', '2019-12-31'
    startValidation, endValidation= '2019-12-29', '2022-04-01'
    burn_in= 48
    
    
    df_BTC_hourly= pd.read_csv('empirical_data/Hourly/BTC.csv')
    df_BTC_hourly['date']= pd.to_datetime(df_BTC_hourly['date'], format= '%Y-%m-%d %H:%M:%S')
    df_Training_BTC_hourly, df_Validation_BTC_hourly= data_splitter(df_BTC_hourly, startTraining, endTraining, startValidation, endValidation, alpha)
    dict_BTC_hourly_FZ_guesses= {'FZ_GAS1F_guess': [-1.164, -1.757, 0.995, 0.007],
                                 'FZ_GAS2F_guess': [-0.0813, -0.051, 0.886, 0.966, -0.339, 0.007, -0.564, 0.004],
                                 'FZ_GARCH_guess': [-1.955, -2.829, 0.994, 0.031],
                                 'FZ_hybrid_guess': [-2.320, -3.434, 0.974, 0.003, 0.017]}
    # dict_BTC_hourly_parameters= inSample_parameter_estimator(df_Training_BTC_hourly, alpha, 'BTC', burn_in, 'hourly', dict_BTC_hourly_FZ_guesses)
    dict_BTC_hourly_parameters= {'FZ_GAS1F_est_par': [-0.6128, -1.1787, 0.9964, 0.0069],
                                 'FZ_GAS2F_est_par': [-0.0052, -0.0055, 0.9921, 0.9956, -0.4158, 0.0018, -0.5477, 0.0008],
                                 'FZ_GARCH_est_par': [-0.0693, -0.1357, 0.9121, 36.4026],
                                 'FZ_hybrid_est_par': [-2.6806, -5.3527, 0.9643, 0.0075, 0.0284],
                                 'ML_gauss_est_par': [0.0190, 0.8796, 0.1001],
                                 'ML_skewT_est_par': [0.0022, 0.9568, 0.0686, 2.3802, 0.9831]}
    
    df_hourly_extension= pd.read_csv('empirical_data/Extension/BTC_hourly_extension.csv')
    df_hourly_extension['date']= pd.to_datetime(df_hourly_extension['date'], format= '%Y-%m-%d %H:%M:%S')
    df_hourly_extension['forecasted']= df_hourly_extension['returns'].isna()
    
    
    # returns= df_hourly_extension[df_hourly_extension['date'] < '2020-01-01']['returns'].tolist()
    # mod= ar_select_order(returns, maxlag= 13, ic= 'aic', glob= True)
    # res= AutoReg(returns, lags= mod.ar_lags, trend= 'n').fit()
    
    for i in range(19928):
        if df_hourly_extension['forecasted'].iloc[14305+i] == True:
            a= df_hourly_extension['returns'].iloc[14305-2+i]*-0.0292 + df_hourly_extension['returns'].iloc[14305-9+i]*0.0166
            b= df_hourly_extension['returns'].iloc[14305-10+i]*0.0122 + df_hourly_extension['returns'].iloc[14305-12+i]*0.0226
            df_hourly_extension.at[14305+i, 'returns']= a+b
    
    training= df_hourly_extension[df_hourly_extension['date'] < endTraining]
    validation= df_hourly_extension[df_hourly_extension['date'] > startValidation]
    
    loss_diff_hourly_iterated= extension(validation, dict_BTC_hourly_parameters, 'BTC', 78, 8)    
    
    
    print('Results for 4-hourly iterated forecasts:')
    startTraining, endTraining= '2015-01-01', '2019-12-31'
    startValidation, endValidation= '2019-12-25', '2022-04-01'
    burn_in= 36
    
    df_BTC_4hourly= pd.read_csv('empirical_data/4-Hourly/BTC.csv')
    df_BTC_4hourly['date']= pd.to_datetime(df_BTC_4hourly['date'], format= '%Y-%m-%d %H:%M:%S')
    df_BTC_4hourly= RW_adder(df_BTC_4hourly, alpha)
    
    df_Training_BTC_4hourly, df_Validation_BTC_4hourly= data_splitter(df_BTC_4hourly, startTraining, endTraining, startValidation, endValidation, alpha)
    dict_BTC_4hourly_FZ_guesses= {'FZ_GAS1F_guess': [-1.164, -1.757, 0.995, 0.007],
                                 'FZ_GAS2F_guess': [-0.0813, -0.051, 0.886, 0.966, -0.339, 0.007, -0.564, 0.004],
                                 'FZ_GARCH_guess': [-1.5, -2.0, 0.7, 0.1],
                                 'FZ_hybrid_guess': [-2.320, -3.434, 0.974, 0.003, 0.017]}
    # dict_BTC_4hourly_parameters= inSample_parameter_estimator(df_Training_BTC_4hourly, alpha, 'BTC', burn_in, '4-hourly', dict_BTC_4hourly_FZ_guesses)
    dict_BTC_4hourly_parameters= {'FZ_GAS1F_est_par': [-0.6674, -1.3093,  0.9952,  0.0083],
                                 'FZ_GAS2F_est_par': [-0.0267, -0.0178,  0.9545,  0.9856, -0.6179, -0.0015, -0.7898, -0.0043],
                                 'FZ_GARCH_est_par': [-0.0462, -0.0904,  0.9552, 37.9584],
                                 'FZ_hybrid_est_par': [-2.8247, -5.7307, 0.9644, 0.0043, 0.0294],
                                 'ML_gauss_est_par': [0.0049, 0.9606, 0.0304],
                                 'ML_skewT_est_par': [0.0090, 0.9288, 0.1222, 2.2763, 0.9759]}
    
    df_4hourly_extension= pd.read_csv('empirical_data/Extension/BTC_4hourly_extension.csv')
    df_4hourly_extension['date']= pd.to_datetime(df_4hourly_extension['date'], format= '%Y-%m-%d %H:%M:%S')
    df_4hourly_extension['forecasted']= df_4hourly_extension['returns'].isna()
    
    # returns = df_4hourly_extension[df_4hourly_extension['date'] < '2020-01-01']['returns'].tolist()
    # mod= ar_select_order(returns, maxlag= 13, ic= 'aic', glob= True)
    # res= AutoReg(returns, lags= mod.ar_lags, trend= 'c').fit()
    
    for i in range(4981):
        if df_4hourly_extension['forecasted'].iloc[3577+i] == True:
            a= df_4hourly_extension['returns'].iloc[3577-6+i]*-0.0378 + df_4hourly_extension['returns'].iloc[3577-7+i]*-0.0365
            df_4hourly_extension.at[3577+i, 'returns']= a
        
    training= df_4hourly_extension[df_4hourly_extension['date'] < endTraining]
    validation= df_4hourly_extension[df_4hourly_extension['date'] > startValidation]
    
    
    loss_diff_4hourly_iterated= extension(validation, dict_BTC_4hourly_parameters, 'BTC', 43, 2)      
    
    
    print('Results for directed forecasts:')
    startTraining, endTraining= '2015-01-01', '2019-12-31'
    startValidation, endValidation= '2019-12-18', '2022-04-01'
    burn_in= 39
    
    df_BTC_8hourly= pd.read_csv('empirical_data/8-Hourly/BTC.csv')
    df_BTC_8hourly['date']= pd.to_datetime(df_BTC_8hourly['date'], format= '%Y-%m-%d %H:%M:%S')
    
    df_Training_BTC_8hourly, df_Validation_BTC_8hourly= data_splitter(df_BTC_8hourly, startTraining, endTraining, startValidation, endValidation, alpha)
    dict_BTC_8hourly_FZ_guesses= {'FZ_GAS1F_guess': [-1.164, -1.757, 0.995, 0.007],
                                 'FZ_GAS2F_guess': [0.07, -0.67, -0.17, 0.47, 0.75, -0.01, 0.82, 0.04],
                                 'FZ_GARCH_guess': [-1.5, -2.0, 0.7, 0.1],
                                 'FZ_hybrid_guess': [-2.320, -3.434, 0.974, 0.003, 0.017]}
    # dict_BTC_8hourly_parameters= inSample_parameter_estimator(df_Training_BTC_8hourly, alpha, 'BTC', burn_in, '8-hourly', dict_BTC_8hourly_FZ_guesses)
    dict_BTC_8hourly_parameters= {'FZ_GAS1F_est_par': [-0.7766 , -1.6459,  0.9872,  0.0089],
                                 'FZ_GAS2F_est_par': [-0.5171, -0.9339,  0.4791,  0.4954,  0.2199, 0.0355,  0.4739,  0.0364],
                                 'FZ_GARCH_est_par': [-0.0722, -0.1457,  0.9055, 37.0159],
                                 'FZ_hybrid_est_par': [-2.7937, -5.5081, 0.9467, -0.0005, 0.0403],
                                 'ML_gauss_est_par': [0.0381, 0.8574, 0.0851],
                                 'ML_skewT_est_par': [0.5551, 0.8652, 3.9982, 2.0117, 0.9836]}
    
    df_8hourly_extension= pd.read_csv('empirical_data/Extension/BTC_8hourly_extension.csv')
    df_8hourly_extension['date']= pd.to_datetime(df_8hourly_extension['date'], format= '%Y-%m-%d %H:%M:%S')
    df_8hourly_extension['forecasted']= df_8hourly_extension['returns'].isna()
    
    training= df_8hourly_extension[df_8hourly_extension['date'] < endTraining]
    validation= df_8hourly_extension[df_8hourly_extension['date'] > startValidation]
    
    loss_diff_8hourly_direct= extension(validation, dict_BTC_8hourly_parameters, 'BTC', 42, 1)  
    
    
    # DM_table= extension_DM_table(loss_diff_hourly_iterated, loss_diff_4hourly_iterated, loss_diff_8hourly_direct)
    # DM_table.to_csv('DM_table.csv', index= True)
    
