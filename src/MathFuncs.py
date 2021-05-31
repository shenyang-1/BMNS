# Numpy imports
from numpy import array, asanyarray
from numpy import diag
from numpy import log10
from numpy import outer
from numpy import sqrt

# Scipy Scientific constants
from scipy.constants import k as kB # Boltzmann constant
from scipy.constants import h as hC # Plancks constant
from scipy.constants import R as rG # Gas constant
from scipy.constants import calorie # cal

# Uncertainties calculations
from uncertainties import umath
from uncertainties import ufloat

#---------------------------#---------------------------#
# 'OrdMag' takes in two values and calculates the orders
#  of magnitude difference between them.
#---------------------------#---------------------------#
def OrdMag(A, B):
    maxV, minV = A, B
    if maxV < minV:
        minV, maxV = maxV, minV

    return log10(maxV/minV)

#---------------------------#---------------------------#
# 'CalcRateTau' Takes in populations and exchange rates
#   as numpy arrays of [val, error],
#  Unpacks these to ufloats for propogating uncertainty
#  Returns ufloats of
#     k12, k21, k13, k31, k23, k32 (s^-1)
#     Tau1, Tau2, Tau3 (sec)
#---------------------------#---------------------------#
def CalcRateTau(pB, pC, kexAB, kexAC, kexBC, rettype="list"):
    # Unpack numpy arrays to ufloats
    pB = ufloat(pB[0], pB[1])
    pC = ufloat(pC[0], pC[1])
    pA = 1. - (pB + pC)
    # Recast pA as ufloat
    pA = ufloat(pA.n, pA.std_dev)
    # Calculate pA (prop err)
    kexAB = ufloat(kexAB[0], kexAB[1])
    kexAC = ufloat(kexAC[0], kexAC[1])
    kexBC = ufloat(kexBC[0], kexBC[1])
    #Define forward/backward exchange rates
    k12 = kexAB * pB / (pB + pA)
    k21 = kexAB * pA / (pB + pA)
    k13 = kexAC * pC / (pC + pA)
    k31 = kexAC * pA / (pC + pA)
    if kexBC != 0.:
        k23 = kexBC * pC / (pB + pC)
        k32 = kexBC * pB / (pB + pC)
    else:
        k23 = ufloat(0., 0.)
        k32 = ufloat(0., 0.)

    # Calculate 3-state lifetimes
    # GS lifetime
    if k12 != 0. and k13 != 0.:
        tau1 = 1./k12 + 1./k13
    elif k12 == 0. and k13 != 0.:
        tau1 = 1./k13
    elif k12 != 0. and k13 == 0.:
        tau1 = 1./k12
    else:
        tau1 = ufloat(0., 0.)

    # ES1 lifetime
    if k21 != 0. and k23 != 0.:
        tau2 = 1./k21 + 1./k23
    elif k21 == 0. and k23 != 0.:
        tau2 = 1./k23
    elif k21 != 0. and k23 == 0.:
        tau2 = 1./k21
    else:
        tau2 = ufloat(0., 0.)

    # ES2 lifetime
    if k31 != 0. and k32 != 0.:
        tau3 = 1./k31 + 1./k32
    elif k31 == 0. and k32 != 0.:
        tau3 = 1./k32
    elif k31 != 0. and k32 == 0.:
        tau3 = 1./k31
    else:
        tau3 = ufloat(0., 0.)
    if rettype == "list":
        return k12, k21, k13, k31, k23, k32, tau1, tau2, tau3
    elif rettype == "dict":
        # Get separate dictionaries of errors and parameter values
        #  then combine and return the master dict
        parD = {"pA":pA.n, "k12":k12.n, "k21":k21.n, "k13":k13.n, "k31":k31.n,
                "k23":k23.n, "k32":k32.n, "tau1":tau1.n, "tau2":tau2.n, "tau3":tau3.n}
        parErr = {"pA_err":pA.std_dev, "k12_err":k12.std_dev, "k21_err":k21.std_dev,
                  "k13_err":k13.std_dev, "k31_err":k31.std_dev, "k23_err":k23.std_dev,
                  "k32_err":k32.std_dev, "tau1_err":tau1.std_dev, "tau2_err":tau2.std_dev,
                  "tau3_err":tau3.std_dev}
        retDict = parD.copy()
        retDict.update(parErr)
        return retDict

#---------------------------#---------------------------#
# 'CalcRateTau' Takes in temp (K) and exchange rates
#   as ufloats, the returns the following as ufloats:
#     dG2, ddG12, ddG21, dG3, ddG13, ddG31, ddG23, ddG32
#     in kcal/mol
#---------------------------#---------------------------#
def CalcG(te, k12, k21, k13, k31, k23, k32, pB, pC, rettype="list"):
    # Recast exchange rates as ufloats
    k12 = ufloat(k12.n, k12.std_dev)
    k13 = ufloat(k13.n, k13.std_dev)
    k21 = ufloat(k21.n, k21.std_dev)
    k31 = ufloat(k31.n, k31.std_dev)
    k23 = ufloat(k23.n, k23.std_dev)
    k32 = ufloat(k32.n, k32.std_dev)
    # Unpack numpy arrays to ufloats
    pB = ufloat(pB[0], pB[1])
    pC = ufloat(pC[0], pC[1])
    pA = 1. - (pB + pC)

    # Calc kcals
    kcal = calorie * 1e3
    # Delta G's of ESs (dG) and transition barriers (ddG)
    dG12, dG13 = ufloat(0., 0.), ufloat(0., 0.)
    ddG12, ddG13 = ufloat(0., 0.), ufloat(0., 0.)
    ddG21, ddG31 = ufloat(0., 0.), ufloat(0., 0.)
    ddG23, ddG32 = ufloat(0., 0.), ufloat(0., 0.)

    # Calculate energies of excited states
    if k12 != 0. and k21 != 0.:
        dG12 = (-umath.log((k12*hC)/(kB*te))*rG*te) - (-umath.log((k21*hC)/(kB*te))*rG*te)
        dG12 = dG12 / kcal

    elif pB != 0.:
        dG12 = -rG * te * umath.log(pB/pA)
        dG12 = dG12 / kcal

    if k13 != 0. and k31 != 0.:
        dG13 = (-umath.log((k13*hC)/(kB*te))*rG*te) - (-umath.log((k31*hC)/(kB*te))*rG*te)
        dG13 = dG13 / kcal

    elif pC != 0.:
        dG13 = -rG * te * umath.log(pC/pA)
        dG13 = dG13 / kcal

    # Calculate forward and reverse barriers (Joules/mol)
    #  Then convert to kcal/mol
    if k12 != 0.:
        ddG12 = -umath.log((k12*hC)/(kB*te))*rG*te
        ddG12 = ddG12 / kcal # Convert to kcal/mol
    if k21 != 0.:
        ddG21 = -umath.log((k21*hC)/(kB*te))*rG*te
        ddG21 = ddG21 / kcal # Convert to kcal/mol
    if k13 != 0.:
        ddG13 = -umath.log((k13*hC)/(kB*te))*rG*te
        ddG13 = ddG13 / kcal # Convert to kcal/mol
    if k31 != 0.:
        ddG31 = -umath.log((k31*hC)/(kB*te))*rG*te
        ddG31 = ddG31 / kcal # Convert to kcal/mol
    if k23 != 0.:
        ddG23 = -umath.log((k23*hC)/(kB*te))*rG*te
        ddG23 = ddG23 / kcal # Convert to kcal/mol
    if k32 != 0.:
        ddG32 = -umath.log((k32*hC)/(kB*te))*rG*te
        ddG32 = ddG32 / kcal # Convert to kcal/mol
    if rettype == "list":
        return dG12, ddG12, ddG21, dG13, ddG13, ddG31, ddG23, ddG32
    elif rettype == "dict":
        # Get separate dictionaries of errors and parameter values
        #  then combine and return the master dict
        parD = {"dG12":dG12.n, "dG13":dG13.n, "ddG12":ddG12.n, "ddG21":ddG21.n,
                "ddG13":ddG13.n, "ddG31":ddG31.n, "ddG23":ddG23.n, "ddG32":ddG32.n}
        parErr = {"dG12_err":dG12.std_dev, "dG13_err":dG13.std_dev,
                  "ddG12_err":ddG12.std_dev, "ddG21_err":ddG21.std_dev,
                  "ddG13_err":ddG13.std_dev, "ddG31_err":ddG31.std_dev,
                  "ddG23_err":ddG23.std_dev, "ddG32_err":ddG32.std_dev}
        retDict = parD.copy()
        retDict.update(parErr)
        return retDict

#---------------------------#---------------------------#
# 'cov2corr' Takes in a covariance matrix and returns
#   a correlation matrix.
#---------------------------#---------------------------#
def cov2corr(cov, return_std=False):
    '''convert covariance matrix to correlation matrix

    Parameters
    ----------
    cov : array_like, 2d
        covariance matrix, see Notes

    Returns
    -------
    corr : ndarray (subclass)
        correlation matrix
    return_std : bool
        If this is true then the standard deviation is also returned.
        By default only the correlation matrix is returned.

    Notes
    -----
    This function does not convert subclasses of ndarrays. This requires
    that division is defined elementwise. np.ma.array and np.matrix are allowed.

    '''
    cov = asanyarray(cov)
    std_ = sqrt(diag(cov))
    corr = cov / outer(std_, std_)
    if return_std:
        return corr, std_
    else:
        return corr
