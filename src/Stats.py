#########################################################################
# Statistics package - Calculates Akaike's Information Criterion,
#   Bayesian Information Criterion, etc
#########################################################################
# General Imports
import os, sys
import datetime

# Numnpy imports
from numpy import array, asarray, asanyarray
from numpy import diag, dot
from numpy import exp
from numpy import log
from numpy import outer
from numpy import savetxt, sqrt, square
from numpy import sum as sumM
from numpy import unique
from numpy import zeros
from numpy.linalg import inv
# Scipy Scientific constants
from scipy.constants import k as kB # Boltzmann constant
from scipy.constants import h as hC # Plancks constant
from scipy.constants import R as rG # Gas constant
from scipy.constants import calorie # cal
# Uncertainties calculations
from uncertainties import umath
from uncertainties import ufloat
# Pandas imports
import pandas as pd

#---------------------------#---------------------------#
# 'CompareModels' takes in two model .csv files and compares
#  them row-by-row using AIC, BIC, and F-test.
#  The inputed stats .csv files contain the AIC, BIC,
#   RSS, and DF values needed to compare the models
#---------------------------#---------------------------#
def CompareModels(paths):
    # Create list of model panda dataframes
    pdArry = []
    nvals = []
    aics = []
    bics = []
    mnames = []
    # Current directory
    curDir = os.getcwd()
    # Get timestamp for generating folder
    mydate = datetime.datetime.now()
    tst = mydate.strftime("%m%d%y-%H%Mh%Ss_ModelComparisons.csv")

    for n in paths:
        pdArry.append(pd.read_csv(n, sep=","))
        # Get model names from input data files
        mnames.append(n.replace(".csv", "").split("/")[-1])
    # Check same size
    for i in pdArry:
        nvals.append(i.N[0])
    nvals = asarray(nvals)
    # Check that all N-values are the same
    if len(unique(nvals)) == 1:
        for i in pdArry:
            aics.append(i.AIC[0])
            bics.append(i.BIC[0])

    else:
        print("!! ERROR: Models do not contain same length data !!")
        print("   All comparative models must use the same data.")
    aics, bics = asarray(aics), asarray(bics)
    outStats = []
    nIC_Test(aics, outStats, mnames, name="AIC")
    nIC_Test(bics, outStats, mnames, name="BIC")
    # Write out stats
    statsP = os.path.join(curDir, tst)
    with open(statsP, "w") as file:
        file.write("ModelNum,ModelName,ModelSelection,nIC,dnIC,Prob\n")
        for line in outStats:
            file.write(line)
#---------------------------#---------------------------#
# 'cAICwt' computes Akaike/Bayesian weights for likelihoods
# Ref:  Psychonomic Bulletin & Review 2004, 11 (1), 192-196
#---------------------------#---------------------------#
def cnICwt(dnic, dnic_all):
    return exp(-0.5 * dnic) / array([exp(-0.5*x) for x in dnic_all]).sum()

#---------------------------#---------------------------#
# 'AIC_Test' compares AIC values and quantifies best model
#---------------------------#---------------------------#
def nIC_Test(nics, outStats, models, name = "AIC"):
    # Normalize AIC values to get dAIC for all possible AIC
    dnics = nics - min(nics)
    nicwts = []
    for idx, m in enumerate(dnics):
        nicwts.append([idx+1, cnICwt(m, dnics)])
    nicwts = asarray(nicwts)

    outStr ='''o Model #%i is %s.
  - %s = %s
  - Model probability = %s'''
    print("\n------ %s Tests ------" % name)
    for n,m,d,mdl in zip(nics, nicwts, dnics, models):
        if m[1] == nicwts[:,1].max():
            flag = "FAVORED"
        else:
            flag = "DISFAVORED"
        print(outStr % (m[0], flag, name, n, m[1]))
        # Put statistics in to list that will be written out
        appStr = "%s,%s,%s,%s,%s,%s\n" % (m[0], mdl, name, n, d, m[1])
        outStats.append(appStr)

def F_test(RSS1, RSS2, df1, df2):
    print("\n------ F Test ------")
    if df1 == df2:
        print("  * F-test cannot be performed because degrees of freedom of both models are the same.")
    else:
        Fv = ((RSS1/RSS2) / (df1/df2)) / (RSS2/df2)
        print("  *", Fv)
#---------------------------#---------------------------#
# 'WriteStats' takes in a 'least_squares' fit, global
#   object, fitnum, and fit-type flag and writes out
#   statistical measures of the fit
# Takes in:
#   outPath : Folder where statistical fits are to be written
#   mPath : sub-folder of outPath, where matrices will be written
#---------------------------#---------------------------#
def WriteStats(outPath, mPath, fit, ob, dof, N, K, chisq, redchisq, fitnum, flag, matrices=True):
    ## Definte stat path names
    # Main fit file
    if flag == "polish":
        statsP = os.path.join(outPath, "PolishedStats_%s.csv" % ob.name)
    elif flag == "local":
        statsP = os.path.join(outPath, "LocalStats_%s.csv" % ob.name)
    # Fit residuals matrix path
    pResid = os.path.join(mPath, "Residuals_%s_%s.csv")
    # Covariance matrix path
    pCov = os.path.join(mPath, "Covariance_%s_%s.csv")
    # Jacobian matrix path
    pJac = os.path.join(mPath, "Jacobian_%s_%s.csv")
    # Correlation matrix path
    pCorr = os.path.join(mPath, "Correlation_%s_%s.csv")
    # Standard error
    pSerr = os.path.join(mPath, "StdErr_%s_%s.csv")
    # Calculate statistical measures
    rss = cRSS(fit.fun)
    tss = cTSS(ob.R1pD[:,2])
    aic = cAIC(rss, K, N)
    bic = cBIC(rss, K, N)
    rsq, adjrsq = cRvals(rss, tss, dof, N)
    if ob.R1pD[:,3].sum() != 0.:
        serr, cov, corr, sdr = cStdErr(fit.x, fit.fun, fit.jac, dof)
    else:
        serr = cov = corr = sdr = zeros((3,3))
    # Generate stats dictionary of numerical fit values
    statsN = ["N", "DF", "K", "ChiSq", "RedChiSq", "Rsq", "AdjRsq",
              "AIC", "BIC", "RSS", "TSS", "SDR"]
    stats = {"N":N, "DF": dof, "K":K, "ChiSq":chisq, "RedChiSq":redchisq,
             "Rsq":rsq, "AdjRsq":adjrsq, "AIC":aic, "BIC":bic, "RSS":rss,
             "TSS":tss, "SDR":sdr}
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write out  local stats
    if len(ob.localFits) != 0 and flag == "local":
        # Check to see if a file exists already
        if not os.path.isfile(statsP): wo = "w"
        else: wo = "a"
        # If file does not exist, write out header and values.
        if wo == "w":
            with open(statsP, "w") as file:
                # Header
                file.write("Name,FitNum," + ",".join(statsN) + "\n")
                # Values
                file.write("%s,%s," % (ob.name, fitnum) + ",".join([str(stats[x]) for x in statsN]) + "\n")

        else:
            with open(statsP, "a") as file:
                file.write("%s,%s," % (ob.name, fitnum) + ",".join([str(stats[x]) for x in statsN]) + "\n")

        ## Write out fit matrices
        if matrices == True:
            savetxt(pResid % (ob.name, fitnum), fit.fun, delimiter=",")
            savetxt(pCov % (ob.name, fitnum), cov, delimiter=",")
            savetxt(pSerr % (ob.name, fitnum), serr, delimiter=",")
            savetxt(pJac % (ob.name, fitnum), fit.jac, delimiter=",")
            savetxt(pCorr % (ob.name, fitnum), corr, delimiter=",")
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write out polished stats
    elif len(ob.polishedFits) != 0 and flag == "polish":
        # Check to see if a file exists already
        if not os.path.isfile(statsP): wo = "w"
        else: wo = "a"
        # If file does not exist, write out header and values.
        if wo == "w":
            with open(statsP, "w") as file:
                # Header
                file.write("Name,FitNum," + ",".join(statsN) + "\n")
                # Values
                file.write("%s,%s," % (ob.name, fitnum) + ",".join([str(stats[x]) for x in statsN]) + "\n")
        else:
            with open(statsP, "a") as file:
                file.write("%s,%s," % (ob.name, fitnum) + ",".join([str(stats[x]) for x in statsN]) + "\n")

        ## Write out fit matrices
        savetxt(pResid % (ob.name, fitnum), fit.fun, delimiter=",")
        savetxt(pCov % (ob.name, fitnum), cov, delimiter=",")
        savetxt(pSerr % (ob.name, fitnum), serr, delimiter=",")
        savetxt(pJac % (ob.name, fitnum), fit.jac, delimiter=",")
        savetxt(pCorr % (ob.name, fitnum), corr, delimiter=",")
#---------------------------#---------------------------#
# 'cRSS' calculates the residual sum of squares from
#  a residual matrix
#---------------------------#---------------------------#
def cRSS(resid):
    return sumM(square(resid))

#---------------------------#---------------------------#
# 'cTSS' calculates the total sum of squares from
#  a residual matrix
#---------------------------#---------------------------#
def cTSS(R1p):
    return sumM(array([(y - R1p.mean())**2. for y in R1p]))

#---------------------------#---------------------------#
# 'cAIC' calculates Akaike's Information Criterion
#   for a given fit
# Ref: http://www.originlab.com/doc/Origin-Help/PostFit-CompareFitFunc
#---------------------------#---------------------------#
def cAIC(RSS, K, N):
    aic = None
    if (N / K) >= 40.:
        aic = N * log(RSS/N) + 2. * K
    else:
        aic = N * log(RSS/N) + 2. * K + ((2. * K * (K + 1.)) / (N - K - 1.))
    return aic

#---------------------------#---------------------------#
# 'cBIC' calculates Bayesian Information Criterion
#   for a given fit
# Ref: http://www.originlab.com/doc/Origin-Help/PostFit-CompareFitFunc
#---------------------------#---------------------------#
def cBIC(RSS, K, N):
    bic = N * log(RSS / N) + K * log(N)
    return bic

#---------------------------#---------------------------#
# 'cStdErr' calculates the standard error of a least squares
#   fit given the jacobian, residuals, and deg of freedom
# Returns: standard error array, covariance matrix,
#          correlation coefficient matrix, and standard
#          deviation of residual variance
# Ref: http://www.mathworks.com/matlabcentral/newsreader/view_thread/157530
# Ref: http://www.originlab.com/doc/Origin-Help/NLFit-Algorithm
#---------------------------#---------------------------#
def cStdErr(Params, resid, jac, dof):
    # Calculate standard-deviation of the residuals
    sdr = sqrt(cRSS(resid)/dof)
    # Calculate variance-covariance matrix
    cov = sdr**2. * inv(dot(jac.T, jac))
    # Calculate standard error as square root of the diagonals
    #   of the covariance matrix
    serr = sqrt(diag(cov))
    corrcoef = cov2corr(cov)

    return serr, cov, corrcoef, sdr

#---------------------------#---------------------------#
# 'cRvals' calculates R-squared and adj. R-square values
#  Takes in resid. sum of squares and tot. sum of squares
# Ref: http://www.originlab.com/doc/Origin-Help/Interpret-Regression-Result
# Ref: https://en.wikipedia.org/wiki/Coefficient_of_determination
#---------------------------#---------------------------#
def cRvals(RSS, TSS, dof, N):
    rsq = 1. - (RSS/TSS)
    adjrsq = 1. - (RSS / (dof - 1))/(TSS / (N - 1))
    return rsq, adjrsq

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
