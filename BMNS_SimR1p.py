#########################################################################
# Bloch-McConnell 2-/3-state Fitting Program v1.24
#  Beta version.
#  Isaac Kimsey 12-09-2015
#
# External Dependencies: numpy, matplotlib, scipy, leastsqbound, uncertainties
#########################################################################

### General imports ###
import sys
### Direct numpy imports ###
from numpy import absolute, append, arctan, array, asarray
from numpy import cos
from numpy import diag, dot
from numpy import exp
from numpy import float64
from numpy import hstack
from numpy import iscomplexobj, iscomplex, isinf, isnan
from numpy import log
from numpy import nan_to_num
from numpy import pi
from numpy import shape, sin, std, sqrt
from numpy import tan
from numpy import vstack
from numpy import zeros
### Numpy sub imports ###
from numpy.linalg import eig, inv, norm
from numpy.random import normal
### Scipy Sub Imports ###
from scipy.stats import norm as normDist
from scipy.optimize import curve_fit
### Uncertainties imports for error propagation
from uncertainties import umath
from uncertainties import ufloat

import matplotlib.pyplot as plt

#########################################################################
# Monte-Carlo Error Estimator
#   - fx : function to be iterated over
#   - Params : Numpy array that will be passed to fx, in order
#   - err : Error in given value being passed to fx
#   - MCnum : number of MC iterations to go over fx
#   - Normal : Return normal distrubtion, if false, returns std.dev
#   - AlignMag : Flag to pass to function
# Returns a singular error value based on a std. dev or normal dist fit
#  from MCnum of iterations over fx
#########################################################################
def MCError(fx, Params, err, MCnum, Normal=False, AlignMag="auto"):
    if Normal == False:
        return std(normal(fx(*Params, AlignMag=AlignMag), err, MCnum))
    else:
        return normDist.fit(normal(fx(*Params, AlignMag=AlignMag), err, MCnum))[1]

#########################################################################
# Calculate R2eff given R1rho and parameters
#  Takes in:
#  kR1p - known R1rho value
#  pB, pC
#  dwB, dwC - in ppm (will convert to rad/sec within)
#  kexAB/AC - exch rates
#  R1
#  w1 - in Hz
#  wrf - in Hz
#  lf - MHz
#  AlignMag - to properly calculate thetaAvg needed for R2eff plot
#  Error - flag to use uncertainties error propagation
#########################################################################
def CalcR2eff(kR1p, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, R1, w1, wrf, lf, AlignMag = "auto", Error = False):
    # Convert w1, wrf to rad/sec from Hz
    w1 = float(w1) * 2. * pi
    wrf = float(wrf) * 2. * pi

    # Calc other simple parameters
    pA = 1. - (pB + pC)
    lf = float(lf)
    #Convert dw from ppm to rad/s
    dwB = dwB * lf * 2. * pi # dw(ppm) * base-freq (eg 151 MHz, but just 151) * 2PI, gives rad/s
    dwC = dwC * lf * 2. * pi

    ################################
    #####  R2eff Calculations  #####
    ################################

    # Calculate pertinent frequency offsets/etc for alignment and projection
    lOmegaA, lOmegaB, lOmegaC, uOmega1, uOmega2, uOmega3, uOmegaAvg,\
    delta1, delta2, delta3, deltaAvg, theta1, theta2, theta3, thetaAvg = \
                            AlignMagVec(w1, wrf, pA, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, AlignMag)

    if Error == False:
        return (kR1p/sin(thetaAvg)**2.) - (R1/(tan(thetaAvg)**2.))
    else:
        return (kR1p/umath.sin(thetaAvg)**2.) - (R1/(umath.tan(thetaAvg)**2.))

#########################################################################
# Magnetization Alignment Vector tool. Needs:
#   - w1 (rad/sec)
#   - wrf (rad/sec)
#   - pA, pB, pC
#   - dwB, dwC (rad/sec)
#   - kexAB/AC/BC (s-1)
#   - AlignMag : Flag for GS, Avg or Auto mag alignment depending on kex/dw
# Returns vectors corresponding to alignment of magnetization type:
#   - uOmega1/2/3/avg (rad/sec) : diff. b/w GS omega and ES omegas
#   - delta1/2/3/avg (rad/sec) : omega - wrf
#   - theta1/2/3/avg (rad) : tilt angle
#########################################################################
def AlignMagVec(w1, wrf, pA, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, AlignMag = "auto"):

    # Simplistic scenario where in 3-state exchange, dominant population of
    #  excited state will dominate how magnetization is aligned.
    # Does not treat separate aligments
    if pB > pC:
        exchReg = kexAB / absolute(dwB)
    else:
        exchReg = kexAC / absolute(dwC)

    ######################################
    #### AUTO Magnetization Alignment ####
    ######################################
    if AlignMag == "auto":
        # If ES_1 or ES_2 is in slow exchange (kex/dw <= 0.5), then toggle
        #  to align magnetization along GS.
        # This neglects the population contribution to everything.
        if exchReg <= 1.:
            ## Align magnetization along GS ##
            #Calculate Resonant Frequencies of GS, ES1, ES2 (in rad/sec)
            uOmega1 = -(pB*dwB + pC*dwC) / (pA + pB + pC) # (rad/sec)
            uOmega2 = uOmega1 + dwB # (rad/sec)
            uOmega3 = uOmega1 + dwC # (rad/sec)
            uOmegaAvg = pA*uOmega1 + pB*uOmega2 + pC*uOmega3 # Resonance Offset (rad/sec) wrt GS
                                                                # Given as uOmega-bar = sum(i=1, N)[ p_i + uOmega_i]
            # Offsets wrt GS
            uOmega2 = uOmega2 - uOmega1                                                
            uOmega3 = uOmega3 - uOmega1  
            uOmegaAvg = uOmegaAvg - uOmega1
            uOmega1 = uOmega1 - uOmega1

            # Calculate resonance offset from the carrier (wrf, ie. spinlock offset)
            # delta(uOmega) = uOmega-bar - lOmega_rf (rad/s)
            # All values in rad/sec
            delta1 = (uOmega1 - wrf) # rad/s
            delta2 = (uOmega2 - wrf) # rad/s
            delta3 = (uOmega3 - wrf) # rad/s
            deltaAvg = (uOmega1 - wrf) # rad/s, avg delta is GS - carrier

            # Because need to calculate arctan instead of arcot, delta is in denominator
            # When on-res, arcot(0/w1*2pi) == pi/2,
            #  However, arctan(w1*2pi/0) is undefined
            #  So when deltaX == 0, must manually declare thetaX == pi/2
            if deltaAvg == 0.:
                theta1 = theta2 = theta3 = thetaAvg = pi/2.
            else:
                thetaAvg = float(arctan(w1/deltaAvg)) # == arccot(deltaAvg/(w1*))
                theta1 = theta2 = theta3 = thetaAvg    

            ## GS,ES1,ES2 along average state
            tempState = asarray([w1,0.,deltaAvg], float64)
            # Normalize vector
            lOmegaA = tempState / norm(tempState)
            lOmegaB = lOmegaC = lOmegaA

        # If exchange regime is non-slow (>0.5), the align along average
        else:
            ## Along Avg calculation ##  
            #Calculate Resonant Frequencies of GS, ES1, ES2 (in Hz)
            uOmega1 = -(pB*dwB + pC*dwC) / ((pA + pB + pC)) # (rad/sec)
            uOmega2 = uOmega1 + dwB # (rad/sec)
            uOmega3 = uOmega1 + dwC # (rad/sec)
            uOmegaAvg = pA*uOmega1 + pB*uOmega2 + pC*uOmega3 #Average Resonance Offset (rad/sec)
                                                           # Given as uOmega-bar = sum(i=1, N)[ p_i + uOmega_i]

            # Calculate resonance offset from the carrier (wrf, ie. spinlock offset)
            # delta(uOmega) = uOmega-bar - lOmega_rf (rad/s)
            # All values in rad/sec
            delta1 = (uOmega1 - wrf) # rad/s
            delta2 = (uOmega2 - wrf) # rad/s
            delta3 = (uOmega3 - wrf) # rad/s
            deltaAvg = (uOmegaAvg - wrf) # rad/s
            # deltaAvg = delta1 # Align along GS

            # Because need to calculate arctan instead of arcot, deltaAvg in denominator
            # When on-res, arcot(0/w1*2pi) == pi/2,
            #  However, arctan(w1*2pi/0) is undefined
            #  So when deltaAvg == 0, must manually declare theta1 == pi/2
            if deltaAvg == 0.:
                theta1 = theta2 = theta3 = thetaAvg = pi/2.
            else:
                theta1 = float(arctan(w1/deltaAvg)) # == arccot(deltaAvg/(w1*2pi))
                theta2 = theta3 = thetaAvg = theta1

            ## GS,ES1,ES2 along average state
            tempState = asarray([w1,0.,deltaAvg], float64)
            # Normalize vector
            lOmegaA = tempState / norm(tempState)
            lOmegaB = lOmegaA
            lOmegaC = lOmegaA

    ######################################
    ##### AVG Magnetization Alignment ####
    ######################################
    elif AlignMag == "avg":
        ## Along Avg calculation ##  
        #Calculate Resonant Frequencies of GS, ES1, ES2 (in Hz)
        uOmega1 = -(pB*dwB + pC*dwC) / ((pA + pB + pC)) # (rad/sec)
        uOmega2 = uOmega1 + dwB # (rad/sec)
        uOmega3 = uOmega1 + dwC # (rad/sec)
        uOmegaAvg = pA*uOmega1 + pB*uOmega2 + pC*uOmega3 #Average Resonance Offset (rad/sec)
                                                       # Given as uOmega-bar = sum(i=1, N)[ p_i + uOmega_i]

        # Calculate resonance offset from the carrier (wrf, ie. spinlock offset)
        # delta(uOmega) = uOmega-bar - lOmega_rf (rad/s)
        # All values in rad/sec
        delta1 = (uOmega1 - wrf) # rad/s
        delta2 = (uOmega2 - wrf) # rad/s
        delta3 = (uOmega3 - wrf) # rad/s
        deltaAvg = (uOmegaAvg - wrf) # rad/s
        # deltaAvg = delta1 # Align along GS

        # Because need to calculate arctan instead of arcot, deltaAvg in denominator
        # When on-res, arcot(0/w1*2pi) == pi/2,
        #  However, arctan(w1*2pi/0) is undefined
        #  So when deltaAvg == 0, must manually declare theta1 == pi/2
        if deltaAvg == 0.:
            theta1 = theta2 = theta3 = thetaAvg = pi/2.

        else:
            theta1 = float(arctan(w1/deltaAvg)) # == arccot(deltaAvg/(w1*2pi))
            theta2 = theta3 = thetaAvg = theta1

        ## GS,ES1,ES2 along average state
        tempState = asarray([w1,0.,deltaAvg], float64)
        # Normalize vector
        lOmegaA = tempState / norm(tempState)
        lOmegaB = lOmegaA
        lOmegaC = lOmegaA

    ######################################
    ##### GS Magnetization Alignment #####
    ######################################
    elif AlignMag == "gs":
        ## Align magnetization along GS ##
        #Calculate Resonant Frequencies of GS, ES1, ES2 (in rad/sec)
        uOmega1 = -(pB*dwB + pC*dwC) / (pA + pB + pC) # (rad/sec)
        uOmega2 = uOmega1 + dwB # (rad/sec)
        uOmega3 = uOmega1 + dwC # (rad/sec)
        uOmegaAvg = pA*uOmega1 + pB*uOmega2 + pC*uOmega3 # Resonance Offset (rad/sec) wrt GS
                                                            # Given as uOmega-bar = sum(i=1, N)[ p_i + uOmega_i]
        # print w1/(2.*pi), wrf/(2.*pi)
        # print "uOmega1", uOmega1/(2.*pi)
        # print "uOmega2", uOmega2/(2.*pi)
        # print "uOmega3", uOmega3/(2.*pi)
        # print "uOmegaAvg", uOmegaAvg/(2.*pi)
        # print "-----------------"

        # Offsets wrt GS
        uOmega2 = uOmega2 - uOmega1                                                
        uOmega3 = uOmega3 - uOmega1  
        uOmegaAvg = uOmegaAvg - uOmega1
        uOmega1 = uOmega1 - uOmega1
        # print w1/(2.*pi), wrf/(2.*pi)
        # print "uOmega1", uOmega1/(2.*pi)
        # print "uOmega2", uOmega2/(2.*pi)
        # print "uOmega3", uOmega3/(2.*pi)
        # print "uOmegaAvg", uOmegaAvg/(2.*pi)
        # print "*************************"

        # Calculate resonance offset from the carrier (wrf, ie. spinlock offset)
        # delta(uOmega) = uOmega-bar - lOmega_rf (rad/s)
        # All values in rad/sec
        delta1 = (uOmega1 - wrf) # rad/s
        delta2 = (uOmega2 - wrf) # rad/s
        delta3 = (uOmega3 - wrf) # rad/s
        deltaAvg = (uOmega1 - wrf) # rad/s, avg delta is GS - carrier
        # print w1/(2.*pi), wrf/(2.*pi)
        # print "delta1", delta1/(2.*pi)
        # print "delta2", delta2/(2.*pi)
        # print "delta3", delta3/(2.*pi)
        # print "deltaAvg", deltaAvg/(2.*pi)
        # print "/////////////////////////////"
        # sys.exit()
        # Because need to calculate arctan instead of arcot, delta is in denominator
        # When on-res, arcot(0/w1*2pi) == pi/2,
        #  However, arctan(w1*2pi/0) is undefined
        #  So when deltaX == 0, must manually declare thetaX == pi/2
        if deltaAvg == 0.:
            thetaAvg = pi/2.
            theta1 = theta2 = theta3 = thetaAvg
        else:
            thetaAvg = float(arctan(w1/deltaAvg)) # == arccot(deltaAvg/(w1*))
            theta1 = theta2 = theta3 = thetaAvg    

        ## GS,ES1,ES2 along average state
        tempState = asarray([w1,0.,deltaAvg], float64)
        # Normalize vector
        lOmegaA = tempState / norm(tempState)
        lOmegaB = lOmegaC = lOmegaA

    return (lOmegaA, lOmegaB, lOmegaC, uOmega1, uOmega2, uOmega3, uOmegaAvg,
            delta1, delta2, delta3, deltaAvg,
            theta1, theta2, theta3, thetaAvg)

#########################################################################
# Fitting function using Laguerre Approximations
# Fits 2-state with Laguerre
# Fits 3-state with no minor to general 3-state
# Fits 3-state with minor to starlike equation
#  Returns R1ho and R2eff
#   Params = vector of fitted parameters to be unpacked
#            Parameters in vector can vary
#   w1 = SLP (given in Hz, converted to rad/s for calcs later on)
#   wrf = Offset from carrier
#         (given in "corrected" Hz, converted to rad/s later for calcs)
#   lf = Larmor freq (MHz, to calc dw from ppm)
#   time = vector of time increments (sec) from Tmin-Tmax
#   AlignMag = How to align magnetization / project
#              Auto = calculates slow or fast exchange, aligns gs or avg
#              Avg = aligns along average
#              GS  = Aligns along ground-state
#   R2eff_flag = 0 : returns R1p, 1 : returns R1p+R2eff
#   kR1p = known R1rho value, if known, will be used to calculate Tmax
#########################################################################
def LagFitFunc(Params,w1,wrf,lf,time,AlignMag="auto",R2eff_flag=0,kR1p=None):

    # Unpack Parameters
    pB,pC,dwB,dwC,kexAB,kexAC,kexBC,R1,R1b,R1c,R2,R2b,R2c = Params
    pA = 1. - (pB + pC)

    ################################
    ##### Pre-run Calculations #####
    ################################
    # Convert w1, wrf to rad/sec from Hz
    w1 = w1 * 2. * pi
    wrf = wrf * 2. * pi
    #Convert dw from ppm to rad/s
    dwB = dwB * lf * 2. * pi # dw(ppm) * base-freq (eg 151 MHz, but just 151) * 2PI, gives rad/s
    dwC = dwC * lf * 2. * pi
    #Define forward/backward exchange rates
    k12 = kexAB * pB / (pB + pA)
    k21 = kexAB * pA / (pB + pA)
    k13 = kexAC * pC / (pC + pA)
    k31 = kexAC * pA / (pC + pA)
    if kexBC != 0.:
        k23 = kexBC * pC / (pB + pC)
        k32 = kexBC * pB / (pB + pC)
    else:
        k23 = 0.
        k32 = 0.

    # Calculate pertinent frequency offsets/etc for alignment and projection
    lOmegaA, lOmegaB, lOmegaC, uOmega1, uOmega2, uOmega3, uOmegaAvg,\
    delta1, delta2, delta3, deltaAvg, theta1, theta2, theta3, thetaAvg = \
                            AlignMagVec(w1, wrf, pA, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, "avg")

    # Effective field in the rotating frame
    weA = sqrt(delta1**2. + w1**2.)
    weB = sqrt(delta2**2. + w1**2.)
    weC = sqrt(delta3**2. + w1**2.)
    weAvg = sqrt(deltaAvg**2. + w1**2.)

    if pC == 0.:
        # Laguerre 2-state. See Massi&Palmer Chem Rev 2006, eqn #34
        Rex_1 = sin(thetaAvg)**2. * pA * pB * dwB**2. * kexAB
        Rex_2 = ((weA**2. * weB**2.) / weAvg**2.) + kexAB**2.
        Rex_3 = sin(thetaAvg)**2. * pA * pB * dwB**2.
        Rex_4 = 1. + (2*kexAB**2. * (pA*weA**2. + pB*weB**2.))/(weA**2.*weB**2. + weAvg**2.*kexAB**2.)
        Rex = Rex_1 / (Rex_2 - Rex_3*Rex_4)
        R1p = (R1 * cos(thetaAvg)**2.) + (R2 * sin(thetaAvg)**2.) + Rex

    elif pC != 0. and kexBC == 0.:
        # # Laguerre 3-state. See Massi&Palmer Chem Rev 2006, eqn #??
        # # AB Leg
        # Rex_1a = sin(thetaAvg)**2. * pA * pB * dwB**2. * kexAB
        # Rex_2a = ((weA**2. * weB**2.) / weAvg**2.) + kexAB**2.
        # Rex_3a = sin(thetaAvg)**2. * pA * pB * dwB**2.
        # Rex_4a = 1. + (2*kexAB**2. * (pA*weA**2. + pB*weB**2.))/(weA**2.*weB**2. + weAvg**2.*kexAB**2.)
        # # AC Leg
        # Rex_1b = sin(thetaAvg)**2. * pA * pC * dwC**2. * kexAC
        # Rex_2b = ((weA**2. * weC**2.) / weAvg**2.) + kexAC**2.
        # Rex_3b = sin(thetaAvg)**2. * pA * pC * dwC**2.
        # Rex_4b = 1. + (2*kexAC**2. * (pA*weA**2. + pC*weC**2.))/(weA**2.*weC**2. + weAvg**2.*kexAC**2.)
        # Rex = (Rex_1a / (Rex_2a - Rex_3a * Rex_4a)) + (Rex_1b / (Rex_2b - Rex_3b * Rex_4b))
        # R1p = (R1 * cos(thetaAvg)**2.) + (R2 * sin(thetaAvg)**2.) + Rex

        # Palmer general 3-state w/ no minor. See Massi&Palmer Chem Rev 2006, eqn #40
        # AB Leg
        Rex_1a = k12 * dwB**2.
        Rex_2a = weB**2. + k21**2.
        # AC Leg
        Rex_1b = k13 * dwC**2.
        Rex_2b = weC**2. + k31**2.

        Rex = sin(thetaAvg)**2. * ((Rex_1a/Rex_2a) + (Rex_1b/Rex_2b))
        R1p = (R1 * cos(thetaAvg)**2.) + (R2 * sin(thetaAvg)**2.) + Rex

    elif pC != 0. and kexBC != 0.:
        print "Cannot yet fit Laguerre 3-state with minor exchange. Try Bloch-McConnell."
        sys.exit(-1)

    # Check to make sure R1rho is not-NaN
    if isnan(R1p) == True:
        if R2eff_flag == 0:
            return(0.)
        elif R2eff_flag == 1:
            R2eff = None
            return array([0., 0.])
    # If R1p is not NaN
    else:
        # If flagged to return just R1p
        if R2eff_flag == 0:
            return R1p
        # Else, calculate R2eff and return both R1p and R2eff
        elif R2eff_flag == 1:
            # If on-resonance, thetaAvg = pi/2
            if deltaAvg == 0.: thetaAvg = pi/2.
            R2eff = (R1p/sin(thetaAvg)**2.) - (R1/(tan(thetaAvg)**2.))
            return array([R1p, R2eff])

#########################################################################
# Fitting function BM Simulation Routine (using 3-state matrix)
#  Returns R1ho and R2eff
#   Params = vector of fitted parameters to be unpacked
#            Parameters in vector can vary
#   w1 = SLP (given in Hz, converted to rad/s for calcs later on)
#   wrf = Offset from carrier
#         (given in "corrected" Hz, converted to rad/s later for calcs)
#   lf = Larmor freq (MHz, to calc dw from ppm)
#   time = vector of time increments (sec) from Tmin-Tmax
#   AlignMag = How to align magnetization / project
#              Auto = calculates slow or fast exchange, aligns gs or avg
#              Avg = aligns along average
#              GS  = Aligns along ground-state
#   R2eff_flag = 0 : returns R1p, 1 : returns R1p+R2eff
#   kR1p = known R1rho value, if known, will be used to calculate Tmax
#########################################################################
def BMFitFunc(Params,w1,wrf,lf,time,AlignMag="auto",R2eff_flag=0,kR1p=None):

    # Estimate maximum Trelax needed to efficiently calculate a 2-point exponential decay
    #  Use known R1rho value if it is given, as 1/R1p will give int decay to ~0.36  
    if kR1p is not None:
        tmax = 1./kR1p
    # Else, just use maximum in time delay (might be non-optimal)
    else:
        tmax = time.max()

    # Unpack Parameters
    pB,pC,dwB,dwC,kexAB,kexAC,kexBC,R1,R1b,R1c,R2,R2b,R2c = Params
    pA = 1. - (pB + pC)

    ################################
    ##### Pre-run Calculations #####
    ################################
    # Convert w1, wrf to rad/sec from Hz
    w1 = w1 * 2. * pi
    wrf = wrf * 2. * pi
    #Convert dw from ppm to rad/s
    dwB = dwB * lf * 2. * pi # dw(ppm) * base-freq (eg 151 MHz, but just 151) * 2PI, gives rad/s
    dwC = dwC * lf * 2. * pi
    #Define forward/backward exchange rates
    k12 = kexAB * pB / (pB + pA)
    k21 = kexAB * pA / (pB + pA)
    k13 = kexAC * pC / (pC + pA)
    k31 = kexAC * pA / (pC + pA)
    if kexBC != 0.:
        k23 = kexBC * pC / (pB + pC)
        k32 = kexBC * pB / (pB + pC)
    else:
        k23 = 0.
        k32 = 0.

    # Calculate pertinent frequency offsets/etc for alignment and projection
    lOmegaA, lOmegaB, lOmegaC, uOmega1, uOmega2, uOmega3, uOmegaAvg,\
    delta1, delta2, delta3, deltaAvg, theta1, theta2, theta3, thetaAvg = \
                            AlignMagVec(w1, wrf, pA, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, AlignMag)

    #Calculate initial magnetization
    Ma = pA*lOmegaA # ES1
    Mb = pB*lOmegaB # GS
    Mc = pC*lOmegaC # ES2

    # Magnetization matrix
    Ms = MatrixBM3(k12,k21,k13,k31,k23,k32,delta1,delta2,delta3,
                   w1, R1, R2, R1b, R1c, R2b, R2c)

    # Initial magnetization of GS (Mb), ES1 (Ma), ES2 (Mc)
    M0 = array([Ma[0],Mb[0],Mc[0],Ma[1],Mb[1],Mc[1],Ma[2],Mb[2],Mc[2]], float64)

    #################################################################
    #### Calculate Evolution of Magnetization for Fitting Func ######
    #################################################################

    # # Project magnetization along average state in Jameson way
    # #   Note: this PeffVec + Fit Exp gives nearly identical
    # #         values to Flag 2 way
    # PeffVec = asarray([AltCalcMagT(x,M0,Ms,lOmegaA,lOmegaB,lOmegaC,w1,wrf) for x in time])
    # popt, pcov = curve_fit(ExpDecay,
    #                        time,
    #                        PeffVec,
    #                        (1., 5.))
    # R1p = popt[1]

    # Calculate effective magnetization at Tmax and Tmin, respectively
    #  Returns floats corresponding to magnetization projected back along Meff at time T   
    magMin,magMax = AltCalcMagT(tmax,M0,Ms,lOmegaA,lOmegaB,lOmegaC,w1,wrf), AltCalcMagT(time[0],M0,Ms,lOmegaA,lOmegaB,lOmegaC,w1,wrf)
    # Check to make sure magnetization at Tmax is not <= 0, would give errorneous result
    if magMin <= 0.:
        # Project magnetization along average state in Jameson way
        #   Note: this PeffVec + Fit Exp gives nearly identical
        #         values to Flag 2 way
        PeffVec = asarray([AltCalcMagT(x,M0,Ms,lOmegaA,lOmegaB,lOmegaC,w1,wrf) for x in time])
        popt, pcov = curve_fit(ExpDecay,
                               time,
                               PeffVec,
                               (1., 5.))
        R1p = popt[1]

    ## R1rho Calc Opt #1
    # Kay method (Korhznev, JACS, 2004) Solve with 2-pts
    # R1rho = -1/Tmax*ln(I1/I0), where I1 is Peff at Tmax
    #   NOTE: If use this, don't calc Peff above
    # Kay Method with Jameson Alt. Calc. Mag. T
    else:
        # If magnetization at Tmax is < 0,
        # Means odd exponential decay
        # In this case, walk backwards along time vector
        #  until the time where magMin(t) > 0
        #  Then solve for R1rho with new min magnetization
        R1p = -1./tmax*log(magMin/magMax)

    # # Faster alternative, assumes mag at T=0 is 0.0!
    # R1p = -1./time[-1]*log(AltCalcMagT(time[-1],M0,Ms,lOmegaA,lOmegaB,lOmegaC,w1,wrf))

    if isnan(R1p) == True:
        if R2eff_flag == 0:
            return(0.)
        elif R2eff_flag == 1:
            R2eff = None
            return array([0., 0.])
    # If R1p is not NaN
    else:
        # If flagged to return just R1p
        if R2eff_flag == 0:
            return R1p
        # Else, calculate R2eff and return both R1p and R2eff
        elif R2eff_flag == 1:
            # If on-resonance, thetaAvg = pi/2
            if deltaAvg == 0.: thetaAvg = pi/2.
            R2eff = (R1p/sin(thetaAvg)**2.) - (R1/(tan(thetaAvg)**2.))
            return array([R1p, R2eff])

#########################################################################
# Fitting function BM Simulation Routine (using 3-state matrix)
#   Params = Dictionary of parameter names and associate values
#            must have: pb, pc, kexab, kexac, kexbc, r1, r1b, r1c
#                       r2, r2b, r2c, dwb, dwc, lf, alignmag keys
#   w1 = SLP (given in Hz, converted to rad/s for calcs later on)
#   wrf = Offset from carrier
#         (given in "corrected" Hz, converted to rad/s later for calcs)
#   time = vector of time increments (sec) from Tmin-Tmax
#   err = error percentage to noise corrupt magnetization by
# Returns:
# o R1rho array
# -- 0 R1rho
# -- 1 R1rho_err
# -- 2 R2eff
# -- 3 R2eff_err
# -- 4 Preexponential
# o Mag Sim array
# -- 0  Peff : effective mag proj along avg effective
# -- 1  PeffA : mag proj along A-state effective
# -- 2  PeffB : mag proj along B-state effective
# -- 3  PeffC : mag proj along C-state effective
# -- 4  Mxa : x-comp of A-state at time t
# -- 5  Mya : y-comp of A-state at time t
# -- 6  Mza : z-comp of A-state at time t
# -- 7  Mxb : x-comp of B-state at time t
# -- 8  Myb : y-comp of B-state at time t
# -- 9  Mzb : z-comp of B-state at time t
# -- 10 Mxc : x-comp of C-state at time t
# -- 11 Myc : y-comp of C-state at time t
# -- 12 Mzc : z-comp of C-state at time t
# o Eigenvalue array
# -- 0 w1-ax : eigenval 1 of state A, x-comp
# -- 1 w2-ay : eigenval 2 of state A, y-comp
# -- 2 w3-az : eigenval 3 of state A, z-comp
# -- 3 w4-bx : eigenval 1 of state B, x-comp
# -- 4 w5-by : eigenval 2 of state B, y-comp
# -- 5 w6-bz : eigenval 3 of state B, z-comp
# -- 6 w7-cx : eigenval 1 of state C, x-comp
# -- 7 w8-cy : eigenval 2 of state C, y-comp
# -- 8 w9-cz : eigenval 3 of state C, z-comp
#########################################################################
def BMSim(ParD, wrf, w1, time, dec_err=0.0, dec_mc=500, rho_err=0.0, rho_mc=500):
    # Numpy array to store mag vectors
    magVecs = zeros(13)
    # Numpy array to store eigenvalues
    eigVals = zeros(9)
    # Decay sim flag - 2pt or monoexp fit
    decFlag = False
    # Check to see if vdlist is defined
    if len(time) > 2:
        decFlag = True
        kR1p = 2.
        tmax = 1./kR1p
    else:
        kR1p = 2.
        # Estimate maximum Trelax needed to efficiently calculate a 2-point exponential decay
        #  Use known R1rho value if it is given, as 1/R1p will give int decay to ~0.36  
        if kR1p is not None:
            tmax = 1./kR1p

    # Unpack Parameters
    pB,pC,dwB,dwC = ParD['pb'], ParD['pc'], ParD['dwb'], ParD['dwc']
    kexAB,kexAC,kexBC = ParD['kexab'], ParD['kexac'], ParD['kexbc']
    R1,R1b,R1c = ParD['r1'], ParD['r1b'], ParD['r1c']
    R2,R2b,R2c = ParD['r2'], ParD['r2b'], ParD['r2c']
    lf, AlignMag = ParD['lf'], ParD['alignmag']
    pA = 1. - (pB + pC)

    ################################
    ##### Pre-run Calculations #####
    ################################
    # Convert w1, wrf to rad/sec from Hz
    w1 = w1 * 2. * pi
    wrf = wrf * 2. * pi
    #Convert dw from ppm to rad/s
    dwB = dwB * lf * 2. * pi # dw(ppm) * base-freq (eg 151 MHz, but just 151) * 2PI, gives rad/s
    dwC = dwC * lf * 2. * pi
    #Define forward/backward exchange rates
    k12 = kexAB * pB / (pB + pA)
    k21 = kexAB * pA / (pB + pA)
    k13 = kexAC * pC / (pC + pA)
    k31 = kexAC * pA / (pC + pA)
    if kexBC != 0.:
        k23 = kexBC * pC / (pB + pC)
        k32 = kexBC * pB / (pB + pC)
    else:
        k23 = 0.
        k32 = 0.

    # Calculate pertinent frequency offsets/etc for alignment and projection
    lOmegaA, lOmegaB, lOmegaC, uOmega1, uOmega2, uOmega3, uOmegaAvg,\
    delta1, delta2, delta3, deltaAvg, theta1, theta2, theta3, thetaAvg = \
                            AlignMagVec(w1, wrf, pA, pB, pC, dwB, dwC, kexAB, kexAC, kexBC, AlignMag)

    #Calculate initial magnetization
    Ma = pA*lOmegaA # ES1
    Mb = pB*lOmegaB # GS
    Mc = pC*lOmegaC # ES2

    # Magnetization matrix
    Ms = MatrixBM3(k12,k21,k13,k31,k23,k32,delta1,delta2,delta3,
                   w1, R1, R2, R1b, R1c, R2b, R2c)

    # Initial magnetization of GS (Mb), ES1 (Ma), ES2 (Mc)
    M0 = array([Ma[0],Mb[0],Mc[0],Ma[1],Mb[1],Mc[1],Ma[2],Mb[2],Mc[2]], float64)

    #################################################################
    #### Calculate Evolution of Magnetization for Fitting Func ######
    #################################################################

    if decFlag == True:
        # Calculate evolving magnetization at a given time increment
        # Returns array of projected magnetizations and indv components of mag
        # Col0 = Peff - mag projected along average
        # Col1 = Peff_err, if any
        # Col2,3,4 = PeffA,B,C - Projected along respective states
        # Col5,6,7 = Mxa, Mya, Mza - x-comps of indv states
        # Col8,9,10 = Mxb, Myb, Mzb
        # Col11,12,13 = Mxc, Myc, Mzc
        # Col14 = time
        magVecs = asarray([SimMagVecs(x,M0,Ms,lOmegaA,lOmegaB,lOmegaC,w1,wrf) for x in time])
        # Append time to vectors
        magVecs = append(magVecs, time[:,None], axis=1)

        ## -- Monoexp Decay Error Corruption -- ##
        if dec_err != 0.0:
            # MC error number
            mcnum = dec_mc
            err = magVecs[:,0].max() * dec_err
            # Get normal distributions given error scaled to max int value
            # Generates a 2xN array of error corrupted Peff and Peff_err given
            #  a normal fit to the mcnum of random values
            # tp = array([normDist.fit(normal(x, err, size=mcnum)) for x in magVecs[:,0]])
            tp = array([normal(x, err, size=mcnum) for x in magVecs[:,0]])
             # Get mu and sigma for plots
            #  Get mu from first random normal selection of Peff values
            magVecs[:,0] = tp[:,0]
            # magVecs[:,0] = tp.mean(axis=1)
            magVecs[:,1] = tp.std(axis=1)

        # Calculate eigenvalues for export
        # append offset and slp (Hz) to front of eigenvalues
        eigVals = array([wrf/(2.*pi), w1/(2.*pi)])
        eigVals = append(eigVals, matrix_exponential(Ms, w1, wrf, 0.0, EigVal=True)[1])

        # If mag vecs are not nan
        if not isnan(magVecs.sum()):
            ## -- Monoexp fitting -- ##
            # If decay noise corruption non-zero, noise corrupt fitted R1rhos
            if dec_err != 0.0:
                # Weighted fit to get best R1rho value
                popt, pcov = curve_fit(ExpDecay, time, magVecs[:,0], (1., R1), sigma=magVecs[:,1])
                # MC error generation of R1ho from noise corrupted intensities
                popts = array([curve_fit(ExpDecay, time, x, (1., R1))[0] for x in tp.T])
                preExp, R1p, R1p_err = popt[0], popt[1], popts.std(axis=0)[1]

            # If no decay corruption, simply fit for R1rho
            else:
                popt, pcov = curve_fit(ExpDecay, time, magVecs[:,0], (1., R1))
                R1p = popt[1]
                R1p_err = 0.0
                preExp = popt[0]

        else:
            R1p = 0.0
            R1p_err = 0.0
            preExp = 1.
    else:
        # Calculate effective magnetization at Tmax and Tmin, respectively
        #  Returns floats corresponding to magnetization projected back along Meff at time T   
        magMin,magMax = AltCalcMagT(tmax,M0,Ms,lOmegaA,lOmegaB,lOmegaC,w1,wrf),\
                        AltCalcMagT(time[0],M0,Ms,lOmegaA,lOmegaB,lOmegaC,w1,wrf)

        # Check to make sure magnetization at Tmax is not <= 0, would give errorneous result
        if magMin <= 0.:
            # Project magnetization along average state in Jameson way
            #   Note: this PeffVec + Fit Exp gives nearly identical
            #         values to Flag 2 way
            PeffVec = asarray([AltCalcMagT(x,M0,Ms,lOmegaA,lOmegaB,lOmegaC,w1,wrf) for x in time])
            popt, pcov = curve_fit(ExpDecay,
                                   time,
                                   PeffVec,
                                   (1., 5.))
            R1p = popt[1]

        ## R1rho Calc Opt #1
        # Kay method (Korhznev, JACS, 2004) Solve with 2-pts
        # R1rho = -1/Tmax*ln(I1/I0), where I1 is Peff at Tmax
        #   NOTE: If use this, don't calc Peff above
        # Kay Method with Jameson Alt. Calc. Mag. T
        else:
            # If magnetization at Tmax is < 0,
            # Means odd exponential decay
            # In this case, walk backwards along time vector
            #  until the time where magMin(t) > 0
            #  Then solve for R1rho with new min magnetization
            R1p = -1./tmax*log(magMin/magMax)

    if isnan(R1p) == True:
        return array([0., 0., 1.]), magVecs, eigVals
    # If R1p is not NaN
    else:
        ## -- R1rho Direct Error Corruption -- ##
        if rho_err != 0.:
            tv = normal(R1p, R1p*rho_err, size=rho_mc)
            # Pick mean R1rho from first random normal distribution
            R1p = tv[0]
            R1p_err = tv.std()

        # Calculate R2eff - take on-res in to account
        # If on-resonance, thetaAvg = pi/2
        if deltaAvg == 0.: thetaAvg = pi/2.
        # Propagate error in R2eff, if applicable
        if R1p_err == 0.0:
            R2eff = (R1p/sin(thetaAvg)**2.) - (R1/(tan(thetaAvg)**2.))
            R2eff_err = 0.0
        else:
            R2eff = (ufloat(R1p, R1p_err)/umath.sin(thetaAvg)**2.) - (R1/(umath.tan(thetaAvg)**2.))
            R2eff_err = R2eff.std_dev
            R2eff = R2eff.n

        return array([R1p, R1p_err, R2eff, R2eff_err, preExp]), magVecs, eigVals

#########################################################################
# Normalize a vector with lambda function #
#########################################################################
def normalize(vec):
    return (lambda norm: [x/norm for x in vec])(sum(x**2 for x in vec) **0.5)

#########################################################################
# BM 3-state Matrix                                                     #
# sub-divided into: exchange, delta, w1, intrinsic relaxataion matrices #
# Returns a combined Matrix, M = K + Der + W + R                        #
#########################################################################
def MatrixBM3(k12,k21,k13,k31,k23,k32,delta1,delta2,delta3,
              w1, R1, R2, R1b, R1c, R2b, R2c):

    # Exchange Matrix (exchange rates)
    #  See Trott & Palmer JMR 2004, n-site chemical exchange
    K = array([[-k12 -k13, k21, k31, 0., 0., 0., 0., 0., 0.],
               [k12, -k21 - k23, k32, 0., 0., 0., 0., 0., 0.],
               [k13, k23, -k31 - k32, 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., -k12 -k13, k21, k31, 0., 0., 0.],
               [0., 0., 0., k12, -k21 - k23, k32, 0., 0., 0.],
               [0., 0., 0., k13, k23, -k31 - k32, 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., -k12 -k13, k21, k31],
               [0., 0., 0., 0., 0., 0., k12, -k21 - k23, k32],
               [0., 0., 0., 0., 0., 0., k13, k23, -k31 - k32]], float64)

    # Delta matrix (offset and population)
    Der = array([[0., 0., 0., -delta1, 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., -delta2, 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0., -delta3, 0., 0., 0.],
                 [delta1, 0., 0., 0., 0., 0., 0., 0., 0.],
                 [0., delta2, 0., 0., 0., 0., 0., 0., 0.],
                 [0., 0., delta3, 0., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0., 0., 0., 0., 0.]], float64)

    # Spinlock power matrix (SLP/w1)
    #  Matrix is converted to rad/s here
    W = array([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., 0.], 
               [0., 0., 0., 0., 0., 0., -w1, 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., -w1, 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., -w1],
               [0., 0., 0., w1, 0., 0., 0., 0., 0.], 
               [0., 0., 0., 0., w1, 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., w1, 0., 0., 0.]], float64)

    # Intrinsic rate constant matrix (R1 and R2)
    R = array([[-R2, 0., 0., 0., 0., 0., 0., 0., 0.],
               [0., -R2b, 0., 0., 0., 0., 0., 0., 0.],
               [0., 0., -R2c, 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., -R2, 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., -R2b, 0., 0., 0., 0.], 
               [0., 0., 0., 0., 0., -R2c, 0., 0., 0.], 
               [0., 0., 0., 0., 0., 0., -R1, 0., 0.],
               [0., 0., 0., 0., 0., 0., 0., -R1b, 0.],
               [0., 0., 0., 0., 0., 0., 0., 0., -R1c]], float64) 

    return(K + Der + W + R)

#########################################################################
# Calculate the exact matrix exponential using the eigenvalue decomposition approach.
#  Returns only real components of matrix, A, if it is complex
#
# A is a square matrix we wish to have the matrix exponential of
# If A has a full set of eigenvectors V, then A can be factorized as:
#  A = V.D.V^-1,
#   where D is the diagonal matrix whose diagonal elements, di, are the
#   eigenvalues of A. 
# e^A is then given by: 
#  e^A = V.e^D.V^-1
#   where e^D is the diagonal matrix whose ith diag elem is e^di
# Code modified from 'nmr-relax' program. http://www.nmr-relax.com/
#########################################################################
def matrix_exponential(A, w1, wrf, t, EigVal=False):
    """Calculate the exact matrix exponential using the eigenvalue decomposition approach.

    @param A:   The square matrix to calculate the matrix exponential of.
    @type A:    numpy rank-2 array
    @return:    The matrix exponential.  This will have the same dimensionality as the A matrix.
    @rtype:     numpy rank-2 array
    """

    # Handle nan or inf elements of matrix
    if isnan(A).any() == True or isinf(A).any() == True:
        A = nan_to_num(A) # How to handle this?
        print "NaN or Inf elements in BM Relaxation Matrix. Check parameter bounds or initial starting state."

    # The eigenvalue decomposition.
    # W are the real and imaginary eigenvalues of the matrix A
    # V are all eigenvectors of the matrix A
    W, V = eig(A)

    # Strip non-real components of complex eigenvalues
    W_orig = asarray(list(W))
    if iscomplexobj(W):
        W = W.real

    # Calculate the exact exponential.
    # Solution is of the form:
    #  e^A = V.e^D.V^-1
    #   where D is the diag matrix of the eigenvalues of A
    eA = dot(dot(V, diag(exp(W))), inv(V))
    if EigVal == False:
        return eA.real
    else:
        return eA.real, W

#########################################################################
# Used to fit effective bulk magnetization vector dot product
#  decay as a function of time.
#  Peff vs. time
#########################################################################
def ExpDecay(x,a,b):
    return a*exp(-b*x)

#########################################################################
# Evolve magnetization of Ms starting from M0 with increment of time
# Takes the matrix exponential of Ms and time_increment
# Calculates the dot product of the dot product of M0 and (M0, exp(Ms*t))
#########################################################################
def CalcMagT(time_incr, M0, Ms):
    # preMag = dot((matrix_exponential(Ms*time_incr)), M0)
    # Ma = dot(M0[:3],preMag[:3])
    # Mb = dot(M0[3:6],preMag[3:6])
    # Mc = dot(M0[6:],preMag[6:])
    magatT = dot(M0, dot((matrix_exponential(Ms*time_incr,w1,wrf)), M0))
    return(magatT)

def SimMagVecs(dt,M0,Ms,lOmegaA,lOmegaB,lOmegaC,w1,wrf):
    # Sim mag at time increment
    M = dot((matrix_exponential(Ms*dt, w1, wrf, dt)), M0)
    Mxa = M[0]
    Mxb = M[1]
    Mxc = M[2]
    Mya = M[3]
    Myb = M[4]
    Myc = M[5]
    Mza = M[6]
    Mzb = M[7]
    Mzc = M[8]
    # Project effective mag along indv effective lOmegas
    PeffA = dot(vstack((Mxa,Mya,Mza)).T, lOmegaA)[0]
    PeffB = dot(vstack((Mxb,Myb,Mzb)).T, lOmegaB)[0]
    PeffC = dot(vstack((Mxc,Myc,Mzc)).T, lOmegaC)[0]
    # Project mag along average effective
    Peff = PeffA + PeffB + PeffC
    Peff_err = 0.0 # Placeholder

    return array([Peff, Peff_err, PeffA, PeffB, PeffC,
                     Mxa, Mya, Mza,
                     Mxb, Myb, Mzb,
                     Mxc, Myc, Mzc])

#########################################################################
# Alternative evolution of magnetization of Ms starting from M0 with increment of time
# Takes the matrix exponential of Ms and time_increment
# Break evolved matrix into indiv. coordinates of A,B,C
# Stack, and project back along respective effective mag states
#########################################################################
def AltCalcMagT(time_incr,M0,Ms,lOmegaA,lOmegaB,lOmegaC,w1,wrf):
    M = dot((matrix_exponential(Ms*time_incr,w1,wrf,time_incr)), M0)
    Mxa = M[0]
    Mxb = M[1]
    Mxc = M[2]
    Mya = M[3]
    Myb = M[4]
    Myc = M[5]
    Mza = M[6]
    Mzb = M[7]
    Mzc = M[8]

    # Calculate effective magnetization decay from composition
    # of all vectors projected along lOmegaA,B,C
    Peff = (dot(vstack((Mxa,Mya,Mza)).T, lOmegaA)
            +
            dot(vstack((Mxb,Myb,Mzb)).T, lOmegaB)
            +
            dot(vstack((Mxc,Myc,Mzc)).T, lOmegaC))

    return Peff[0]
