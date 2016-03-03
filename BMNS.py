#########################################################################
# Bloch-McConnell 2-/3-state Fitting Program v1.25
#  Beta version.
#  Isaac Kimsey 02-11-2016
#
# External Dependencies: numpy, matplotlib, scipy, leastsqbound, uncertainties
#########################################################################

### Libraries Needed: numpy, matplotlib, scipy, ampgo, leastsqbound ###
# General libraries
import os, sys
import subprocess
import datetime

### Local BMNS Libraries ###
import BMNS_FitData as fd
import BMNS_SimR1p as sim
import BMNS_SimFits as simf
import BMNS_Errors as bme
import BMNS_AMPGO as ampgo  # Global fitting
import BMNS_MathFuncs as mf
import BMNS_Stats as sf
### Direct numpy imports ###
from numpy import absolute, array, asarray
from numpy import diag
from numpy import isinf, isnan
from numpy import linspace
from numpy import nan_to_num
from numpy import sqrt
from numpy import zeros
### Scipy/Other General Fitting Algs ###
from scipy.optimize import minimize # Local minimum, see Nelder-Mead
from scipy.optimize import least_squares
# Uncertainties calculations
from uncertainties import umath
from uncertainties import ufloat

#########################################################################
# Create a folder if it does not already exist #
#########################################################################
def makeFolder(pathToFolder):
  if not os.path.exists(pathToFolder): 
    os.makedirs(pathToFolder) 

def Main():
  #########################################################################
  # Bloch-McConnell 2-/3-state R1rho Fitting
  #########################################################################
  #  arg1 '-fit'
  #  arg2 Parameter Text File
  #       - See Example\ParExample.txt
  #  arg3 Parent directory of R1rho.csv data files
  #        corresponding to names in pars file.
  #       - Each csv is ordered:
  #          Col 1: Offset (corrected, Hz)
  #          Col 2: SLP (Hz)
  #          Col 3: R1rho (s-1)
  #          Col 4: R1rho err (s-1)[optional]
  #      If first row is text, will delete first row
  #       and first column, and shift col 2-5 to
  #       col 1-4, as above.
  #  arg4 Output directory for fit data [Optional]
  #         If not given, will generate folder in
  #         parent data directory.
  #-----------------------------------------------------------------------#
  if "fit" in sys.argv[1].lower():
    ## Check for Errors in Passed Arguments ##
    #  This function will terminate program if
    #   not all needed arguments or files are present
    bme.CheckArgs(curDir, argc, sys.argv)

    # Bool to determine if program succeeded in setup
    errBool = False
    # String for error messages.
    #  if runBool is ultimately False, these error messages will show up.
    retMsg = "" 

    ## Define input/output Paths ##
    parPath = os.path.join(curDir, sys.argv[2])  # Path to input parameters
    dataPath = os.path.join(curDir, sys.argv[3]) # Path to parent dir of data
    if argc == 5:                                # Handle output path (if exists)
      outPath = os.path.join(curDir, sys.argv[4])
      makeFolder(outPath)
    else: # If no output path given, create one in the data folder given above.
      outPath = os.path.join(curDir, dataPath, "Output/")
      makeFolder(outPath)
    # Make copies of input data and parameters
    copyPath = os.path.join(outPath, "Copies/")
    makeFolder(copyPath)
    subprocess.call(["cp", parPath, os.path.join(copyPath, "copy-input.txt")])

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Parse given parameter input text file
    # Generate class objects for each corresponding given set of data 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ## Handle input parameter file ##
    # Generate a Data Parsing Class Object to handle input data
    pInp = fd.Parse()
    # Parse given input file to semi-raw format
    #  semi-raw can be handled by Parameters class
    pInp.ParseInp(parPath)
    # Check that given input parameter file is valid
    #  if not valid, set bool to false and return message
    errBool, tMsg = pInp.CheckPars(dataPath)
    # Perform Error checks on data parsing
    #  If fails, exit program
    retMsg += tMsg
    bme.HandleErrors(errBool, retMsg)

    ## Make Global class object to handle sub Fit class objects ##
    gl = fd.Global()

    # Generate fit objects and give them to Global class object
    #  Use index to assign FitNum to each Fit object
    for idx,inp in enumerate(pInp.ParInp):
      gl.gObs.append(fd.Fits(idx))

    ## Loop over fit objects in global class object
    #  Read in and convert parameter and data files
    for i in gl.gObs:
      # Convert semi-raw parameter data to Parameter self.Pars dictionary
      #  Also passes Variable names so that they can be seen to exist
      i.ConvertPars(pInp.ParInp[i.FitNum])
      # Parse and check raw R1rho data given the name in self.Pars
      #  of the corresponding Fit object
      errBool, tMsg = pInp.ParseData(dataPath, i.name)
      retMsg += tMsg
      # Copy original data
      subprocess.call(["cp", os.path.join(dataPath, i.name + ".csv"),
                       os.path.join(copyPath, "copy-" + i.name + ".csv")])
      # Check for any errors in parsing data
      bme.HandleErrors(errBool, retMsg)
      # Convert semi-raw data to Data class objects
      i.ConvertData(pInp.DataInp[i.FitNum])

    ## Grab fit types
    gl.GrabFitType(pInp.FitType)

    ## Generate global P0, bounds, shared and fix value arrays, dicts and sets
    gl.MapGlobalP0()

    ## Calculate total degrees of freedom in data
    gl.CalcDOF()

    ## Make local, global and polished folders in outpath
    outLocal = os.path.join(outPath, "Local")
    outGlobal = os.path.join(outPath, "Global")
    outPolish = os.path.join(outPath, "Polish")
    # Make statistics output folder paths
    pstatsP = os.path.join(outPolish, "Matrices")
    lstatsP = os.path.join(outLocal, "Matrices")
    # Make output folders depending on fit type    
    if gl.FitType == "global":
      makeFolder(outGlobal)
      makeFolder(outPolish)
      ## Make matrices folder path
      makeFolder(pstatsP) # Polished stats

    else:
      makeFolder(outLocal)
      makeFolder(lstatsP) # Local stats

    ## Create 'GraphOut' class object to handle graphing data
    #  this class object will handle graphing the fitted data
    grph = fd.GraphFit()

    # Last chance to catch errors
    if errBool == False:
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Fit Bloch-McConnell (2-/3-state) to experimental or simulated data
    #  Option to fit using R1rho error
    #   inData = 
    #   P0 = Vector of initial guess for parameters for fit
    #        contents vary depending on parameters being fit
    #   time = vector of time increments (sec) from Tmin-Tmax
    #   lf = Larmor freq (MHz, to calc dw from ppm)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      #---------------------------#---------------------------#
      # Chi-square local function used for fitting algorithms
      #  Returns chi-squares
      #   with error: chi-sq = ((R1p_sim - R1p)/(R1p_err))^2
      #   without error: chi-sq = (R1p_sim - R1p)^2 / R1p
      #---------------------------#---------------------------#
      def chi2(Params):
        # Expected R1rho based on simulations
        chisq = 0.

        # Loop over all Fit objects in Global class object
        for ob in gl.gObs:
          # Unpack data
          Offs, Spinlock = ob.R1pD[:,0], ob.R1pD[:,1]
          R1p, R1p_e = ob.R1pD[:,2], ob.R1pD[:,3]
          # Unpack parameters
          lf = ob.lf
          # Parse 'Params' down to only the local values
          #  and handle shared and fix parameters.
          tPars = gl.UnpackgP0(Params, ob)
          # If error in value, chisq = (o-e/err)^2
          if len(R1p_e) > 1:
            chisq += sum([((sim.BMFitFunc(tPars,SL,-1.*OF,lf,ob.time,ob.AlignMag,0,kR1p)-kR1p)/err)**2.
                           for (SL,OF,kR1p,err) in zip(Spinlock,Offs,R1p,R1p_e)])
          # If no error in value, chisq = (o-e)^2/e
          else:
             chisq += sum([((sim.BMFitFunc(tPars,SL,-1.*OF,lf,ob.time,ob.AlignMag,0,kR1p)-kR1p)**2./kR1p)
                           for (SL,OF,kR1p,err) in zip(Spinlock,Offs,R1p,R1p_e)]) 

        ### Check for 'NaN' or 'inf' chi-square ###
        #  These are sometimes genereated when magnetization
        #   no longer decays as a simple monoexponential, and
        #   thus the log solve can be applied on a value <= 0
        #   which can return nan. Other exceptions that give
        #   inf can be related to the fitting algorthing
        # If true, returns a large number to disfavor this area.
        if isnan(chisq) == True or isinf(chisq) == True:
          chisq = 1e4 # Return bad value
        return chisq

      #---------------------------#---------------------------#
      # Residual function used for fitting algorithms
      #  Returns matrix of residuals of: (f(x) - known) / error
      #                              or:  f(x) - known
      #---------------------------#---------------------------#
      def residual(Params):#, nullData):
        # Expected R1rho based on simulations
        resid = []

        # Loop over all Fit objects in Global class object
        for ob in gl.gObs:
          # Unpack data
          Offs, Spinlock = ob.R1pD[:,0], ob.R1pD[:,1]
          R1p, R1p_e = ob.R1pD[:,2], ob.R1pD[:,3]
          # Unpack parameters
          lf = ob.lf
          # Parse 'Params' down to only the local values
          #  and handle shared and fix parameters.
          tPars = gl.UnpackgP0(Params, ob)
          # Calculate residuals using BM numerical solution
          #  if equation is specified in ob.fitEqn
          if gl.gFitEqn == "bm":
            # If error in value, residual matrix = (f(x) - obs) / err
            if len(R1p_e) > 1:
              resid += [((sim.BMFitFunc(tPars,SL,-1.*OF,lf,ob.time,ob.AlignMag,0,kR1p)-kR1p)/err)
                             for (SL,OF,kR1p,err) in zip(Spinlock,Offs,R1p,R1p_e)]
            
            # If no error in value, residual matrix = f(x) - obs
            else:
               resid += [(sim.BMFitFunc(tPars,SL,-1.*OF,lf,ob.time,ob.AlignMag,0,kR1p)-kR1p)
                             for (SL,OF,kR1p) in zip(Spinlock,Offs,R1p)]
          # Calculate residuals using Laguerre approximations
          elif gl.gFitEqn == "lag":
            # If error in value, residual matrix = (f(x) - obs) / err
            if len(R1p_e) > 1:
              resid += [((sim.LagFitFunc(tPars,SL,-1.*OF,lf,ob.time,ob.AlignMag,0,kR1p)-kR1p)/err)
                             for (SL,OF,kR1p,err) in zip(Spinlock,Offs,R1p,R1p_e)]

            # If no error in value, residual matrix = f(x) - obs
            else:
               resid += [(sim.LagFitFunc(tPars,SL,-1.*OF,lf,ob.time,ob.AlignMag,0,kR1p)-kR1p)
                             for (SL,OF,kR1p) in zip(Spinlock,Offs,R1p)]            
        resid = asarray(resid)

        ### Check for 'NaN' or 'inf' chi-square ###
        #  These are sometimes genereated when magnetization
        #   no longer decays as a simple monoexponential, and
        #   thus the log solve can be applied on a value <= 0
        #   which can return nan. Other exceptions that give
        #   inf can be related to the fitting algorthing
        # If true, replace nan/inf elements with zero (nan)
        #  or large positive (inf) or large negative (-inf) values
        if isnan(resid).any() == True or isinf(resid).any() == True:
          resid = nan_to_num(resid) # Replace nan or inf values
        return resid

      #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      # Primary fitting loop.
      #  Loops over gl.FitLoops N times (i.e. finds N times fit minima)
      #  Embedded if statement around gl.FitType dictates if the fit
      #   is carried out globally (with polish) or locally.
      #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      for lp in range(gl.FitLoops):
        if gl.FitType == "global":
          print "~~~~~~~~~~~~~~~~~ GLOBAL FIT START (%s) ~~~~~~~~~~~~~~~~~" % str(lp+1)  
          print "  (Adaptive Memory Programming for Global Optimums)  "    

          # Randomize initial guess, if flagged          
          if gl.rndStart == True:
            tP0 = gl.RandomgP0()
          else:
            tP0 = gl.gP0
          # Convert bounds array to tuple for use in AMPGO algorithm
          bnds = tuple((x,y) for x,y in zip(gl.gBnds[0], gl.gBnds[1]))
          fitted = ampgo.AMPGO(chi2, tP0, local='L-BFGS-B',
                               bounds=bnds, maxiter=5, tabulistsize=8,
                               totaliter=10, maxfunevals=5000)
          
          #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
          ### Update Fit (global) Class Objects Here ###         
          # 1. Unpack global fitted parameters
          # 2. Write out fits and reduced chi^2 (chi-sq/dof)
          # 3. Write out graphs of fitted R1rho and R2+Rex and the residuals
          #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
          for ob in gl.gObs:
            # Reduced chi-square = chi-square / (N (data points) - M (free parameters))
            redChiSq = fitted[1] / gl.dof
            # Unpack global fit param array to local values for Fit object
            gl.UnPackFits(lp+1, gl.UnpackgP0(fitted[0], ob), redChiSq, fitted[2], "global", ob)

            # Write out / append latest fit data
            gl.WriteFits(outPath, ob, lp+1, "global") 

            # Graph fitted data with trend-lines, and also export R1rho/R2eff values
            grph.WriteGraph(ob, outGlobal, lp+1, ob.time, "global")

          print "     Polish Global Fit with Levenberg-Marquardt"

          # !! For least_squares function/Lev-Mar !! #
          fitted = least_squares(residual, fitted[0], bounds = gl.gBnds, max_nfev=10000)
          
          #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
          ### Update Fit (polish) Class Objects Here ###         
          # 1. Unpack local fitted parameters
          # 2. Write out fits and reduced chi^2 (chi-sq/dof)
          # 3. Write out graphs of fitted R1rho and R2+Rex and the residuals
          #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
          for ob in gl.gObs:
            # Reduced chi-square = chi-square / (N (data points) - M (free parameters))
            chisq = chi2(fitted.x)
            redChiSq = chisq / gl.dof

            # Calculate fit error
            #   Here: Standard error of the fit is used
            fiterr,_,_,_ = sf.cStdErr(fitted.x, fitted.fun, fitted.jac, gl.dof)

            # Unpack global fit param array to local values for Fit object
            gl.UnPackFits(lp+1, gl.UnpackgP0(fitted.x, ob), redChiSq,
                          fitted.nfev, "polish", ob, errPars=gl.UnpackErr(fiterr, ob))
          
            # Write out / append latest fit data
            gl.WriteFits(outPath, ob, lp+1, "polish")       

            # Graph fitted data with trend-lines, and also export R1rho/R2eff values
            grph.WriteGraph(ob, outPolish, lp+1, ob.time, "polish")

            # Calculate fit stats
            sf.WriteStats(outPath, pstatsP, fitted, ob, gl.dof, gl.dataSize,
                          gl.freePars, chisq, redChiSq, lp+1, "polish")
        # Local Fit
        elif gl.FitType == "local":
          print "~~~~~~~~~~~~~~~~~ LOCAL FIT START (%s) ~~~~~~~~~~~~~~~~~" % str(lp+1)  
          print "                 (Levenberg-Marquardt)  " 
          
          # Randomize initial guess, if flagged
          if gl.rndStart == True:
            tP0 = gl.RandomgP0()
          else:
            tP0 = gl.gP0
          # Least_squares / Lev-Mar fit
          fitted = least_squares(residual, tP0, bounds = gl.gBnds, max_nfev=10000)

          #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
          ### Update Fit (local) Class Objects Here ###         
          # 1. Unpack local fitted parameters
          # 2. Write out fits and reduced chi^2 (chi-sq/dof)
          # 3. Write out graphs of fitted R1rho and R2+Rex and the residuals
          #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
          for ob in gl.gObs:
            # Reduced chi-square = chi-square / (N (data points) - M (free parameters))
            chisq = chi2(fitted.x)
            redChiSq = chisq / gl.dof

            # Calculate fit error
            #   Here: Standard error of the fit is used
            fiterr,_,_,_ = sf.cStdErr(fitted.x, fitted.fun, fitted.jac, gl.dof)

            # Unpack global fit param array to local values for Fit object
            gl.UnPackFits(lp+1, gl.UnpackgP0(fitted.x, ob), redChiSq,
                          fitted.nfev, "local", ob, errPars=gl.UnpackErr(fiterr, ob))
            # Write out / append latest fit data
            gl.WriteFits(outPath, ob, lp+1, "local")       

            # Graph fitted data with trend-lines, and also export R1rho/R2eff values
            grph.WriteGraph(ob, outLocal, lp+1, ob.time, FitType="local", FitEqn=gl.gFitEqn)

            # Calculate fit stats
            sf.WriteStats(outPath, lstatsP, fitted, ob, gl.dof, gl.dataSize,
                          gl.freePars, chisq, redChiSq, lp+1, "local")
        # Brute-force across parameter range
        elif "brute" in gl.FitType:
          # Keep track of reduced chi-squares mapped to P0 array
          allfits = {}
          # Loop over all P0 arrays in brute-force array
          for idx, gf in enumerate(gl.brutegP0):
            # sys.stdout.write("--- BRUTE FORCE PARAMETERS (%s of %s) ---" % (idx+1, len(gl.brutegP0)))
            sys.stdout.write("\r--- BRUTE FORCE PARAMETERS (%s of %s) ---" % (idx+1, len(gl.brutegP0)))
            sys.stdout.flush()
            # Don't let it fit, just 1 iteration
            fitted = least_squares(residual, gf, bounds = gl.gBnds, max_nfev=1)       
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            ### Update Fit (local) Class Objects Here ###         
            # 1. Unpack local fitted parameters
            # 2. Write out fits and reduced chi^2 (chi-sq/dof)
            # 3. Write out graphs of fitted R1rho and R2+Rex and the residuals
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            for ob in gl.gObs:
              # Reduced chi-square = chi-square / (N (data points) - M (free parameters))
              chisq = chi2(fitted.x)
              redChiSq = chisq / gl.dof
              # Store all the fits
              allfits[redChiSq] = gf

              # Calculate fit error
              #   Here: Standard error of the fit is used
              fiterr,_,_,_ = sf.cStdErr(fitted.x, fitted.fun, fitted.jac, gl.dof)

              # Unpack global fit param array to local values for Fit object
              gl.UnPackFits(idx+1, gl.UnpackgP0(fitted.x, ob), redChiSq,
                            fitted.nfev, "local", ob, errPars=gl.UnpackErr(fiterr, ob))
              # Write out / append latest fit data
              gl.WriteFits(outPath, ob, idx+1, "local")       

              # If flagged in input file, generate graphs for all curves
              if gl.FitType == "brutep":
                # Graph fitted data with trend-lines, and also export R1rho/R2eff values
                grph.WriteGraph(ob, outLocal, idx+1, ob.time, FitType="local", FitEqn=gl.gFitEqn)

              # Calculate fit stats
              sf.WriteStats(outPath, lstatsP, fitted, ob, gl.dof, gl.dataSize,
                            gl.freePars, chisq, redChiSq, idx+1, "local", matrices=False)
          
          # Start the last fit, from the best fit
          print "\n    Lowest red. chi-square found. Minimizing within bounds.    "
          lastval = len(gl.brutegP0) + 1
          tP0 = allfits[min(allfits.iterkeys())]
          # Least_squares / Lev-Mar fit
          fitted = least_squares(residual, tP0, bounds = gl.gBnds, max_nfev=10000)
          # fitted = least_squares(residual, tP0, max_nfev=10000)
          for ob in gl.gObs:
            # Reduced chi-square = chi-square / (N (data points) - M (free parameters))
            chisq = chi2(fitted.x)
            redChiSq = chisq / gl.dof
            # Store all the fits
            allfits[redChiSq] = gf

            # Calculate fit error
            #   Here: Standard error of the fit is used
            fiterr,_,_,_ = sf.cStdErr(fitted.x, fitted.fun, fitted.jac, gl.dof)

            # Unpack global fit param array to local values for Fit object
            gl.UnPackFits(lastval, gl.UnpackgP0(fitted.x, ob), redChiSq,
                          fitted.nfev, "local", ob, errPars=gl.UnpackErr(fiterr, ob))
            # Write out / append latest fit data
            gl.WriteFits(outPath, ob, lastval, "local")       

            # Graph fitted data with trend-lines, and also export R1rho/R2eff values
            grph.WriteGraph(ob, outLocal, lastval, ob.time, FitType="local", FitEqn=gl.gFitEqn)

            # Calculate fit stats
            sf.WriteStats(outPath, lstatsP, fitted, ob, gl.dof, gl.dataSize,
                          gl.freePars, chisq, redChiSq, lastval, "local")          

        else:
          print "Fit Type not declared properly (global or local)"

    else:
      print "----- Cannot Run Fit Because of Errors -----"
      print retMessage

  #########################################################################
  # Bloch-McConnell 2-/3-state R1rho Simulation
  #########################################################################
  #  arg1 '-sim'
  #  arg2 Parameter Text File
  #  arg3 (Optional) Specific output folder, will be made if does not exist
  #-----------------------------------------------------------------------#
  elif "sim" in sys.argv[1].lower():
    # Create parent output directory
    if len(sys.argv) >= 4:
      outPath = os.path.join(curDir, sys.argv[3])
      makeFolder(outPath)
    else:
      # Get timestamp for generating folder
      mydate = datetime.datetime.now()
      tst = mydate.strftime("Simulation_%m%d%y-%H%Mh%Ss")
      outPath = os.path.join(curDir, tst)
      makeFolder(outPath)
    # Create folder for all magnetization vectors
    outVec = os.path.join(outPath, "MagVecs")
    makeFolder(outVec)
    # Create simulation class object
    sfo = simf.SimFit()
    # Clean and handle input args
    sfo.PreSim(sys.argv)
    # Simulate R1rho values
    sfo.simFit()
    # Plot R1rho values
    sfo.plotR1p(outPath)
    # Plot R2eff values
    sfo.plotR2eff(outPath)
    # Plot onres R1rho values
    sfo.plotOnRes(outPath)
    # Plot monoexponential decays
    sfo.plotDec(outPath)
    # Write-out simulated R1rho values
    sfo.writeR1p(outPath)
    # Write-out simulated vectors and eigenvalues
    sfo.writeSimPars(outPath)
    # Write-out sim parameters
    sfo.writeVecVal(outVec, outPath)

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # 3D Magnetization Visualization
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #  arg1 '-3d'
  #  arg2 Magnetization vector CSV from simulation
  # Plots the 3D decay of coherence
  #---------------------------------------------------    
  elif "3d" in sys.argv[1].lower():
    # Create simulation class object
    sfo = simf.SimFit()
    sfo.plot3DVec(sys.argv[2])
    #VecAnimate3D(M_4)

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # Tab to CSV splitter
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #  arg1 '-tab2csv'
  #  arg2 Tab delimited file
  # Will dump csv file of same name to same directory
  #---------------------------------------------------
  elif argc == 3 and sys.argv[1].lower() == "-tab2csv" and os.path.isfile(os.path.join(curDir, sys.argv[2])):
    tabPath = os.path.join(curDir, sys.argv[2])
    csvPath = os.path.join(curDir, sys.argv[2].replace(".tab",".csv"))
    FILE = open(tabPath, "rU")
    tabData = [x.strip().split() for x in FILE]
    FILE.close()

    FILE = open(csvPath, "wb")
    for line in tabData:
      FILE.write(",".join(line) + "\n")
    FILE.close()

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # Compare fitted models using statistics files
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #  arg1 '-compare'
  #  arg2 list of model csvs
  # This will compare the first row of each model csv
  #  to each other model to calculate best fit model
  #---------------------------------------------------
  elif sys.argv[1].lower() == "-compare":
    paths = []
    # Get all fit models
    for i in sys.argv[2:]:
      if os.path.isfile(os.path.join(curDir, i)):
        paths.append(os.path.join(curDir, i))
      else:
        print "Model ( %s ) does not exist." % i
    # Make sure at least 2 models to compare
    if len(paths) >= 2:
      sf.CompareModels(paths)
    else:
      print "Not enough models to compare."

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # Calculate thermodynamic parameters from fit file
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #  arg1 '-thermo'
  #  arg2 BMNS fit .csv file
  # Will append thermo values to fit file
  #---------------------------------------------------
  elif argc == 4 and sys.argv[1].lower() == "-thermo" \
       and os.path.isfile(os.path.join(curDir, sys.argv[2])):
    # Path to BMNS fit csv file
    pPath = os.path.join(curDir, sys.argv[2])
    
    # Temperature (Kelvin), assume 0.2K error
    try:
      # Assume spectrometer variance of +/- 0.2K in parameter
      te = ufloat(sys.argv[3], 0.2)
      if te.n < 100.:
        print "Temperature seems to be in centigrade instead of Kelvin"
        print "  Converting from %sC to %sK" % (te.n, te.n + 273.15)
        te = ufloat(te.n + 273.15, 0.2)
    except ValueError:
      print "Invalid temperature given (%s)" % sys.argv[3]
      print "  Setting temperature to 298K"

    # Data to append thermo parameters to and write out
    outData = []
    outPath = pPath.replace(".csv", "") + "_thermo_%0.1f.csv" % te.n

    # Define parse class object
    pInp = fd.Parse()

    # Parse fit data
    fitd = pInp.ParseFitCSV(pPath)

    # Start loop over fit values
    for fit in fitd:
      # Check to make sure not the header
      if fit[0] != "Name":
        # Get populations
        pB = array([fit[5], fit[18]]).astype(float)
        pC = array([fit[6], fit[19]]).astype(float)
        # Get observed rate constants
        #  Make them as numpy array of [val, error]
        kexAB = array([fit[9], fit[22]]).astype(float)
        kexAC = array([fit[10], fit[23]]).astype(float)
        kexBC = array([fit[11], fit[24]]).astype(float)
        # Get rate constants and lifetimes of excited
        #  states and ground-state
        k12, k21, k13, k31, k23, k32, tau1, tau2, tau3 \
          = mf.CalcRateTau(pB, pC, kexAB, kexAC, kexBC)
        # Get free energies and energetic barriers
        dG12, ddG12, ddG21, dG13, ddG13, ddG31, ddG23, ddG32 \
          = mf.CalcG(te, k12, k21, k13, k31, k23, k32, pB, pC)
        # Append old data and new data to output data
        outData.append(",".join(fit) +
          ",".join([
          str(k12.n), str(k21.n), str(k13.n), str(k31.n), str(k23.n), str(k32.n),
          str(tau1.n), str(tau2.n), str(tau3.n),
          str(dG12.n), str(ddG12.n), str(ddG21.n),
          str(dG13.n), str(ddG13.n), str(ddG31.n),
          str(ddG23.n), str(ddG32.n),
          str(k12.std_dev), str(k21.std_dev), str(k13.std_dev), str(k31.std_dev),
          str(k23.std_dev), str(k32.std_dev),
          str(tau1.std_dev), str(tau2.std_dev), str(tau3.std_dev),
          str(dG12.std_dev), str(ddG12.std_dev), str(ddG21.std_dev),
          str(dG13.std_dev), str(ddG13.std_dev), str(ddG31.std_dev),
          str(ddG23.std_dev), str(ddG32.std_dev)]) + ",\n")
      # Handle header
      else:
        headerStr = ",".join(fit) + "k12,k21,k13,k31,k23,k32," \
          + "tau1,tau2,tau3," \
          + "dG12,ddG12,ddG21,dG13,ddG13,ddG31,ddG23,ddG32," \
          + "k12_err,k21_err,k13_err,k31_err,k23_err,k32_err," \
          + "tau1_err,tau2_err,tau3_err," \
          + "dG12_err,ddG12_err,ddG21_err,dG13_err,ddG13_err,ddG31_err,ddG23_err,ddG32_err,\n"
        outData.append(headerStr)
    # Write out fit data
    FILE = open(outPath, "wb")
    for line in outData:
      FILE.write(line)
    FILE.close()

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # Generate Example Simulation Input file
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #  arg1 '-gensim'
  #  arg2 output directory
  #---------------------------------------------------
  elif sys.argv[1].lower() == "-gensim":
    outstr = '''

'''
    outPath = os.path.join(curDir, sys.argv[2])
    makeFolder(outPath)
    FILE = open(os.path.join(outPath, "BMNS-SimParams.txt"), "wb")
    FILE.writelines(outstr)
    FILE.close()

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # Generate Example Parameters Text file
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #  arg1 '-genpar'
  #  arg2 output directory
  #  arg3 optional name to put in example par
  #---------------------------------------------------
  elif sys.argv[1].lower() == "-genpar":
    if argc == 4:
      name = sys.argv[3]
    else:
      name = "FileName"
    outstr = '''
##################################################################################
# Run the BMNS fitting program:
# > python BMNS.py -fit [BM Parameter Input File] [R1rho Data Directory] (Optional Output directory)
##################################################################################
# Define fitting setup.
# FitType: can be 'global' or 'local' or 'brute'
#          This is for global or local optimizations, not shared parameter fits.
#          'Brute' designates brute-force fixed calculations of the range of parameter
#                   space designated by lower/upper bounds on parameters.
#          - 'brutep' will generate plots at each increment point.
#             WARNING: This can take a LONG time.
#
#          'Local' uses Levenberg-Marquardt semi-gradient descent/Gauss-Newton algorithm
#          'Global' uses the "Adaptive Memory Programming for Global Optimizations"
#                   algorithm, with the local 'L-BFGS-B' function, and polishes the
#                   global optimum with L-M.
# FitEqn: fit equation, "BM" for Bloch-McConnell or "Lag" for Laguerre 2-/3-state
# NumFits: is number of fit minima to find (ie. loop over fitting algorithm)
# RandomFitStart : can be 'Yes' or 'No'
#                  if 'Yes', randomly selects initial guess from parameter bounds
##################################################################################
+
FitType local
FitEqn BM
NumFits 1
RandomFitStart No

##################################################################################
# Define fit parameter data, data names, base freqs,
#  initial parameter guesses, and paramter lower and upper bounds. 
#
# Add '+' to read in an additional set of parameters with given 'Name XYZ'
#   The 'Name' must match a .csv data file in given directory of the same name.
#
# Rows for parameters are as follows:
#  [Par name] [initial value] [lower bounds] [upper bounds] ([optional brute force number])
#
# If both lower and upper bounds are not given, they will be set to large values.
# '!' designates a fixed parameter that will not change throughout the fit.
# '*' designates a shared parameter that will be fitted for all data sets
#     also containing the 'x' flag, in a shared manner.
# '@' designates linear brute-force over parameter range of low to upper bounds
# '$' designates log brute-force over parameter range of low to upper bounds
#
# If R1b/c or R2b/c are fixed to 0, they will be shared with R1 / R2
#  e.g. "R1b! = 0.0" will be interpreted as "R1b = R1"
# 
# lf = Larmor frequency (MHz) of the nucleus of interest
#      15N:   60.76302 (600) or  70.960783 (700)
#      13C: 150.784627 (600) or 176.090575 (700)
#
# AlignMag [Auto/Avg/GS]
#          Auto : calculates kex/dw and aligns mag depending on slow (gs) vs. fast (avg)
#          Avg : Aligns magnetization/projects along average effective field of GS/ESs
#          GS : Aligns magnetization along ground-state
#
# x-axis Lower Upper (Hz): Sets lower and upper x-axis limits for both plots
#   if not given, will automatically set them
#
# y-axis Lower Upper : Sets lower and upper y-axis limits for both plots
#   if not given, will automatically set them
#
# Trelax increment Tmax (seconds) : sets the increment delay and maximum relaxation
#  delay to simulate R1rho at.
#  Use caution with this flag, recommended that is remains commented out.
#  Array of delays is given as a linear spacing from 0 - Tmax in Tmax/Tinc number of points
#  If not defined, the program will calculate the best Tmax from the experimental
#   R1rho data.
##################################################################################

+
Name %s
lf 70.960783
AlignMag Auto
#Trelax 0.0005 0.5
#x-axis -2000 2000
#y-axis 0 50
pB 0.002 1e-6 0.5
pC! 0.0 1e-6 0.5
dwB 20.0 -80 80
dwC! 0.0 -80 80
kexAB 3000. 1. 50000.
kexAC! 0.0 1. 50000.
kexBC! 0.0 1. 50000.
R1 2.5 1e-6 20.
R2 6.0 1e-6 200.
R1b! 0.0
R2b! 0.0
R1c! 0.0
R2c! 0.0
''' % name

    outPath = os.path.join(curDir, sys.argv[2])
    makeFolder(outPath)
    FILE = open(os.path.join(outPath, "BMNS-Parameters.txt"), "wb")
    FILE.writelines(outstr)
    FILE.close()

  else: bme.help()

# Get command line arguments
curDir = os.getcwd()
argc = len(sys.argv)

# Run the actual program
Main()