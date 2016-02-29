#########################################################################
# Bloch-McConnell 2-/3-state Fitting Program v1.25
#  Beta version.
#  Isaac Kimsey 02-11-2016
#
# External Dependencies: numpy, matplotlib, scipy, leastsqbound, uncertainties
#########################################################################

### General imports ###
import itertools as it
import os, sys
### BMNS Imports
import BMNS_SimR1p as sim
### Direct numpy imports ###
from numpy import append, array, asarray
from numpy import column_stack, concatenate
from numpy import delete
from numpy import float64
from numpy import interp
from numpy import linspace, logspace, log10
from numpy import shape
### Numpy sub imports ###
from numpy.random import uniform
from numpy.core.defchararray import rstrip
### Matplotlib imports ###
import matplotlib.pyplot as plt
import matplotlib as mpl
### Uncertainties imports for error propagation
from uncertainties import umath
from uncertainties import ufloat

#########################################################################
# *GraphFit class* is solely used graph fitted data and trend lines
#########################################################################
class GraphFit:
  def __init__(self):
    pass

  #---------------------------#---------------------------#
  # Takes a path to input text file and parses it accordingly
  #  Must be run after fitted parameters have been updated in the
  #  pass fit class objects self.Pars dictionary, as the fit
  #  graph will be using these values.
  #   ob : fit class object containing raw data and fitted parameters
  #   outpath : output file path to dump out figures.
  #   loopNum : Integer number from loops over fitting algorithm + 1
  #   time : numpy array of times (sec)
  #---------------------------#---------------------------#
  def WriteGraph(self, ob, outpath, loopNum, time, FitType="local", FitEqn="bm"):
    mpl.rcParams['pdf.fonttype'] = 42

    # Null reduced chi-sq
    redchisq = 999.
    # Need to split data up by spinlock powers
    # Stores split data in this list (eventually numpy array)
    exptData = []
    # Figure output name and path
    figPath = os.path.join(outpath, "R1pR2eff_Figs_%s_%s.pdf" % (ob.name, loopNum))
    # Figure output name and path for R2eff only
    figR2effPath = os.path.join(outpath, "R2eff_Figs_%s_%s.pdf" % (ob.name, loopNum))
    # Figure output name and path for R1rho only
    figR1rhoPath = os.path.join(outpath, "R1rho_Figs_%s_%s.pdf" % (ob.name, loopNum))
    # Figure output name and path for R1rho only
    figOnResPath = os.path.join(outpath, "OnResR1rho_Figs_%s_%s.pdf" % (ob.name, loopNum))
    # Figure output name and path
    dataPath = os.path.join(outpath, "Data_%s_%s.csv" % (ob.name, loopNum))
    # Get all offset spinlock powers
    #  Ignore on-resonance points
    cvals = sorted(list(set([x for x,y in zip(ob.R1pD[:,1],ob.R1pD[:,0]) if y != 0.0])))
    # Get offset min and max values and add 5%
    offmin, offmax = min(ob.R1pD[:,0])*1.05, max(ob.R1pD[:,0])*1.05

    # Get reduced chi-square
    if FitType == "local":
      redchisq = ob.localFits[loopNum]["RedChiSq"]
    elif FitType == "global":
      redchisq = ob.globalFits[loopNum]["RedChiSq"]
    elif FitType == "polish":
      redchisq = ob.polishedFits[loopNum]["RedChiSq"]

    # Number of increments for simulating values for trendline
    offIncrNum = 1000
    # Simulate offset increments
    simoffs = linspace(offmin, offmax, offIncrNum)

    # Loop over spinlock power (offres only) values in cvals list
    #  to split the R1pD numpy array in to blocks dependending on only
    #  off-res data
    for val in cvals:
      td = [] # Temporary data list
      # Iterate over data in Fit object R1pD data numpy array
      for d in ob.R1pD:
        # Match spinlock power to val in loop over cvals
        if d[1] == val:
          td.append(d)
      # Append to exptData array storing all pertinent exp data
      exptData.append(td)

    # Just to make sure master and subarrays are numpy array
    exptData = [asarray(x) for x in exptData]
    exptData = asarray(exptData)

    # Create numpy parameter array to be passed to sim.BMFitFunc to simulate
    #  R1rho and R2eff values based on fitted parameters
    fitPars = array([ob.Pars['pB_%s'%ob.FitNum][5], ob.Pars['pC_%s'%ob.FitNum][5],
                            ob.Pars['dwB_%s'%ob.FitNum][5], ob.Pars['dwC_%s'%ob.FitNum][5],
                            ob.Pars['kexAB_%s'%ob.FitNum][5], ob.Pars['kexAC_%s'%ob.FitNum][5],
                            ob.Pars['kexBC_%s'%ob.FitNum][5], ob.Pars['R1_%s'%ob.FitNum][5],
                            ob.Pars['R1b_%s'%ob.FitNum][5], ob.Pars['R1c_%s'%ob.FitNum][5],
                            ob.Pars['R2_%s'%ob.FitNum][5], ob.Pars['R2b_%s'%ob.FitNum][5],
                            ob.Pars['R2c_%s'%ob.FitNum][5]])

    # Create a limited numpy parameter array to be passed to sim.CalcR2eff and sim.MCError
    #  to calculate R2eff based on fitted R1 and known/simulated R1rho and associated error
    pB, pC, dwB, dwC, kexAB, kexAC, kexBC, R1 = (ob.Pars['pB_%s'%ob.FitNum][5], ob.Pars['pC_%s'%ob.FitNum][5],
                            ob.Pars['dwB_%s'%ob.FitNum][5], ob.Pars['dwC_%s'%ob.FitNum][5],
                            ob.Pars['kexAB_%s'%ob.FitNum][5], ob.Pars['kexAC_%s'%ob.FitNum][5],
                            ob.Pars['kexBC_%s'%ob.FitNum][5], ob.Pars['R1_%s'%ob.FitNum][5])
    

    ### Create a master data list that will contain ###
    #  all experimental data, calculated R2eff, simulated R1p/R2eff
    #  and the corresponding residulas
    mData = []
    ### Simulate data for trend-lines ###
    sData = []
    for d in exptData:
      # Calculate R2eff and R2eff error from the experimental R1rho values
      # Returns a numpy array in the order of:
      #  [[R2eff, R2eff error],...]
      # Where R2eff error propagated from R1rho error using linear propagation theory
      #  in the uncertainties package
      if FitEqn == "bm":
        td1 = array([[sim.CalcR2eff(ufloat(kR1p, err),pB,pC,dwB,dwC,kexAB,kexAC,
                      kexBC,R1,SL,-1*OF,ob.lf,AlignMag=ob.AlignMag,Error=True).n,
                      sim.CalcR2eff(ufloat(kR1p, err),pB,pC,dwB,dwC,kexAB,kexAC,
                      kexBC,R1,SL,-1*OF,ob.lf,AlignMag=ob.AlignMag,Error=True).std_dev]
                    for (SL,OF,kR1p,err) in zip(d[:,1],d[:,0],d[:,2],d[:,3])])
        # Simulate R1rho and R2eff based on fitted parameters in the ob.Pars object (by fitPars numpy array)
        # Returns a numpy array in the order of:
        #  [[SimR1p, SimR2eff],...]
        td2 = array([sim.BMFitFunc(fitPars,SL,-1.*OF,ob.lf,ob.time,ob.AlignMag,1,kR1p)
                    for (SL,OF,kR1p,err) in zip(d[:,1],d[:,0],d[:,2],d[:,3])])

      elif FitEqn == "lag":
        td1 = array([[sim.CalcR2eff(ufloat(kR1p, err),pB,pC,dwB,dwC,kexAB,kexAC,
                      kexBC,R1,SL,-1*OF,ob.lf,AlignMag=ob.AlignMag,Error=True).n,
                      sim.CalcR2eff(ufloat(kR1p, err),pB,pC,dwB,dwC,kexAB,kexAC,
                      kexBC,R1,SL,-1*OF,ob.lf,AlignMag=ob.AlignMag,Error=True).std_dev]
                    for (SL,OF,kR1p,err) in zip(d[:,1],d[:,0],d[:,2],d[:,3])])
        # Simulate R1rho and R2eff based on fitted parameters in the ob.Pars object (by fitPars numpy array)
        # Returns a numpy array in the order of:
        #  [[SimR1p, SimR2eff],...]
        td2 = array([sim.LagFitFunc(fitPars,SL,-1.*OF,ob.lf,ob.time,ob.AlignMag,1,kR1p)
                    for (SL,OF,kR1p,err) in zip(d[:,1],d[:,0],d[:,2],d[:,3])])

      # Concatenate td1 and td2 horizontally to give
      #  [[R2eff, R2eff error, SimR1p, SimR2eff],...]
      td3 = concatenate((td1,td2), axis=1)
      # Concatenate original expt data with td3 numpy array to give:
      #  [[Offset, SLP, R1p, R1p error, R2eff, R2eff error, SimR1p, SimR2eff]]
      td4 = concatenate((d,td3), axis=1)
      # Calculate R1p and R2eff residuals:
      #  [[(R1p-SimR1p),(R2eff-SimR2eff)],...]
      residArr = td4[:,[2,4]] - td2
      # Now combine them all together, again horizontally, to give the master array:
      #  [[Offset, SLP, R1p, R1p err, R2eff, R2eff error, SimR1p, SimR2eff, Resid. R1p, Resid. R2eff],...]
      mData.append(concatenate((td4,residArr), axis=1))
      # Spinlock power for simulations
      sSLP = d[0][1]

      #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      # Linearly interpolate R1rho values wrt the simOffs array
      #  using known offsets and R1rho values to interpolate between the two.
      # Need to do this to get an estimate of the "known" R1rho value for
      #  all of these simulated offsets so that the 2-point BM routine
      #  has a good Tmax to simulate to.
      #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      # First sort real offsets and SLPs by offsets, or else
      #  interpolation will fail
      sortedR1p = array(sorted([[i,j] for i,j in zip(d[:,0], d[:,2])]))
      # Interpolate values
      estR1rho = interp(simoffs, sortedR1p[:,0], sortedR1p[:,1])
  
      # Simulate R1rho and R2eff values over the range of simOffs
      #  given an estimated 'known R1rho' based on linear interpolation
      if FitEqn == "bm":
        sd1 = array([sim.BMFitFunc(fitPars,sSLP,-1.*OF,ob.lf,ob.time,ob.AlignMag,1,estR1p)
                    for OF,estR1p in zip(simoffs,estR1rho)])
      elif FitEqn == "lag":
        sd1 = array([sim.LagFitFunc(fitPars,sSLP,-1.*OF,ob.lf,ob.time,ob.AlignMag,1,estR1p)
                    for OF,estR1p in zip(simoffs,estR1rho)])        
      # Stack the simOffs with simulated R1rho/R2eff vals
      sData.append(column_stack((simoffs,sd1)))
    
    # Cast list as numpy array
    mData = array(mData)
    sData = array(sData)

    ### Write out this data to a CSV file
    FILE = open(dataPath, "ab")
    FILE.write("Offset,SLP,R1p,R1p err,R2eff,R2eff err,Sim R1p, Sim R2eff,Residual R1p, Residual R2eff,\n")
    for d in mData:
      for line in d:
        FILE.write(",".join(list([str(x) for x in line])) + "\n")
    FILE.close()

    #### Plot: R1p, R2eff, R1p Residual, R2eff Residual ####
    ### Plot the data in the master data array and in the simulated trend lines
    ## 2x2 Plot: ul (R1p), ll (R1p residual), ur (R2eff), lr (R2eff residual)
    fig, ax = plt.subplots(2,2, sharex=True, figsize=(16,10), dpi=80)

    # Loop over experimental and simulated data and produce graphs
    for i,j in zip(mData,sData):
      ## Up-Left : R1rho plot
        # Plot R1rho with error bars
      # plot1 = ax[0,0].plot(j[:,0]/1e3, j[:,1])
      plot1 = ax[0,0].errorbar(i[:,0]/1e3, i[:,2], yerr = i[:,3], fmt = 'o')
        # Plot R1rho trendlines
      ax[0,0].plot(j[:,0]/1e3, j[:,1], c=plot1[0].get_color(), label = int(i[0][1]))
      ax[0,0].set_title(r'$R_{1\rho}\,\mathrm{ Plot}\,|\,\overline{\chi}^2\,%.2f$'
                        % redchisq, size=18)
      # Set axes limits
      if ob.xmin is None and ob.xmax is None:
        ax[0,0].set_xlim(offmin/1e3, offmax/1e3)
      else:
        ax[0,0].set_xlim(ob.xmin/1e3, ob.xmax/1e3)
      # Y-axes for R1rho plot
      if ob.ymin is not None and ob.ymax is not None:
        ax[0,0].set_ylim(ob.ymin, ob.ymax)
      ax[0,0].set_ylabel(r'$R_{1\rho}\,(s^{-1})$', size=16)
      # Set legend
      legend = ax[0,0].legend(title=r'$\omega\,2\pi^{-1}\,{(Hz)}$', numpoints=1, fancybox=True)
      plt.setp(legend.get_title(), fontsize=16)
      ## Up-Right : R2eff plot
        # Plot R2eff with error bars
      # plot2 = ax[0,1].errorbar(j[:,0]/1e3, j[:,2])
      plot2 = ax[0,1].errorbar(i[:,0]/1e3, i[:,4], yerr = i[:,5], fmt = 'o')
        # Plot R2eff trendlines
      ax[0,1].plot(j[:,0]/1e3, j[:,2], c=plot2[0].get_color())
      ax[0,1].set_title(r'$R_2+R_{ex}\,\mathrm{ Plot}$', size=18)
      # Y-axes for R2eff plot
      if ob.ymin is not None and ob.ymax is not None:
        ax[0,1].set_ylim(ob.ymin, ob.ymax)
      ax[0,1].set_ylabel(r'$R_2+R_{ex}\,(s^{-1})$', size=16)
      ## Bottom-Left : Residual R1rho
        # Plot R1rho residual scatter
      ax[1,0].plot(i[:,0]/1e3, i[:,8], 'o', c=plot1[0].get_color())
      ax[1,0].title.set_visible(False)
      ax[1,0].set_xlabel(r'$\Omega\,2\pi^{-1}\,{(kHz)}$', size=16)
      ax[1,0].set_ylabel(r'$R_{1\rho}\,\mathrm{ Residual}$', size=16)
      ## Bottom-Right : Residual R2eff
        # Plot R2eff residual scatter
      ax[1,1].plot(i[:,0]/1e3, i[:,9], 'o', c=plot2[0].get_color())
      ax[1,1].title.set_visible(False)
      ax[1,1].set_xlabel(r'$\Omega\,2\pi^{-1}\,{(kHz)}$', size=16)
      ax[1,1].set_ylabel(r'$R_2+R_{ex}\,\mathrm{ Residual}$', size=16)
    fig.set_tight_layout(True)
    # Write out figure
    fig.savefig(figPath)
    # Clear figure
    fig.clf()
    # Close plot
    plt.close(fig)

    #### Plot R2eff only ####
    ### Plot the data in the master data array and in the simulated trend lines
    fig2, ax2 = plt.subplots(1,1, figsize=(8,6), dpi=80)
    # Loop over experimental and simulated data and produce graphs
    for i,j in zip(mData,sData):
      # plot1 = ax2.errorbar(j[:,0]/1e3, j[:,2])
      plot1 = ax2.errorbar(i[:,0]/1e3, i[:,4], yerr = i[:,5], fmt = 'o')
        # Plot R2eff trendlines
      ax2.plot(j[:,0]/1e3, j[:,2], c=plot1[0].get_color(), label = int(i[0][1]))
      
      # Set plot title to include the red. chi-square value
      ax2.set_title(r'$\overline{\chi}^2\,%.2f$'
                    % redchisq, size=18)

      # Xlims
      if ob.xmin is None and ob.xmax is None:
        ax2.set_xlim(offmin/1e3, offmax/1e3)
      else:
        ax2.set_xlim(ob.xmin/1e3, ob.xmax/1e3)
      # Y-ax2es for R2eff plot
      if ob.ymin is not None and ob.ymax is not None:
        ax2.set_ylim(ob.ymin, ob.ymax)
      ax2.set_ylabel(r'$R_2+R_{ex}\,(s^{-1})$', size=18)
      ax2.set_xlabel(r'$\Omega\,2\pi^{-1}\,{(kHz)}$', size=18)

      # Set figure legend
      legend = ax2.legend(title=r'$\omega\,2\pi^{-1}\,{(Hz)}$', numpoints=1, fancybox=True)
      plt.setp(legend.get_title(), fontsize=18)

    fig2.set_tight_layout(True)
    # Write out fig2ure
    fig2.savefig(figR2effPath)
    # Clear fig2ure
    fig2.clf()
    # Close plot
    plt.close(fig2)

    #### Plot R1rho only ####
    ### Plot the data in the master data array and in the simulated trend lines
    fig2, ax2 = plt.subplots(1,1, figsize=(8,6), dpi=80)
    # Loop over experimental and simulated data and produce graphs
    for i,j in zip(mData,sData):
      # plot1 = ax2.errorbar(j[:,0]/1e3, j[:,2])
      plot1 = ax2.errorbar(i[:,0]/1e3, i[:,2], yerr = i[:,3], fmt = 'o')
        # Plot R1rho trendlines
      ax2.plot(j[:,0]/1e3, j[:,1], c=plot1[0].get_color(), label = int(i[0][1]))

      # Set plot title to include the red. chi-square value
      ax2.set_title(r'$\overline{\chi}^2\,%.2f$'
                    % redchisq, size=18)

      # Xlims
      if ob.xmin is None and ob.xmax is None:
        ax2.set_xlim(offmin/1e3, offmax/1e3)
      else:
        ax2.set_xlim(ob.xmin/1e3, ob.xmax/1e3)
      # Y-ax2es for R2eff plot
      if ob.ymin is not None and ob.ymax is not None:
        ax2.set_ylim(ob.ymin, ob.ymax)
      ax2.set_ylabel(r'$R_{1\rho}\,(s^{-1})$', size=18)
      ax2.set_xlabel(r'$\Omega\,2\pi^{-1}\,{(kHz)}$', size=18)

      # Set figure legend
      legend = ax2.legend(title=r'$\omega\,2\pi^{-1}\,{(Hz)}$', numpoints=1, fancybox=True)
      plt.setp(legend.get_title(), fontsize=18)

    fig2.set_tight_layout(True)
    # Write out fig2ure
    fig2.savefig(figR1rhoPath)
    # Clear fig2ure
    fig2.clf()
    # Close plot
    plt.close(fig2)

    ### Now Handle On-Resonance Data ###
    # Get all on-resonance spinlock powers (if they exist)
    onvals = sorted(list(set([x for x,y in zip(ob.R1pD[:,1],ob.R1pD[:,0]) if y == 0.0])))
    if len(onvals) != 0:
       # Get onres SLP min and max values and add 5%
      onmin, onmax = min(onvals) * 0.95, max(onvals)*1.05
      # Simulate SLPs for onres increments
      onIncrNum = 100
      simSLs = linspace(onmin, onmax, onIncrNum)
      # Grab only the on-resonance data
      onresD = array([x for x,y in zip(ob.R1pD, ob.R1pD[:,0]) if y == 0.0])
      # Interpolate R1rho values for different SLPs
      # First sort real onres SLPs
      sortedR1p = array(sorted([[i,j] for i,j in zip(onresD[:,1], onresD[:,2])]))
      # Interpolate R1rho values between onres SLPs
      estOnR1rho = interp(simSLs, sortedR1p[:,0], sortedR1p[:,1])
      # Simulate the on-resonance R1rho points
      if FitEqn == "bm":
        simOnRes = array([sim.BMFitFunc(fitPars,sSLP,0.0,ob.lf,ob.time,ob.AlignMag,0,estR1p)
                        for sSLP,estR1p in zip(simSLs,estOnR1rho)])
      elif FitEqn == "lag":
        simOnRes = array([sim.LagFitFunc(fitPars,sSLP,0.0,ob.lf,ob.time,ob.AlignMag,0,estR1p)
                        for sSLP,estR1p in zip(simSLs,estOnR1rho)])

      #### Plot On-Res R1rho ####
      ## if there is any data to plot ##
      ### Plot the data in the master data array and in the simulated trend lines
      fig2, ax2 = plt.subplots(1,1, figsize=(8,6), dpi=80)

      # Plot individual onres R1rho points
      plot1 = ax2.errorbar(onresD[:,1]/1e3, onresD[:,2], 
                           c='black', yerr = onresD[:,3], fmt = 'o')
      # Plot onres R1rho trendlines
      ax2.plot(simSLs/1e3, simOnRes, c='red')
      
      # Set plot title to include the red. chi-square value
      ax2.set_title(r'$\overline{\chi}^2\,%.2f$'
                    % redchisq, size=18)

      # Xlims
      if ob.xmin is None and ob.xmax is None:
        ax2.set_xlim(0./1e3, onmax/1e3)
      else:
        ax2.set_xlim(ob.xmin/1e3, ob.xmax/1e3)
      # Y-ax2es for R2eff plot
      if ob.ymin is not None and ob.ymax is not None:
        ax2.set_ylim(ob.ymin, ob.ymax)
      ax2.set_ylabel(r'$R_{1\rho}\,(s^{-1})$', size=18)
      ax2.set_xlabel(r'$\omega\,2\pi^{-1}\,{(kHz)}$', size=18)

      fig2.set_tight_layout(True)
      # Write out fig2ure
      fig2.savefig(figOnResPath)
      # Clear fig2ure
      fig2.clf()
      # Close plot
      plt.close(fig2)

#########################################################################
# *Parse class* is solely used to parse input data (R1rho or parameter files)
#########################################################################
class Parse:
  def __init__(self):
    # Variables expected to be in the input parameter file
    #  excludes potential shared fit flags ('*')
    self.Variables = ["Name","lf","pB","pC","dwB","dwC","kexAB","kexAC","kexBC","R1","R2","R1b","R1c","R2b","R2c"]
    self.AltFlags = ["x-axis", "y-axis", "trelax", "alignmag"]
    self.ParInp = []  # Store semi-raw parameters read from file
                      #  Sub lists for each '+' delimited parameter block
    self.DataInp = [] # Store semi-raw data read from files
                      #  Sub lists for each data file
    self.NumInps = 0
    # Array with fit type strings
    self.FitType = []

  #---------------------------#---------------------------#
  # Takes a path to BMNS fit file and parses it accordingly
  #  Returns a numpy array of values
  #---------------------------#---------------------------#
  def ParseFitCSV(self, PathToFit):
    # Fit numpy array
    fitArry = None

    # Check that file exists
    if os.path.isfile(PathToFit):
      # Open data and split to list
      FILE = open(PathToFit, "rU")
      td = [x.strip().split(",") for x in FILE]
      FILE.close()

    return td

  #---------------------------#---------------------------#
  # Takes a path to input text file and parses it accordingly
  #---------------------------#---------------------------#
  def ParseInp(self, PathToPars):
    out = []
    # Open input parameter file
    with open(PathToPars,'r') as f:
        # Split input parameter file by delimiter '+' to lists
        for key,group in it.groupby(f,lambda line: line.startswith('+')):
            if not key:
                out.append(list(group))
    # Strip and remove comments
    self.ParInp = [[x.strip().split(" ") for x in y if "#" not in x and "Fit" not in x and len(x)!= 1] for y in out]
    # Remove null sub-lists
    self.ParInp = [x for x in self.ParInp if len(x) > 0]

    ## Get fit type comments
    # Strip and remove comments
    self.FitType = [[x.strip().split(" ") for x in y if "#" not in x and "Fit" in x and len(x)!= 1] for y in out]
    # Remove null sub-lists
    self.FitType = [x for x in list(it.chain.from_iterable(self.FitType)) if len(x) > 0]

  #---------------------------#---------------------------#
  # This function checks that all expected variables are in
  #  the self.ParInp list. If not, it returns a False bool
  #  and the corresponding error message
  #---------------------------#---------------------------#
  def CheckPars(self, dataPath):
    errBool = False
    if len(self.ParInp) == 0:
      return True, "Parameter Input list was empty. Check your input file again.\n"
    else:
      missStr = "The following are missing from your parameters file:\n"
      matchNum = 0 # Number of variables in input parameter file matching self.Variables
      
      # Set the number of parameter sets in input parameter file
      self.NumInps += len(self.ParInp)

      # Loop over input clusters to make sure parameter names are included
      #  and that values are given for each, and bounds
      for idx,inps in enumerate(self.ParInp):

        ### Check that names exist ###
        # Flatten ParInp and remove any shared flags, '*'
        flat2d = [x.replace('*','').replace('!','').replace('@','').replace('$','') for x in 
                  list(it.chain.from_iterable(inps))]
        # Assign list of matching variables given in input parameter file
        #  that match the self.Variables list (should be 14)
        matchNum += len([i for i in self.Variables if i in flat2d])
        # Give missing variables
        without = [i for i in self.Variables if i not in flat2d]
        # Set output error message with missing variables and input
        #  parameter number.
        if len(without) != 0:
          missStr += "  ("+",".join(without) + ") is missing from input set #%s\n" % str(idx+1)
          errBool = True

        ## Now check to make sure names have:
        #   1. correct bounds
        #   2. correct values or numbers
        #   3. Make sure bounds and initial guesses are floats
        #   4. Name corresponds to an actual file
        #   5. Catch any extra undefined parameters.
        for val in inps:
          # Check that par name has initial guess and/or bounds
          if 2 > len(val) < 4 and val[0].lower() not in self.AltFlags:
            missStr += "  (%s) in input set #%s is missing an upper or lower bounds.\n" % (val[0], str(idx+1))
            errBool = True
          elif len(val) == 1 and val[0].lower() not in self.AltFlags:
            missStr += "  (%s) in input set #%s is missing initial guess and parameter bounds\n" % (val[0], str(idx+1))
            errBool = True
          # Check to make sure the input parameters that are to be cast as floats
          #  do not contain string values
          elif len(val) == 2 and "name" not in val[0].lower() and val[0].lower() not in self.AltFlags:
            try:
              float(val[1])
            except ValueError:
              missStr += "  (%s) in input set #%s has a bad initial guess (%s)\n" % (val[0], str(idx+1), val[1])
              errBool = True                       
          elif len(val) == 4 and "name" not in val[0].lower() and "lf" not in val[0].lower():
            try:
              float(val[1])
              float(val[2])
              float(val[3])
            except ValueError:
              missStr += ("  (%s) in input set #%s has a bad initial guess or bounds (%s)\n" 
                          % (val[0], str(idx+1), ", ".join(val[1:])))
              errBool = True   
          # Now check if data file in name matches real file
          # Check for .csv and .tab files
          #  Will handle parsing these later, just check that they exist first.
          if "name" in val[0].lower():
            csvpath = os.path.join(dataPath, os.path.splitext(val[1])[0] + ".csv")
            tabpath = os.path.join(dataPath, os.path.splitext(val[1])[0] + ".tab")
            if not os.path.isfile(csvpath) and not os.path.isfile(tabpath):
              missStr += ("  Filename (%s) for set #%s does not exist in datapath ( %s )\n" 
                          % (val[1], str(idx+1), dataPath))
              errBool = True
          
          # Check how magnetization is to be aligned
          #  Make sure parameter is defined as auto, gs or average
          #  and make sure value is defined
          elif "alignmag" in val[0].lower():
            if len(val) == 2:
              # Check to see if user defines the alignment of magnetization as 'auto', 'gs', or 'average'
              topts = ["auto", "gs", "avg"]
              if val[1].lower() not in topts:
                missStr += ("  Alignment/projection of magnetization has not been defined correctly (%s)"
                            % val[1])
                missStr += ("    Please define as 'Auto', 'Avg', or 'GS'")
                errBool = True
            # If not given, tell user to define alignment of magnetization
            else:
              missStr += ("  Alignment/projection of magnetization has not been declared.")
              errBool = True

          # Now check if set graphing axis
          #  If not flagged, it's okay, will just use defaults.
          #  but need to check that right number of vals given
          #  and they are numerical values.
          elif "axis" in val[0].lower():
            if len(val) == 3:
              try:
                float(val[1])
                float(val[2])
              except ValueError:
                missStr += ("  Axes limits (%s %s) are not defined as numbers." % (val[1], val[2]))
                errBool = True
            else:
              missStr += ("  Incorrectly defined upper or lower axes limits (%s) ." % (" ".join(val)))
              errBool = True
          # Check for relaxation time (max)
          elif val[0].lower() == "trelax":
            if len(val) == 3:
              try:
                float(val[1])
                float(val[2])
              except ValueError:
                missStr += ("  Relaxation delay increment (%s) or max (%s) is non-numerical."
                            % (val[1], val[2]))
                errBool = True
            else:
              missStr += ("  Set relaxation delay increment (sec) and maximum (sec)")
              errBool = True              
          # Now check that we don't have extra parameters
          elif val[0].replace('*','').replace('!','').replace('@','').replace('$','') not in self.Variables + self.AltFlags and val[0] not in without:
            missStr += ("  Parameter (%s) in set #%s is not defined, please remove it.\n" 
                        % (val[0], str(idx+1)))
            errBool = True            
      # If everything is good, return True and no error message
      if errBool == False:
        return errBool, ""
      else:
        return True, missStr
      # if matchNum == (self.NumInps * len(self.Variables)):
      #   return False, ""
      # else:
      #   return True, missStr

  #---------------------------#---------------------------#
  # Takes a path to .csv or .tab data file, and file extension type
  #  If .tab file, assume:
  #    col0 folder #, col1 SLP, col2 corr offset,
  #    col3 R1rho, col4 R1rho err
  # Assumes file exists.
  #---------------------------#---------------------------#
  def CsvTabData(self, PathToTab, fileExt):
    FILE = open(PathToTab, "rU")
    # Read .csv data
    if fileExt == ".csv":
      tabData = array([x.strip().split(",") for x in FILE])
    # Read .tab data
    elif fileExt == ".tab":
      # Split by tab
      tabData = array([x.strip().split() for x in FILE])
      # Swap col1 (SLP, Hz) with col2 (Offset, Hz)
      tabData[:,[1,2]] = tabData[:,[2,1]]
      # Strip any colons from col0
      tabData[:,0] = rstrip(tabData[:,0], ":")
    FILE.close()

    return tabData

  #---------------------------#---------------------------#
  # Takes a path to data folder and a list of names
  #  Checks that data files exists, then parses data
  #  to semi-raw format.
  # Performs size checks as well,
  #  returns Bool and Error string
  #---------------------------#---------------------------#
  def ParseData(self, PathToData, Name=None):
    # File extension type, default .csv
    fileExt = ".csv"
    # Check to see if need to combine path and name
    if Name != None:
    # Assign file name and convert to path
    # Discern between tab and csv
      # Check to see if .csv file exists
      tName = Name + ".csv"
      dCsv = os.path.join(PathToData, tName)
      # Check to see if .tab file exists
      tName = Name + ".tab"
      dTab = os.path.join(PathToData, tName)
      # If both filetypes exist, opt for .csv file
      if os.path.isfile(dCsv) and os.path.isfile(dTab):
        print "  Warning: Both %s.csv and %s.tab files exist." % (Name, Name)
        print "           %s.csv will be used." % Name
        dPath = dCsv
      elif os.path.isfile(dCsv) and not os.path.isfile(dTab):
        dPath = dCsv
      else:
        dPath = dTab
        fileExt = ".tab"
    else:
      dPath = PathToData
    # String to add error messages
    missStr = ""
    errBool = False
    # Check to see if file exists. This is just a double-check, should have been caught already.
    #  If it does, unpack it.
    if os.path.isfile(dPath):
      # FILE = open(dPath, "rU")
      # tData = array([x.strip().split() for x in FILE])
      # FILE.close()
      tData = self.CsvTabData(dPath, fileExt)
      # First check to see if first column is a alphanumeric title
      try:
        # Cast first column as floats
        [float(x) for x in tData[0]]
      # First column is non-numerical, indicating a header
      #  Remove the header
      except ValueError:
        # Delete first row, assumed to be the header
        tData = delete(tData, 0, 0)

      # Check to see if any non-numeric numbers are in the
      #  import data file.
      #  If there are, return False and error message
      try:
        tData = tData.astype(float)

      except ValueError:
        missStr += "  ERROR: Non-numeric values found in data file ( %s )\n" % Name
        errBool = True

      # Check to make sure input data has at least 2 rows
      #  and 3 columns (offset, spinlock, R1rho, [err optional])
      # If more columns than 4, will just ignore them and only
      #  consider the first 1-4
      if len(tData.shape) == 2:
        if tData.shape[0] >= 2 and tData.shape[1] >= 3:
          # Now check to see if there are more than 4 columns
          #  in data array.
          # If there are, assume that column 0 is folder numbers
          if tData.shape[1] >= 5:
            tData = delete(tData, 0, 1)
          self.DataInp.append(tData)

        else:
          missStr += "  ERROR: Too few data rows and/or columns for ( %s )\n" % Name
          errBool = True
      # Check that numpy array does not have irregular columns
      else:
        missStr += ("  ERROR: Irregular data shape ( %s ). Check for extra/missing data points or columns.\n"
                    % Name)
        errBool = True

    # If data files do not exist, do nothing and return.
    #  Loss of data will be caught with CheckData func
    else:
      missStr += "  ERROR: No data file found for ( %s )\n" % Name
      errBool = True

    return errBool, missStr

#########################################################################
# *Parameters class* is used to store and manipulate the parameters
#   of the BMNS being simulated and fitted
# Inherited by: Fits
#########################################################################
class Parameters:
  def __init__(self, FitNum = 0):
    self.P0 = array([])
    # Name given in parameter file
    self.name = None
    # Larmor freq. Needed to convert between ppm->Hz/rads
    self.lf = None
    # parName_Fit# : [p0 Val 0, (lowBound,upBound) 1, Share 2, Fixed 3, 
    #                 Share Partner 4, Fit Val 5, Fit Error 6, Brute-force Type 7
    #                 Brute-force it number 8]
    self.Pars = {
                'pB_%s' % FitNum : [None,(None,None),False, False, 'pB_%s' % FitNum, None, None, None, 10],
                'pC_%s' % FitNum : [None,(None,None),False, False, 'pC_%s' % FitNum, None, None, None, 10],
                'dwB_%s' % FitNum : [None,(None,None),False, False, 'dwB_%s' % FitNum, None, None, None, 10],
                'dwC_%s' % FitNum : [None,(None,None),False, False, 'dwC_%s' % FitNum, None, None, None, 10],
                'kexAB_%s' % FitNum : [None,(None,None),False, False, 'kexAB_%s' % FitNum, None, None, None, 10],
                'kexAC_%s' % FitNum : [None,(None,None),False, False, 'kexAC_%s' % FitNum, None, None, None, 10],
                'kexBC_%s' % FitNum : [None,(None,None),False, False, 'kexBC_%s' % FitNum, None, None, None, 10],
                'R1_%s' % FitNum : [None,(None,None),False, False, 'R1_%s' % FitNum, None, None, None, 10],
                'R1b_%s' % FitNum : [None,(None,None),False, False, 'R1b_%s' % FitNum, None, None, None, 10],
                'R1c_%s' % FitNum : [None,(None,None),False, False, 'R1c_%s' % FitNum, None, None, None, 10],
                'R2_%s' % FitNum : [None,(None,None),False, False, 'R2_%s' % FitNum, None, None, None, 10],
                'R2b_%s' % FitNum : [None,(None,None),False, False, 'R2b_%s' % FitNum, None, None, None, 10],
                'R2c_%s' % FitNum : [None,(None,None),False, False, 'R2c_%s' % FitNum, None, None, None, 10]
                }

    # Define how to align magnetization during BM simulation
    self.AlignMag = "auto"

    # Graphing limits
    self.xmin = None
    self.xmax = None
    self.ymin = None
    self.ymax = None

    # Default Relaxation delays (can be overwritten in parameter input file)
    self.tInc = 0.005
    self.tMax = 0.25
    self.time = linspace(0.0, self.tMax, self.tMax / self.tInc)

  #---------------------------#---------------------------#
  # Grabs parameter values of Pars dict and returns
  #  a numpy array with these values in order:
  #  pB,pC,dwB,dwC,kexAB,kexAC,kexBC,R1,R1b,R1c,R2,R2b,R2c
  #---------------------------#---------------------------#
  def GrabPars(self):
    return array([self.Pars['pB_%s'%self.FitNum][5], self.Pars['pC_%s'%self.FitNum][5],
              self.Pars['dwB_%s'%self.FitNum][5], self.Pars['dwC_%s'%self.FitNum][5],
              self.Pars['kexAB_%s'%self.FitNum][5], self.Pars['kexAC_%s'%self.FitNum][5],
              self.Pars['kexBC_%s'%self.FitNum][5], self.Pars['R1_%s'%self.FitNum][5],
              self.Pars['R1b_%s'%self.FitNum][5], self.Pars['R1c_%s'%self.FitNum][5],
              self.Pars['R2_%s'%self.FitNum][5], self.Pars['R2b_%s'%self.FitNum][5],
              self.Pars['R2c_%s'%self.FitNum][5]])

  #---------------------------#---------------------------#
  # Resets self.time variable given self.tInc and self.tMax
  #  Returns nothing
  #---------------------------#---------------------------#
  def SetTime(self):
    self.time = linspace(0.0, self.tMax, self.tMax / self.tInc)

  #---------------------------#---------------------------#
  # Takes in a sublist of ParseInp.ParInp
  # Converts this semi-raw data to the corresponding
  #  Parameter variables
  # e.g. Given the semi-raw:
  #      ['pB*','0.01','1e-5','0.15'] on index 0
  #  This function would match it to the self.Pars dict
  #   convert str to floats, match '*' shared parameter flag
  #   and assign values and boundaries to get:
  #      'pB_0' : [0.01,(1e-5,0.15),True]
  #---------------------------#---------------------------#
  def ConvertPars(self, InpParsList):
    for inp in InpParsList:
      # Check for shared parameter flag
      if '*' in inp[0]:
        shareFlag = True
      else: shareFlag = False

      # Check for fixed parameter flag
      if '!' in inp[0]:
        fixFlag = True
      else: fixFlag = False

      # Check for linear brute force flag
      if '@' in inp[0]:# and '$' not in inp[0]:
        bruteFlag = "lin"
      # Check for log brute force flag
      elif '$' in inp[0]:# and '@' not in inp[0]:
        bruteFlag = "log"
      else: bruteFlag = None

      ## Convert parameter name to that as can be recognized
      #  by the key of self.Pars
      #  i.e. strip '*' and add '_index#'
      pName = inp[0].replace('*','').replace('!','').replace('@','').replace('$','') + "_%s" % self.FitNum

      ## Assign self.Pars[pName] to correct values in inp
      if "Name" in pName:
        self.name = os.path.splitext(inp[1])[0]
      elif "lf" in pName:
        self.lf = float(inp[1])
      # Set magnetization alignment
      elif "alignmag" in pName.lower():
        self.AlignMag = inp[1].lower()
      # Define graphing axes limits
      elif "x-axis" in pName.lower():
        self.xmin = float(inp[1])
        self.xmax = float(inp[2])
      elif "y-axis" in pName.lower():
        self.ymin = float(inp[1])
        self.ymax = float(inp[2])
      # Set relaxation delays / etc
      elif "trelax" in pName.lower():
        self.tInc = float(inp[1])
        self.tMax = float(inp[2])
        self.SetTime() # Reset time numpy array
      else:
        # Assign initial value for given parameter
        self.Pars[pName][0] = float(inp[1])        
        # Assign (lower,upper) bounds tuple
        if len(inp) != 2:
          self.Pars[pName][1] = (float64(inp[2]), float64(inp[3]))
          if len(inp) >= 5:
            # Set brute force numbers
            self.Pars[pName][8] = int(inp[4])
        else: # If bounds not given, set to wide range
          if "p" in pName:
            self.Pars[pName][1] = (0.,1.)
          elif "dw" in pName:
            self.Pars[pName][1] = (-100.,100.)  
          elif "kex" in pName:
            self.Pars[pName][1] = (0.,100000.)   
          elif "R1" in pName:
            self.Pars[pName][1] = (0.,100.)
          elif "R2" in pName:
            self.Pars[pName][1] = (0.,250.)       
        # Assign variable shared parameter flag
        self.Pars[pName][2] = shareFlag
        # Assign variable fixed parameter flag
        self.Pars[pName][3] = fixFlag
        # Assign variable brute force flag for parameter
        self.Pars[pName][7] = bruteFlag

    ## Now match and assign the explicit parameters
    self.pB = self.Pars['pB_%s' % self.FitNum][0]
    self.pC = self.Pars['pC_%s' % self.FitNum][0]
    self.dwB = self.Pars['dwB_%s' % self.FitNum][0]
    self.dwC = self.Pars['dwC_%s' % self.FitNum][0]
    self.kexAB = self.Pars['kexAB_%s' % self.FitNum][0]
    self.kexAC = self.Pars['kexAC_%s' % self.FitNum][0]
    self.kexBC = self.Pars['kexBC_%s' % self.FitNum][0]
    self.R1 = self.Pars['R1_%s' % self.FitNum][0]
    self.R1b = self.Pars['R1b_%s' % self.FitNum][0]
    self.R1c = self.Pars['R1c_%s' % self.FitNum][0]
    self.R2 = self.Pars['R2_%s' % self.FitNum][0]
    self.R2b = self.Pars['R2b_%s' % self.FitNum][0]
    self.R2c = self.Pars['R2c_%s' % self.FitNum][0]

  #---------------------------#---------------------------#
  # Takes in path to a fit CSV file (each row a fit)
  #  and unpacks a row corresponding to its own FitNum
  #  to the self.Pars dictionary
  #---------------------------#---------------------------#
  def ConvertFits(self, FitPath):
    # Error handling flags and messages
    errBool, retMsg = False, ""
    idxfits = None
    idx = self.FitNum

    # Test input file
    if os.path.isfile(FitPath):
      FILE = open(FitPath, "rU")
      idxfits = [x.strip().split(",") for x in FILE]
      FILE.close()

      # Check to make sure the fit line defined is within the range
      #  of the number of rows in the fit csv file.
      if 0 < idx <= len(idxfits):
        try:
          # Name of this fit indice
          self.name = idxfits[idx][0]
          # # Assign correct fit num
          # self.FitNum = int(idxfits[idx][1])
          # Assign fit red. chi-square to local red chi square
          self.lRCS = float(idxfits[idx][2])
          # Set larmor frequency for the dw's
          self.lf = float(idxfits[idx][3])
          # Number of function evals, put to local
          self.lFE = float(idxfits[idx][4])
          # Unpack idxfits to Fit class parameters
          #  Fit parameter value -> Pars[name][5]
          self.Pars['pB_%s' % self.FitNum][5] = float(idxfits[idx][5])
          self.Pars['pC_%s' % self.FitNum][5] = float(idxfits[idx][6])
          self.Pars['dwB_%s' % self.FitNum][5] = float(idxfits[idx][7])
          self.Pars['dwC_%s' % self.FitNum][5] = float(idxfits[idx][8])
          self.Pars['kexAB_%s' % self.FitNum][5] = float(idxfits[idx][9])
          self.Pars['kexAC_%s' % self.FitNum][5] = float(idxfits[idx][10])
          self.Pars['kexBC_%s' % self.FitNum][5] = float(idxfits[idx][11])
          self.Pars['R1_%s' % self.FitNum][5] = float(idxfits[idx][12])
          self.Pars['R1b_%s' % self.FitNum][5] = float(idxfits[idx][13])
          self.Pars['R1c_%s' % self.FitNum][5] = float(idxfits[idx][14])
          self.Pars['R2_%s' % self.FitNum][5] = float(idxfits[idx][15])
          self.Pars['R2b_%s' % self.FitNum][5] = float(idxfits[idx][16])
          self.Pars['R2c_%s' % self.FitNum][5] = float(idxfits[idx][17])
          # Unpack idxfits to Fit class parameter errors
          #  Fit parameter error -> Pars[name][6]
          self.Pars['pB_%s' % self.FitNum][6] = float(idxfits[idx][18])
          self.Pars['pC_%s' % self.FitNum][6] = float(idxfits[idx][19])
          self.Pars['dwB_%s' % self.FitNum][6] = float(idxfits[idx][20])
          self.Pars['dwC_%s' % self.FitNum][6] = float(idxfits[idx][21])
          self.Pars['kexAB_%s' % self.FitNum][6] = float(idxfits[idx][22])
          self.Pars['kexAC_%s' % self.FitNum][6] = float(idxfits[idx][23])
          self.Pars['kexBC_%s' % self.FitNum][6] = float(idxfits[idx][24])
          self.Pars['R1_%s' % self.FitNum][6] = float(idxfits[idx][25])
          self.Pars['R1b_%s' % self.FitNum][6] = float(idxfits[idx][26])
          self.Pars['R1c_%s' % self.FitNum][6] = float(idxfits[idx][27])
          self.Pars['R2_%s' % self.FitNum][6] = float(idxfits[idx][28])
          self.Pars['R2b_%s' % self.FitNum][6] = float(idxfits[idx][29])
          self.Pars['R2c_%s' % self.FitNum][6] = float(idxfits[idx][30])
        # If column cannot be cast as a float
        except ValueError:
          retMsg += "\n  ERROR: Non-numeric values in fit line #%s\n" % self.FitNum
          errBool = True
      # If index is not in the fit file
      else:
        retMsg += "\n  ERROR: Fit slice to index number (%s) does not correspond to data.\n" % self.FitNum
        errBool = True
    # If file does not exist
    else:
      resMsg += "\n  ERROR: Fit file does not exist. (%s)\n" % FitPath
      errBool = True

    return errBool, retMsg

#########################################################################
# *Data class* is used to store and manipulate the R1rho data
#   to be fitted.
# Inherited by: Fits
#########################################################################
class Data:
  def __init__(self):
    # Nx[3-4] numpy array of R1ho data
    #  0: Corrected Offset (Hz)
    #  1: Spinlock power (Hz)
    #  2. R1rho (s^-1)
    #  3. R1rho Error (s^-1), optional
    self.R1pD = array([])
    # Bool to state if error is given
    self.Err = False
  
  #---------------------------#---------------------------#
  # Takes in a numpy array and assigns it to self.R1pD
  #  then checks to see if the shape of the data
  #  indicates there is error with the data (self.Err bool)
  #---------------------------#---------------------------#
  def ConvertData(self, DataArray):
    # Make sure it is an array
    self.R1pD = array(DataArray)

    if self.R1pD.shape[1] == 4:
      self.Err = True
    # If number of columns >4, assume 0:4 are data
    #  and the remaining columns are unwanted
    elif self.R1pD.shape[1] > 4:
      self.R1pD = self.R1pD[:,0:4]
    else:
      self.Err = False

#########################################################################
# *Fits class* is used to store fits of inherited data to the 
#  inherited parameters.
# self.name : Name of this fit object, name given in parameter input text
# self.FitNum : The integer fit number. This is given by the order and
#                 number of Fit class objects that is stored in the
#                 parent Global object self.gObs list.
#               This number is also used to identify the parameter key
#                 in the inherited Pars['key_IDnum'] dictionary
# 
class Fits(Parameters, Data):
  def __init__(self, FitNum):
    # Name of this fit class object. Given by parameter input file.
    self.name = None
    # Integer fit number to match with inherited Pars keys
    self.FitNum = FitNum
    # Reduced chi-square for global, polished, and local red. chi-squares
    self.gRCS, self.pRCS, self.lRCS = None, None, None
    # Assign number of function evaluations for each type of fit:
    #  global, polished, and local
    self.gFE, self.pFE, self.lFE = None, None, None
    # Dictionary to store NxM global fits of Pars values
    #  N is loops over global fitting algorithm (e.g. N global fits)
    #  M is length of parameters and reduced chi-square
    self.globalFits = {}
    # Dictionary to store NxM polished fits of Pars values
    # Polished fits are local fits using P0 from global fit
    #  N is loops over global->polished fitting algorithm (e.g. N global fits)
    #  M is length of parameters and reduced chi-square
    self.polishedFits = {}
    # Dictionary to store NxM local fits of Pars values
    #  N is loops over local fitting algorithm (e.g. N local fits)
    #  M is length of parameters and reduced chi-square
    self.localFits = {}
    # Inheritance
    Parameters.__init__(self, self.FitNum)
    Data.__init__(self)

#########################################################################
# *Global class* is used to globally store:
#   self.gObs : List of Fit class objects
#   self.gP0 : Numpy array of Global Parameter values to be passed
#              to fitting algorithm and to be updated by it.
#   self.keygP0 : List of keys whose order match self.gP0 parameters.
#   self.brutegP0 : Numpy array of arrays of iterated P0's for
#                   brute forced parameters
#   self.gBnds : Tuple of bounds matching the bounds of the parameters
#                found in self.gP0
#   self.gKeys : A list of all keys from all Pars dictionaries of all
#                Fit objects in self.gObs
#   self.gVar : A list of the parent names of variables used in 3-state
#               BM fitting.
#   self.gSared : A dictionary that maps the key of one parameter to the
#                 key of the parameter it shares its value with.
#                 A subset of self.gKeys
#   self.gFixed : A set of all fixed Par keys, a subset of self.gKeys.
#########################################################################
class Global():
  def __init__(self):
    # List to hold N-number of Fit class objects
    self.gObs = []
    # Numpy array to hold global parameter array
    self.gP0 = array([])
    # Numpy array to hold array of P0 arrays that span
    #  range of brute-forceable params
    self.brutegP0 = array([])
    # Set of Keys to map to self.gP0 parameter array
    self.keygP0 = set([])
    # Tuple to hold global bounds
    # Corresponds to parameters in gP0
    self.gBnds = ()
    # Global self.Pars keys
    self.gKeys = []
    # Global keys for Laguerre fitting
    #  Keys not inlcluded here will be excluded later
    self.gLagKeys = set(["pB","pC","dwB","dwC","kexAB","kexAC","R1","R2"])
    # Shared fit par basenames
    #            pB,pC,dwB,dwC,kexAB,kexAC,kexBC,R1,R1b,R1c,R2,R2b,R2c
    self.gVar = ["pB","pC","dwB","dwC","kexAB","kexAC","kexBC","R1","R1b","R1c","R2","R2b","R2c"]
    # Error base names
    #            pB,pC,dwB,dwC,kexAB,kexAC,kexBC,R1,R1b,R1c,R2,R2b,R2c
    self.gErr = ["pB_err","pC_err","dwB_err","dwC_err","kexAB_err","kexAC_err","kexBC_err","R1_err","R1b_err","R1c_err","R2_err","R2b_err","R2c_err"]
    # Shared fit parameters dictionary that maps key sub names to the parent key name
    #  Parent key names are the first keys, Key_0
    #    e.g. if pA_1 is shared with pA_0 and pA_2
    #    gShared = {'pA_1': 'pA_0', ..., etc}
    self.gShared = {}
    # Set of fixed parameter keys.
    #  e.g. set(['R1b', 'R2b', ..., etc])
    self.gFixed = set([])
    # Fit equation to use to fit this data
    #  bm = Bloch-McConnell
    #  lag = Laguerre 2-/3-state
    self.gFitEqn = "bm"
    # Type of fit to carry out
    self.FitType = "local"
    # Number of fit loops to run (essentially how many fit minimums you'll find)
    self.FitLoops = 1
    # Random start flag : if true, it will randomly select P0 values from
    #  a uniform distribution within the bounds of each parameter.
    self.rndStart = False    
  #---------------------------#---------------------------#
  # When called, this function will take the parameters
  #  contained in the gObs.Pars dict and parse out the
  #  shared and fixed parameters.
  # This function updates:
  #   gP0, keygP0, gBnds, gKeys, gShared, gFixed
  #---------------------------#---------------------------#  
  def MapGlobalP0(self):
    # Create a list of all the dictionary keys for
    #  each global object self.Pars dict
    self.gKeys = [x.Pars.keys() for x in self.gObs]
    # Flatten to 1D and include only those within the gVar dictionary
    self.gKeys = [x for x in list(it.chain.from_iterable(self.gKeys)) if x[:-2] in self.gVar]

    ## Match up shared and fixed parameters ##
    # Loop over parent variable names
    for var in self.gVar:
      # -- Shared Parameters -- #
      # Construct a temp list of keys sharing the same 
      #  parent name of var
      # e.g. pB = [pB_0, pB_1, pB_2,...,etc]
      td = [x for x in self.gKeys if var in x and len(var)+2 == len(x)]
      # List of parameter keys to keep if they are shared
      keep = []
      # Loop over td and match subkeys to shared
      for idx, key in enumerate(td):
        # If share flag is true for current parameter key of 
        #  self.Par dict, add it to keep list
        if self.gObs[idx].Pars[key][2] == True:
          keep.append(key)
      # If list of parameters to keep is not empty
      if len(keep) != 0:
        # Map current key in keep list to first key
        #  in the keep list.
        # This will be used later to remove fixed 
        #  parameters from the gP0 array
        #  e.g. 'Key1': 'Key1'
        #       'Key2': 'Key1'
        #       'Key3': 'Key1'
        for k in range(len(keep)):
          if k == 0:
            self.gShared[keep[k]] = keep[k]
          else:
            self.gShared[keep[k]] = keep[0]
            # Map share name in Pars[key]
            self.gObs[int(keep[k].split("_")[-1])].Pars[keep[k]][4] = keep[0]

    # -- Fixed Parameters -- #
    for key in self.gKeys:
      idx = int(key.split("_")[-1])
      # Check if current self.Par[key] fixed flag
      #  is true or not.
      if self.gObs[idx].Pars[key][3] == True:
        self.gFixed.add(key)
      # If doing a Laguerre fit, fix the ES relaxation rates
      if self.gFitEqn == "lag" and key[:-2] not in self.gLagKeys:
        self.gFixed.add(key)

    # -- Generate keygP0 set with shared and fixed parameters handled -- #
    # Iterate over all gKeys
    for key in self.gKeys:
      # If the key in gKeys is in the gShared dictionary,
      #  the add the corresponding item key to the keygP0 set.
      #  e.g. 'Key3':'Key1'
      #        -> do not add 'Key3', add 'Key1' to set
      #           if 'Key1' is redundant, it will be excluded
      #         This basically says that Key3 and Key1 are shared
      #          and thus only one of them needs to be passed to
      #          the fitting algorithm.
      if key in self.gShared:
        self.keygP0.add(self.gShared[key])
      # Check that the key in gKey is not fixed
      #  If it is, do not include it.
      elif key not in self.gFixed:
        self.keygP0.add(key)

    # Recast as list, don't want to risk order of set changing
    self.keygP0 = list(self.keygP0)

    # Generate global numpy parameter array based on items in self.keygP0
    #  This array will be passed to the objective function being used
    #   by the fitting algorithms
    self.gP0 = array([self.gObs[int(x.split("_")[-1])].Pars[x][0] for x in self.keygP0])
    # Generate brute-forced parameter array
    bruteArry = []
    for x in self.keygP0:
      par = self.gObs[int(x.split("_")[-1])].Pars[x]
      if par[7] is not None:
        lb = par[1][0]
        ub = par[1][1]
        nv = par[8]
        if par[7] == "lin":
          bruteArry.append(linspace(lb, ub, nv))
        elif par[7] == "log":
          lb = log10(lb)
          ub = log10(ub)
          # Check to make sure min and max value of 
          # gen array is set to lower or upper bounds of params given
          tArry = logspace(lb, ub, nv)
          tArry[tArry==tArry.max()] = par[1][1]
          tArry[tArry==tArry.min()] = par[1][0]
          # Append this array to the brute-force master array
          bruteArry.append(tArry)
      else:
        bruteArry.append([par[0]])
    self.brutegP0 = asarray(list(it.product(*bruteArry)))

    # Generate global bounds list based on bounds in self.keygP0
    #  NOTE: Transition from leastsqbnd to least_squares neccessitates
    #        migration of bounds from tuple of tuples to a list
    #        of lower and upper bounds list
    # Get lower and upper bounds to separate lists
    lbnds = list(self.gObs[int(x.split("_")[-1])].Pars[x][1][0] for x in self.keygP0)
    ubnds = list(self.gObs[int(x.split("_")[-1])].Pars[x][1][-1] for x in self.keygP0)
    self.gBnds = [lbnds, ubnds]

    # Int describing total size of data
    self.dataSize = 0
    # Int Number of parameters being fitted globally
    self.freePars = 0
    # Int Degrees of freedom
    self.dof = 0
  
  #---------------------------#---------------------------#
  # 'UnpackgP0' will unpack a given parameter numpy array
  #  and return:
  #    pB,pC,dwB,dwC,kexAB,kexAC,kexBC
  #    R1,R1b,R1c,R2,R2b,R2c
  #  values.
  # Function will handle shared and fixed parameters, and
  #  will set relaxation rates as equivalent if dictated so.
  #---------------------------#---------------------------# 
  def UnpackgP0(self, Params, ob):
    keyP0 = []
    # Map keys from self.keygP0 to updated Param numpy array
    tdict = {key:val for key, val in zip(self.keygP0, Params)}

    # Update ob.Pars[key][New Val] with value from
    #  Params array (by way of tdict). These are the
    #  updated fitted values
    for key in ob.Pars.keys():
      if key in tdict:
        ob.Pars[key][5] = tdict[ob.Pars[key][4]]
      elif key in self.gShared:
        ob.Pars[key][5] = tdict[ob.Pars[key][4]]
      # If key is not in tdict or gShared (ie. fixed and not being fitted),
      #  map ob.Pars[key][New Val] = ob.Pars[key][P0 Val]
      else:
        ob.Pars[key][5] = ob.Pars[key][0]

    # For fixed R1b/c and R2b/c values, set equal to R1 and R2 if they are 0.0
    for key in ob.Pars.keys():
      if key in self.gFixed and "R1b" in key or "R1c" in key and ob.Pars[key][0] == 0.0:
        ob.Pars[key][5] = ob.Pars["R1_%s" % ob.FitNum][5]
      elif key in self.gFixed and "R2b" in key or "R2c" in key and ob.Pars[key][0] == 0.0:
        ob.Pars[key][5] = ob.Pars["R2_%s" % ob.FitNum][5]

    # return pB,pC,dwB,dwC,kexAB,kexAC,kexBC,R1,R1b,R1c,R2,R2b,R2c
    return array([ob.Pars['pB_%s'%ob.FitNum][5], ob.Pars['pC_%s'%ob.FitNum][5],
                  ob.Pars['dwB_%s'%ob.FitNum][5], ob.Pars['dwC_%s'%ob.FitNum][5],
                  ob.Pars['kexAB_%s'%ob.FitNum][5], ob.Pars['kexAC_%s'%ob.FitNum][5],
                  ob.Pars['kexBC_%s'%ob.FitNum][5], ob.Pars['R1_%s'%ob.FitNum][5],
                  ob.Pars['R1b_%s'%ob.FitNum][5], ob.Pars['R1c_%s'%ob.FitNum][5],
                  ob.Pars['R2_%s'%ob.FitNum][5], ob.Pars['R2b_%s'%ob.FitNum][5],
                  ob.Pars['R2c_%s'%ob.FitNum][5]])
  #---------------------------#---------------------------#
  # 'UnpackErr' will unpack a given parameter error array
  #  and return:
  #    pB,pC,dwB,dwC,kexAB,kexAC,kexBC
  #    R1,R1b,R1c,R2,R2b,R2c ERRORs
  #  values.
  # Function will handle shared and fixed parameters, and
  #  will set relaxation rates as equivalent if dictated so.
  #---------------------------#---------------------------# 
  def UnpackErr(self, ParErr, ob):
    keyP0 = []
    # Map keys from self.keygP0 to updated Param numpy array
    tdict = {key:val for key, val in zip(self.keygP0, ParErr)}

    # Update ob.Pars[key][New Val] with value from
    #  Params array (by way of tdict). These are the
    #  updated fitted values
    for key in ob.Pars.keys():
      if key in tdict:
        ob.Pars[key][6] = tdict[ob.Pars[key][4]]
      elif key in self.gShared:
        ob.Pars[key][6] = tdict[ob.Pars[key][4]]
      # If key is not in tdict or gShared (ie. fixed and not being fitted),
      #  map ob.Pars[key][New Val] = ob.Pars[key][P0 Val]
      else:
        ob.Pars[key][6] = ob.Pars[key][0]

    # For fixed R1b/c and R2b/c values, set equal to R1 and R2 if they are 0.0
    for key in ob.Pars.keys():
      if key in self.gFixed and "R1b" in key or "R1c" in key and ob.Pars[key][0] == 0.0:
        ob.Pars[key][6] = ob.Pars["R1_%s" % ob.FitNum][6]
      elif key in self.gFixed and "R2b" in key or "R2c" in key and ob.Pars[key][0] == 0.0:
        ob.Pars[key][6] = ob.Pars["R2_%s" % ob.FitNum][6]

    # return pB,pC,dwB,dwC,kexAB,kexAC,kexBC,R1,R1b,R1c,R2,R2b,R2c
    return array([ob.Pars['pB_%s'%ob.FitNum][6], ob.Pars['pC_%s'%ob.FitNum][6],
                  ob.Pars['dwB_%s'%ob.FitNum][6], ob.Pars['dwC_%s'%ob.FitNum][6],
                  ob.Pars['kexAB_%s'%ob.FitNum][6], ob.Pars['kexAC_%s'%ob.FitNum][6],
                  ob.Pars['kexBC_%s'%ob.FitNum][6], ob.Pars['R1_%s'%ob.FitNum][6],
                  ob.Pars['R1b_%s'%ob.FitNum][6], ob.Pars['R1c_%s'%ob.FitNum][6],
                  ob.Pars['R2_%s'%ob.FitNum][6], ob.Pars['R2b_%s'%ob.FitNum][6],
                  ob.Pars['R2c_%s'%ob.FitNum][6]])

  #---------------------------#---------------------------#
  # 'DOF' simply calculates the total length of the
  #  total data being fitted, length of the parameters
  #  being fitted, and the number of degrees of freedom.
  #---------------------------#---------------------------# 
  def CalcDOF(self):
    # Calculate total data size
    for ob in self.gObs:
      self.dataSize += len(ob.R1pD)

    # Calculate number of floating parameters
    self.freePars = len(self.gP0)

    # Degrees of freedom
    self.dof = self.dataSize - self.freePars

  #---------------------------#---------------------------# 
  # 'GrabFitType' takes in fit type flags from the input
  #  parameter file and stores them
  #---------------------------#---------------------------# 
  def GrabFitType(self, ListOfFitTypes):
    for i in ListOfFitTypes:
      if "fittype" in i[0].lower():
        self.FitType = i[1].lower()
      # Get the fit equation
      #  Either Bloch-McConnell
      #   or Laguerre
      elif "fiteqn" in i[0].lower():
        if "bm" in i[1].lower():
          self.gFitEqn = "bm"
        elif "lag" in i[1].lower():
          self.gFitEqn = "lag"
      elif "numfits" in i[0].lower():
        try:
          self.FitLoops = int(i[1])
        except ValueError:
          print "Number of fit loops is not an integer, setting to 1 loop."
      elif "randomfitstart" in i[0].lower():
        if "y" in i[1].lower():
          self.rndStart = True
        else:
          self.rndStart = False
  
  #---------------------------#---------------------------# 
  # 'RandomgP0' generates a random global gP0 by doing a
  #  random uniform selection from within the corresponding
  #  parameter bounds.
  #---------------------------#---------------------------# 
  def RandomgP0(self):
    return array([uniform(*x) for x in self.gBnds])

  #---------------------------#---------------------------# 
  # 'WriteFits' writes out .csv files with all the fit
  #  parameters and fit values in the globalFits,
  #  polishedFits, and localFits dictionaries.
  # If output file exists already, just append to it.
  #  else, write out new file with title.
  # flag = 'global', 'polish', or 'local'
  #         It determines which fit dict to use.
  #---------------------------#---------------------------# 
  def WriteFits(self, outPath, ob, fitnum, flag):
    
    # Append additional keys to gVars
    outKeys = ["RedChiSq", "lf", "FuncEvals"] + self.gVar + self.gErr

    # Generate output path names for writing out .csv files
    gPath = os.path.join(outPath, "GlobalFits_%s.csv" % ob.name)
    pPath = os.path.join(outPath, "PolishedFits_%s.csv" % ob.name)
    lPath = os.path.join(outPath, "LocalFits_%s.csv" % ob.name)
    
    # Generate formatted output names for different fit types
    f2lPath = os.path.join(outPath, "2-state_Formatted_LocalFits_%s.csv" % ob.name)    
    f3lPath = os.path.join(outPath, "3-state_Formatted_LocalFits_%s.csv" % ob.name)  
    f2gPath = os.path.join(outPath, "2-state_Formatted_GlobalFits_%s.csv" % ob.name)    
    f3gPath = os.path.join(outPath, "3-state_Formatted_GlobalFits_%s.csv" % ob.name)  
    f2pPath = os.path.join(outPath, "2-state_Formatted_PolishedFits_%s.csv" % ob.name)    
    f3pPath = os.path.join(outPath, "3-state_Formatted_PolishedFits_%s.csv" % ob.name)  

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write out ob global fit parameters in order.
    if len(ob.globalFits) != 0 and flag == "global":
      # Check to see if a file exists already
      if not os.path.isfile(gPath): wo = "wb"
      else: wo = "ab"
      FILE = open(gPath, "ab")
      # If file exists already, don't write out header.
      if wo == "wb":
        FILE.write("Name,FitNum,")
        for key in outKeys:
          FILE.write(key + ",")
        FILE.write("\n")
      # Write out fit values that match keys
      FILE.write(str(ob.name) + "," + str(fitnum) + ",")
      for key in outKeys:
        FILE.write(str(ob.globalFits[fitnum][key]) + ",")
      FILE.write("\n")
      FILE.close()
    # Write out ob polished fit parameters in order.
    elif len(ob.polishedFits) != 0 and flag == "polish":
      # Check to see if a file exists already
      if not os.path.isfile(pPath): wo = "wb"
      else: wo = "ab"
      FILE = open(pPath, "ab")
      # If file exists already, don't write out header.
      if wo == "wb":
        FILE.write("Name,FitNum,")
        for key in outKeys:
          FILE.write(key + ",")
        FILE.write("\n")
      # Write out fit values that match keys
      FILE.write(str(ob.name) + "," + str(fitnum) + ",")
      for key in outKeys:
        FILE.write(str(ob.polishedFits[fitnum][key]) + ",")
      FILE.write("\n")
      FILE.close()
    # Write out ob local fit parameters in order.
    elif len(ob.localFits) != 0 and flag == "local":
      # Check to see if a file exists already
      if not os.path.isfile(lPath): wo = "wb"
      else: wo = "ab"
      FILE = open(lPath, "ab")
      # If file exists already, don't write out header.
      if wo == "wb":
        FILE.write("Name,FitNum,")
        for key in outKeys:
          FILE.write(key + ",")
        FILE.write("\n")
      # Write out fit values that match keys
      FILE.write(str(ob.name) + "," + str(fitnum) + ",")
      for key in outKeys:
        FILE.write(str(ob.localFits[fitnum][key]) + ",")
      FILE.write("\n")
      FILE.close()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ### Formatted write-out Global Fits
    if len(ob.globalFits) != 0 and flag == "global":
      fitob = ob.globalFits
      f2path = f2gPath
      f3path = f3gPath
    ### Formatted write-out Polished Fits
    elif len(ob.polishedFits) != 0 and flag == "polish":
      fitob = ob.polishedFits      
      f2path = f2pPath
      f3path = f3pPath
    ### Formatted write-out Local Fits
    elif len(ob.localFits) != 0 and flag == "local":
      fitob = ob.localFits
      f2path = f2lPath
      f3path = f3lPath

    if fitob[fitnum]["pC"] == 0.:
      ### Write-out 2-state Fits ###
      # Check to see if a file exists already
      FILE = open(f2path, "ab")
      # If file exists already, don't write out header.
      FILE.write("Fit #%s\n" % fitnum)
      FILE.write("%s Pars,Fit Value,Error\n" % ob.name)
      # Write out formatted values
      FILE.write("pB (%%),%.3f,%.3f\n"
        % (fitob[fitnum]["pB"]*1e2, fitob[fitnum]["pB_err"]*1e2))
      FILE.write("dwB (ppm),%.2f,%.2f\n"
        % (fitob[fitnum]["dwB"], fitob[fitnum]["dwB_err"]))
      FILE.write("kexAB (s^-1),%.0f,%.0f\n"
        % (fitob[fitnum]["kexAB"], fitob[fitnum]["kexAB_err"]))
      FILE.write("R1 (s^-1),%.2f,%.2f\n"
        % (fitob[fitnum]["R1"], fitob[fitnum]["R1_err"]))
      # Optional R1b write-out
      if fitob[fitnum]["R1b"] != fitob[fitnum]["R1"]:
        FILE.write("R1b (s^-1),%.2f,%.2f\n"
          % (fitob[fitnum]["R1b"], fitob[fitnum]["R1b_err"]))
      FILE.write("R2 (s^-1),%.2f,%.2f\n"
        % (fitob[fitnum]["R2"], fitob[fitnum]["R2_err"]))
      # Optional R2b write-out
      if fitob[fitnum]["R2b"] != fitob[fitnum]["R2"]:
        FILE.write("R2b (s^-1),%.2f,%.2f\n"
          % (fitob[fitnum]["R2b"], fitob[fitnum]["R2b_err"]))
      FILE.write("Red. Chi-sq,%.2f\n"
        % (fitob[fitnum]["RedChiSq"]))
      FILE.close()
    else:
      ### Write-out 3-state Fits ###
      # Check to see if a file exists already
      FILE = open(f3path, "ab")
      # If file exists already, don't write out header.
      FILE.write("Fit #%s\n" % fitnum)
      FILE.write("%s Pars,Fit Value,Error\n" % ob.name)
      # Write out formatted values
      FILE.write("pB (%%),%.3f,%.3f\n"
        % (fitob[fitnum]["pB"]*1e2, fitob[fitnum]["pB_err"]*1e2))
      FILE.write("pC (%%),%.3f,%.3f\n"
        % (fitob[fitnum]["pC"]*1e2, fitob[fitnum]["pC_err"]*1e2))
      FILE.write("dwB (ppm),%.2f,%.2f\n"
        % (fitob[fitnum]["dwB"], fitob[fitnum]["dwB_err"]))
      FILE.write("dwC (ppm),%.2f,%.2f\n"
        % (fitob[fitnum]["dwC"], fitob[fitnum]["dwC_err"]))
      FILE.write("kexAB (s^-1),%.0f,%.0f\n"
        % (fitob[fitnum]["kexAB"], fitob[fitnum]["kexAB_err"]))
      FILE.write("kexAC (s^-1),%.0f,%.0f\n"
        % (fitob[fitnum]["kexAC"], fitob[fitnum]["kexAC_err"]))
      # Optional kexBC write-out
      if fitob[fitnum]["kexBC"] != 0.:
        FILE.write("kexBC (s^-1),%.0f,%.0f\n"
          % (fitob[fitnum]["kexBC"], fitob[fitnum]["kexBC_err"]))
      FILE.write("R1 (s^-1),%.2f,%.2f\n"
        % (fitob[fitnum]["R1"], fitob[fitnum]["R1_err"]))
      # Optional R1b write-out
      if fitob[fitnum]["R1b"] != fitob[fitnum]["R1"]:
        FILE.write("R1b (s^-1),%.2f,%.2f\n"
          % (fitob[fitnum]["R1b"], fitob[fitnum]["R1b_err"]))
      # Optional R1c write-out
      if fitob[fitnum]["R1c"] != fitob[fitnum]["R1"]:
        FILE.write("R1c (s^-1),%.2f,%.2f\n"
          % (fitob[fitnum]["R1c"], fitob[fitnum]["R1c_err"]))
      FILE.write("R2 (s^-1),%.2f,%.2f\n"
        % (fitob[fitnum]["R2"], fitob[fitnum]["R2_err"]))
      # Optional R2b write-out
      if fitob[fitnum]["R2b"] != fitob[fitnum]["R2"]:
        FILE.write("R2b (s^-1),%.2f,%.2f\n"
          % (fitob[fitnum]["R2b"], fitob[fitnum]["R2b_err"]))
      # Optional R2c write-out
      if fitob[fitnum]["R2c"] != fitob[fitnum]["R2"]:
        FILE.write("R2c (s^-1),%.2f,%.2f\n"
          % (fitob[fitnum]["R2c"], fitob[fitnum]["R2c_err"]))
      FILE.write("Red. Chi-sq,%.2f\n"
        % (fitob[fitnum]["RedChiSq"]))
      FILE.close()

  #---------------------------#---------------------------# 
  # Takes in loop number (int), unpackaged paramter array (numpy) from 
  #  UnpackgP0 function (pB, pC, ..., R2c) and fit-type and updates
  #  the corresponding fit dictionaries with the fit parameters
  # loopNum : Integer number from loops over fitting algorithm + 1
  # unParams : UnpackgP0(Params) numpy array that has values in order;
  #            [pB,pC,dwB,dwC,kexAB,kexAC,kexBC,R1,R1b,R1c,R2,R2b,R2c]
  # redChiSq : Reduced chi-square.
  # funcEvals : Number of function evaluations.
  # FitType : Flag to denote global, polished, or local fits
  #---------------------------#---------------------------# 
  def UnPackFits(self, loopNum, unParams, redChiSq, funcEvals, FitType, ob, errPars = None):
    # Unpack and update global fits dictionary
    if FitType.lower() == "global":
      # Assign global red. chi-square for this fit
      # Assign number of function evals
      ob.gRCS, ob.gFE = redChiSq, funcEvals
      ob.globalFits[loopNum] = {x:y for x,y in zip(self.gVar, unParams)}
      ob.globalFits[loopNum].update({'RedChiSq' : redChiSq, 'FuncEvals' : funcEvals, "lf" : ob.lf})
      if errPars is None:
        # Add error names, but for now give NO ERROR
        ob.globalFits[loopNum].update({x:0.0 for x in self.gErr})
      else:
        ob.globalFits[loopNum].update({x:y for x,y in zip(self.gErr, errPars)})
    # Unpack and update polished fits dictionary
    elif FitType.lower() == "polish":
      # Assign polished red. chi-square for this fit
      # Assign number of function evals
      ob.pRCS, ob.pFE = redChiSq, funcEvals
      ob.polishedFits[loopNum] = {x:y for x,y in zip(self.gVar, unParams)}
      ob.polishedFits[loopNum].update({'RedChiSq' : redChiSq, 'FuncEvals' : funcEvals, "lf" : ob.lf})
      if errPars is None:
        # Add error names, but for now give NO ERROR
        ob.polishedFits[loopNum].update({x:0.0 for x in self.gErr})
      else:
        ob.polishedFits[loopNum].update({x:y for x,y in zip(self.gErr, errPars)})
    # Unpack and update local fits dictionary
    else:
      # Assign local red chi-square for this fit
      # Assign number of function evals
      ob.lRCS, ob.lFE = redChiSq, funcEvals
      ob.localFits[loopNum] = {x:y for x,y in zip(self.gVar, unParams)}
      ob.localFits[loopNum].update({'RedChiSq' : redChiSq, 'FuncEvals' : funcEvals, "lf" : ob.lf})
      if errPars is None:
        # Add error names, but for now give NO ERROR
        ob.localFits[loopNum].update({x:0.0 for x in self.gErr})
      else:
        ob.localFits[loopNum].update({x:y for x,y in zip(self.gErr, errPars)})