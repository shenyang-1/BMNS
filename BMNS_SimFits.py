import os
import numpy as np
import pandas as pd
import itertools as it
import BMNS_Errors as bme
#########################################################################
# BMNS_SimFits : Simulates R1rho fit curves given parameters
#########################################################################
class SimFit:
  def __init__(self):
    self.Params = ["Name","lf","pB","pC","dwB","dwC","kexAB","kexAC","kexBC","R1","R2","R1b","R1c","R2b","R2c"]
    self.curDir = os.getcwd()
    # -- Simulation SLPs and Offsets numpy array -- #
    # Col0 = Offsets
    # Col1 = SLPs
    self.sloff = None
    self.slon = None
    self.slonoff = None
    self.r1perrpct = 0.0

    # -- Simulation Decaying Intensity values -- #
    self.t0, self.tmax = None, None
    self.vdlist = None
    self.decerr = 0.0

    # -- Plotting variables and their attributes -- #
    self.pltvar = {
      "plot" : "line", # Plot type - symbol or lines or both
      "line" : ["-", 1.5], # Line type
      "symbol" : ["o", 10], # Symbol type
      "r1p_x" : [None, None], # Lower, upper limits of x-dimension R1p
      "r1p_y" : [None, None], # Lower, upper limits of y-dimension R1p
      "r2eff_x" : [None, None], # Lower, upper limits of x-dimension R2eff
      "r2eff_y" : [None, None], # Lower, upper limits of y-dimension R2eff
      "on_x" : [None, None], # Lower, upper limits of x-dimension OnRes
      "on_y" : [None, None], # Lower, upper limits of y-dimension OnRes
      "axis_fs" : [16, 16], # Plots axes font size
      "label_fs" : [18, 18] # Label axes font size
      }
    # -- BM Parameter variables and their values -- #
    self.fitpars = {
      "lf" : None, # MHz
      "alignmag" : "auto", # Mag alignment
      "te" : None, # Kelvin
      "pa" : None, # Probability
      "pb" : None, # Probability
      "pc" : None, # Probability
      "dwb" : None, # ppm
      "dwc" : None, # ppm
      "kexab" : None,
      "kexac" : None,
      "kexbc" : None,
      "k12" : None,
      "k21" : None,
      "k13" : None,
      "k31" : None,
      "k23" : None,
      "k32" : None,
      "r1" : None,
      "r2" : None,
      "r1b" : None,
      "r2b" : None,
      "r1c" : None,
      "r2c" : None
    }
  #########################################################################
  # PreSim - Reads argument command line and checks for all required
  #          parameters and maps them to class values
  #  Input:
  #   - argv with input path
  #  Output:
  #   -
  #########################################################################
  def PreSim(self, argv):
    # Get path to input file
    inpPath = os.path.join(self.curDir, argv[2])
    self.prRawInp(inpPath)

  #########################################################################
  # prRawInp - Parse Raw input file
  #  Input:
  #   - Path to raw input file
  #  Result:
  #   - Assigns parsing jobs to raw input file
  #########################################################################
  def prRawInp(self, inpPath):
    inpRaw = []
    # Open input parameter file
    with open(inpPath, 'r') as f:
        # Split input parameter file by delimiter '+' to lists
        for key,group in it.groupby(f,lambda line: line.startswith('+')):
            if not key:
                inpRaw.append([x.strip().split(" ") for x in group if "#" not in x and len(x) > 0])
    # Iterate over blocks of input parameters in parsed raw input
    # Need to extract the following blocks:
    #  1. SLP's and Offset generation and/or read-in
    #  2. Plot
    #  3. Name
    #  4. (Optional) Decay - Monoexponential decays
    for b in inpRaw:
      # Parse spinlock powers and offsets to be used in the simulation
      if "sloff" in b[0][0].lower():
        self.prSLOff(b)
      # Parse plotting variables
      elif "plot" in b[0][0].lower():
        self.prPlotInp(b)
      # Parse input parameters for BM simulation
      elif "params" in b[0][0].lower():
        self.prParInp(b)
        self.checkFitPars()
      # Parse monoexponential decays, if they exist
      elif "decay" in b[0][0].lower():
        self.prDecay(b)

  #########################################################################
  # prDecay - Parse SL duration values for simulating monoexponential decays
  #  Input:
  #   - Decay pars input block from parsed raw input file (list of lists)
  #     Read-in decay file is just a newline separated text file of decays in sec
  #  Result:
  #   - Assigns decay times and errors to self values
  #########################################################################
  def prDecay(self, pdeb):
    readFlag = False
    # Loop over monoexponential decay block
    for i in pdeb:
      # Check for decay vdlist
      if i[0].lower() == "read":
        readFlag = True
        # If read flag is defined as a value as non-false
        if len(i) >= 2 and i[1].lower() != "false":
          readPath = os.path.join(self.curDir, i[1])
          # Check to make sure decay text file, if not exit program.
          if not os.path.isfile(readPath):
            bme.HandleErrors(True, "\nVdlist path is defined but does not exist.\n %s\n" % readPath)
          else:
            # Read text to numpy array
            dec = np.genfromtxt(readPath, dtype=float)
            # Strip nan values
            self.vdlist = dec[~np.isnan(dec)]

  #########################################################################
  # prParInp - Parse BM parameters input block from raw input and parses it
  #            Maps values (from input csv or text) to dict of BM pars
  #  Input:
  #   - BM pars input block from parsed raw input file (list of lists)
  #  Result:
  #   - Assigns values to all the BM simulation parameters
  #     If defined, a BM fit CSV file is read in in lieu of explicitly
  #     defined variables in the input txt
  #########################################################################
  def prParInp(self, parb):
    readFlag = False
    # Loop over bm input parameter block
    for i in parb:
      # Check for BM fit csv
      if i[0].lower() == "read":
        readFlag = True
        # If read flag is defined as a value as non-false
        if len(i) >= 2 and i[1].lower() != "false":
          readPath = os.path.join(self.curDir, i[1])
          # Check to make sure BM fit csv exists, if not exit program.
          if not os.path.isfile(readPath):
            bme.HandleErrors(True, "\nBM fit path is defined but does not exist.\n %s\n" % readPath)
          else:
            # Read csv file to pandas dataframe
            bf = pd.read_csv(readPath, sep=',')
            # Reassign col headers to lowercase
            bf.columns = [x.lower() for x in bf.columns]
            # Check through input file and find values that
            #  correspond to self.fitpars from the bm fit pandas df
            for v in set(self.fitpars.keys()) & set(bf.columns):
              self.fitpars[v] = bf[v][0]

      elif i[0].lower() in self.fitpars.keys() and len(i) == 2:
        v = i[0].lower()
        # Check to make sure parameter is a value
        #  Assign if it is, except alignmag
        if v != "alignmag":
          try:
            self.fitpars[v] = float(i[1])
          except ValueError:
            bme.HandleErrors(True, "\nBM fit parameter is not a value.\n %s\n" % i[1])
        else:
          if i[1].lower() not in ["auto", "gs", "avg"]:
            self.fitpars[v] = "auto"
          else:
            self.fitpars[v] = i[1].lower()

  #########################################################################
  # checkFitPars - Checks to make sure BM fit pars are sufficient for sims
  #########################################################################
  def checkFitPars(self):
    if self.fitpars['pb'] is None and self.fitpars['pc'] is None:
      bme.HandleErrors(True, "\npB and pC have not been defined.\n")
    if self.fitpars['kexab'] is None and self.fitpars['kexac'] is None:
      bme.HandleErrors(True, "\nkexAB and kexAC have not been defined.\n")
    if (self.fitpars['r1'] is None or self.fitpars['r1b'] is None
        or self.fitpars['r1c'] is None):
      bme.HandleErrors(True, "\nR1, R1b, or R1c has not been defined.\n")
    if (self.fitpars['r2'] is None or self.fitpars['r2b'] is None
        or self.fitpars['r2c'] is None):
      bme.HandleErrors(True, "\nR2, R2b, or R2c has not been defined.\n") 
    if self.fitpars['lf'] is None:
      bme.HandleErrors(True, "\nLarmor frequency has not been defined.\n")      
  #########################################################################
  # prPlotInp - Parse plot input block from raw input and parses it
  #           Maps values to local plotting parameters for 
  #           axes limits and font sizes
  #  Input:
  #   - Plot input block from parsed raw input file (list of lists)
  #  Result:
  #   - Assigns values to all the variable plotting parameters
  #########################################################################
  def prPlotInp(self, plob):
    for i in plob:
      if i[0].lower() != "plot" and len(i) == 3:
        # var name
        v = i[0].lower()
        # Check for numerical values if not line type or symbol
        if v != "line" and v != "symbol":
          # Check variable numbers are nubmers
          tvars = self.checkNums(i[1:])
          # assign variables to dict with ranges
          self.pltvar[v] = tvars
        else:
          # Make sure line/symbol size is numerical
          try:
            symsize = float(i[2])
          except ValueError:
            bme.HandleErrors(True, "\nLine or Symbol size non-numeric\n")
          self.pltvar[v] = [i[1], symsize]
      # Assign plotting type
      elif i[0].lower() == "plot" and len(i) == 2:
        self.pltvar[i[0].lower()] = i[1]

  #########################################################################
  # checkNums - Takes in an N-length list and checks that the values are
  #             not non-numerical
  #  Input:
  #   - List of values
  #  Result:
  #   - Returns a list of floats, if possible. Else, stops program
  #########################################################################
  def checkNums(self, numArr):
    # Check that all values in the array are numbers
    try:
      for i in numArr:
        float(i)
    except ValueError:
      bme.HandleErrors(True, "\nNot enough parameters defined for plotting params\n")

    return np.asarray(numArr).astype(float)

  #########################################################################
  # prSLOff - Parse SL Offset block from raw input and parses it
  #           Generates numpy array of SLPs and Offsets for on-res and
  #           off-res
  #  Input:
  #   - SL/Offset block from parsed raw input file (list of lists)
  #  Result:
  #   - Reads and/or generates SLPs/offsets to be assigned to 
  #     self.slon - onres numpy array
  #     self.sloff - offres numpy array
  #########################################################################
  def prSLOff(self, slob):
    readPath = None
    # Iterate over sloffset block and extract
    # - Read value (i.e. read in .csv with SLP offsets pre-defined)
    #    If null, then generate with following ranges
    # - 'on' onres values: low, upper, N-pts
    # - 'off' offres values: SLP, low, upper, N-points
    for i in slob:
      # Check for read flag to read in SLP-offset values from a CSV
      #  Here col0 = offsets (Hz) and col1 = SLP (Hz)
      if i[0].lower() == "read":
        # If read flag is defined as a value as non-false
        if len(i) >= 2 and i[1].lower() != "false":
          readPath = os.path.join(self.curDir, i[1])
          # Check to make sure SLPOffs csv exists, if not just ignore.
          if not os.path.isfile(readPath):
            pass
            # bme.HandleErrors(True, "\nSLP/Offset path is defined but does not exist.\n %s\n" % readPath)
          else:
            # Read csv to numpy array
            rawsloff = np.genfromtxt(readPath, dtype=float, delimiter=',')
            # Strip nan values
            rawsloff = rawsloff[~np.isnan(rawsloff).any(axis=1)]
            # Split to on-res vs offres
            self.sloff = rawsloff[rawsloff[:,0] != 0.]
            self.slon = rawsloff[rawsloff[:,0] == 0.]
      # Generate on-res SLPs
      elif i[0].lower() == "on" and len(i) >= 4:
        # Check that lower, upper, and N are numbers
        try:
          low, hi = float(i[1]), float(i[2])
          # Swap low/high if low is > high
          if low > hi: low, hi = hi, low
          # Check number parameters
          numv = int(i[3])
        except ValueError:
          bme.HandleErrors(True, "\nBad on-res SLP generation parameters\n %s" % " ".join(i))
        # Generate array of these values with 0 set to offset
        ton = np.array([[0., x] for x in np.linspace(low, hi, numv)])
        # append to existing array if it already exists
        if self.slon is not None:
          self.slon = np.append(self.slon, ton, axis=0)
        # else assign onres array to this array
        else:
          self.slon = ton
      elif i[0].lower() == "on" and len(i) < 4:
        bme.HandleErrors(True, "\nNot enough parameters defined for on-res SLP generation\n")
      # Generate off-res SLPs
      elif i[0].lower() == "off" and len(i) >= 5:
        # Check that lower, upper, and N are numbers
        try:
          # Assign slp
          slp = float(i[1])
          # Assign lower/upper offset limits
          low, hi = float(i[2]), float(i[3])
          # Swap low/high if low is > high
          if low > hi: low, hi = hi, low
          # Check number parameters
          numv = int(i[4])
        except ValueError:
          bme.HandleErrors(True, "\nBad off-res generation parameters\n %s" % " ".join(i))
        # Generate array of these values with slp set to SLP
        toff = np.array([[x, slp] for x in np.linspace(low, hi, numv)])
        # append to existing array if it already exists
        if self.sloff is not None:
          self.sloff = np.append(self.sloff, toff, axis=0)
        # else assign onres array to this array
        else:
          self.sloff = toff
      # Handle bad number of sloff generation parameters
      elif i[0].lower() == "off" and len(i) < 5:
        bme.HandleErrors(True, "\nNot enough parameters defined for off-res SLP generation\n")
      # Get error percentage for corruption of R1rho values
      elif i[0].lower() == "error":
        if len(i) == 2:
          try:
            self.r1perrpct = float(i[1])
          except ValueError:
            bme.HandleErrors(True, "\nError percentage does not exist\n")
    # Combine both on and off-res
    self.slonoff = np.append(self.slon, self.sloff, axis=0)

