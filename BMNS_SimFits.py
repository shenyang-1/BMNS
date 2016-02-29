#########################################################################
# BMNS_SimFits : Simulates R1rho fit curves given parameters
#########################################################################

class SimFit:
  def __init__(self):
    self.Params = ["Name","lf","pB","pC","dwB","dwC","kexAB","kexAC","kexBC","R1","R2","R1b","R1c","R2b","R2c"]

  #########################################################################
  # PreSim - Reads argument command line and checks for all required
  #          parameters and 
  #  Input:
  #   -
  #  Output:
  #   -
  #########################################################################
  def PreSim(self, argv):
    print argv

  #########################################################################
  # SimFit - Main function
  #  Input:
  #   -
  #  Output:
  #   -
  #########################################################################
  def SimFit(self):
    print "weee3"