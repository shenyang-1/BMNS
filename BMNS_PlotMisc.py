import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import BMNS_Errors as bme
import BMNS_MathFuncs as mf
import BMNS_Stats as sf
#########################################################################
# BMNS_PlotMisc : Miscellaneous plotting functions
#########################################################################
#---------------------------#---------------------------#
# 'PlotBrute' plots brute-forced fitting parameters as a 2D contour plot of
#   red. chi square and normalized red. chi square
#---------------------------#---------------------------#
def PlotBrute(args, outPath):
    # Plotting default settings
    rcParams['pdf.fonttype'] = 42
    rcParams['font.sans-serif'] = 'arial'
    # -- Set axes font sizes -- #
    rcParams.update({'font.size': 32})
    fitflag = sys.argv[1]
    fitp = os.path.join(outPath, sys.argv[2])
    # Check that fit file exists
    if os.path.isfile(fitp):
        # Read fit csv file to pandas dataframe
        rd = pd.read_csv(fitp)
        # Set column headers to lowercase
        rd.columns = map(str.lower, rd.columns)
        # headers
        hdr = list(rd.columns.values)
        # Check to make sure enough args given
        if len(sys.argv) >= 5:
            # Check that args given are in dataframe
            # Get values for plotting
            p1 = sys.argv[3].lower()
            p2 = sys.argv[4].lower()
            # Check that parameters for 2D plot exist
            if p1 not in hdr or p2 not in hdr:
                bme.HandleErrors(True, "\nParameter names given are not in fit data.\n")

            # Assumes last row is the 'best fit' and not what we want to compare here
            #  Removes last row of dataframe
            rd = rd.ix[:rd.shape[0]-2]
            # Assign X,Y,Z data
            x = np.asarray(rd[p1])
            y = np.asarray(rd[p2])
            z = np.asarray(rd['redchisq'])

            # Calculate the orders of magnitude difference in parameters
            om1 = mf.OrdMag(x.min(), x.max())
            om2 = mf.OrdMag(y.min(), y.max())

            # Scale red.chi-square
            # Calculate delta red. chi-square from minimum red. chi-square value
            dRCS = np.array([zv - z.min() for zv in z])
            # Scale Z values as if they were nIC weights
            zs = np.array([sf.cnICwt(drs, dRCS) for drs in dRCS])
            zs = np.array([(zi-zs.min())/(zs.max() - zs.min()) for zi in zs])

            # -- Plot Red. Chi-Square Contours -- #
            fig = plt.figure(figsize=(12, 8), dpi=200)
            CS = plt.tricontourf(x,y,z, 50, cmap=plt.cm.coolwarm)
            if fitflag == "-plotbrute":
                cbar = plt.colorbar(CS)
                cbar.ax.set_ylabel(r'$\overline{\chi}^2$', fontsize=24, rotation=0,labelpad=15)
                plt.title(r'$\overline{\chi}^2\,plot$',fontsize=24)
                plt.xlabel(r'$%s$' % p1, fontsize=24)
                plt.ylabel(r'$%s$' % p2, fontsize=24)
            # plt.xticks(np.linspace(round(x.min(), -3), round(x.max(), -3), 5))
            plt.xlim(x.min(), x.max())
            # Set log scales as needed if scale is greater than 2 orders of magnitude
            if om1 > 2:
                plt.xscale("log")
            if om2 > 2:
                plt.yscale("log")
            if fitflag == "-plotbrute":
                plt.tight_layout(2)
            plt.savefig("RedChiSqContour_%s_vs_%s.pdf" % (p1,p2), transparent = True)
            plt.close(fig)
            plt.clf()

            # -- Plot Normalized Red. Chi-Square Contours -- #
            fig = plt.figure(figsize=(12, 8), dpi=200)
            CS = plt.tricontourf(x,y,zs, 50, cmap=plt.cm.coolwarm)
            if fitflag == "-plotbrute":
                cbar = plt.colorbar(CS)
                cbar.ax.set_ylabel(r'$Best\,fit\,probability$', fontsize=24, rotation=-90,labelpad=30)
                plt.title(r'$Normalized\,\overline{\chi}^2\,weights\,plot$',fontsize=24)
                plt.title(r'$Z_i(\overline{\chi}^2)=\frac{e^{-0.5\cdot\Delta\overline{\chi}^2_i}}{\sum_{k=1}^{K}e^{-0.5\cdot\Delta\overline{\chi}^2_k}}$',
                  fontsize=24, y=1.12)
                plt.xlabel(r'$%s$' % p1, fontsize=24)
                plt.ylabel(r'$%s$' % p2, fontsize=24)

            # plt.xticks(np.linspace(round(x.min(), -3), round(x.max(), -3), 5))
            plt.xlim(x.min(), x.max())
            # Set log scales as needed if scale is greater than 2 orders of magnitude
            if om1 > 2:
                plt.xscale("log")
            if om2 > 2:
                plt.yscale("log")
            if fitflag == "-plotbrute":
                plt.tight_layout(2)
            plt.savefig("WeightedContour_%s_vs_%s.pdf" % (p1,p2), transparent = True)
            plt.close(fig)
            plt.clf()

        # Not enough parameters passed from command line
        else:
            bme.HandleErrors(True, "\nNot enough command arguments defined.\n")
    # Fit file does not exist
    else:
        bme.HandleErrors(True, "\nBM fit path does not exist.\n %s\n" % fitp)
