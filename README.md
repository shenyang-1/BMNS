# Bloch-McConnell 2-/3-state Fitting and Simulation Program

## Introduction
This program is designed to fit and simulate experimental nuclear magnetic resonance (NMR) rotating frame relaxation-dispersion (R1 rho) data by solving the Bloch-McConnell partial differential equations describing chemical exchange in an external magnetic field using eigenvalue decomposition. 

## Installation of BMNS in a Conda environment
Create a new Python 2.7 Conda environment and activate it.

conda create --name bmns python=2.7

source activate bmns

Now install package dependencies using:
conda install numpy matplotlib scipy pandas joblib
conda install -c conda-forge uncertainties 
