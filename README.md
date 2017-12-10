# Bloch-McConnell N-State fitting and simulation

## Introduction
This program is designed to fit and simulate experimental nuclear magnetic resonance (NMR) rotating frame relaxation-dispersion (R1rho) data by solving the Bloch-McConnell partial differential equations describing chemical exchange in an external magnetic field using eigenvalue decomposition.

### Fitting Features
 * Can fit 1D R1rho data using the Bloch-McConnell equations or with the Laguerre algegraic approximations.
 * Can fit directly to decaying intensities instead of R1rho (plotting functionality limited).
 * Currently supports 2- and 3-state fitting.
 * Reports fit statistics metrics such as reduced chi-square, adjusted R-square, TSS, etc.
 * Fits for excited state (ES) populations, exchange rates, ES free-energies and transition state barriers, and intrinsic rate constants (R1/R2)
 * Error in fitted parameters can be estimated using standard-error or by Monte-Carlo resampling if given error in R1rho values.
 * Can fit multiple datasets together using combinations of shared parameters.
   * e.g. Datasets A and B fitted sharing pB and kexAB
 * Plots R1rho and R2eff fits as well as residuals.
 * BM fits can be fit assuming ground-state or average-state alignment, or 'auto' which will fit using the best alignment given by the exchange regime.
 * Supports grid-search over N-dimensional parameter space in log or linear spaced increments and plotting the search results.
 * Supports model comparison using Akaike's Information Criterion (AIC) and Bayesian Information Criterion (AIC).

<img src="https://github.com/IsaacJK/BMNS/blob/master/Examples/Fit-Indv-2State/R1rho_Figs_dG6C1p-mc_1-1.png?raw=true" width="400" height="300">
<img src="https://github.com/IsaacJK/BMNS/blob/master/Examples/Fit-Indv-2State/R2eff_Figs_dG6C1p-mc_1-1.png?raw=true" width="400" height="300">

### Simulation Features
 * Simulate decaying intensities, R1rho, R2eff, and 3D projections of decaying GS/ES vectors.
 * Noise corrupt intensities or R1rho.

## Installation of BMNS in a Conda environment
It is recommended that you use an Anaconda environment to run BMNS.
The *conda_bmns.sh* bash script will install the required Python dependencies.

```bash
bash conda_bmns.sh bmns
source activate bmns
```

## Running examples
Examples for how to fit and simulate R1rho data can be found within the *Examples* folder.

```bash
python BMNS.py -fit Examples/Fit-Indv-2State/Fit-InputPars.txt Examples/Fit-Indv-2State/ Output_2state/
```

## ToDo
 * Finish writing function docstrings
 * Port to Python 3
