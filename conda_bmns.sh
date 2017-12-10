#!/bin/bash

# Create Conda environment for installing
# BMNS with Python 2.7
if [ -z "$1" ]
then
    echo "Please specify a name for your conda environment."
fi

if [ -z "$python_version" ]
then
    echo "Using python 2.7 by default"
    export python_version=2.7
fi

export envname=$1
echo $envname
conda create -y --name $envname python=$python_version
source activate $envname
conda install -y -q -c numpy matplotlib scipy joblib pandas
conda install -c conda-forge uncertainties
