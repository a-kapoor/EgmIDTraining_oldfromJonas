# EgmIDTraining
The tools to train BDT Electron and Photon Identifications for CMS.

## How to train an Electron MVA ID

### Overview

The procedure splits up in a few fundamental steps:

1. Make training ntuple with CMSSW
2. Train the ID with xgboost
3. Determine working points
4. Generate configuration files to integrate ID in CMSSW
5. Make validation ntuple with CMSSW
6. Draw performance plots and generate summary slides

Only step 1 and 4 require interaction with CMSSW, the other steps can be done offline.

### Step 1 - Making Ntuples for Training

Start by setting up the CMSSW area (you might want to change the release name):

> `scram project CMSSW_10_1_0`
> `cd CMSSW_10_1_0/src`
> `cmsenv`

Checkout the needed packages:

> `git cms-addpkg RecoEgamma/ElectronIdentification`
