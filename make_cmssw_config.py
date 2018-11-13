from config import cfg
import json
from os.path import join
import os
import argparse
import sys

parser = argparse.ArgumentParser(description='Find working points.')
parser.add_argument('--TMVA' , action='store_true' , help = 'take the TMVA training')
args = parser.parse_args()

wp_file = "working_points_bayes_opt.json"
weight_file_name = "model_optimized/weights.xml.gz"
out_dir_base = join(cfg["out_dir"], cfg['submit_version'])

if args.TMVA:
    weight_file_name = "legacy/BDT.weights.xml.gz"
    wp_file = "working_points_TMVA.json"

cmssw_dir = join(cfg["cmssw_dir"], cfg['submit_version'])

vid_dir = join(cfg["cmssw_dir"], cfg['submit_version'], "src/RecoEgamma/ElectronIdentification/python/Identification")

ntp_new_file = join(cfg["cmssw_dir"], cfg['submit_version'], "src/RecoEgamma/ElectronIdentification/python/Training", cfg['ntuplizer_cfg'].split('/')[-1])

if not os.path.exists(vid_dir):
    os.makedirs(vid_dir)

if not os.path.exists(join(cfg["cmssw_dir"], cfg['submit_version'], "src/RecoEgamma/ElectronIdentification/python/Training")):
    os.makedirs(join(cfg["cmssw_dir"], cfg['submit_version'], "src/RecoEgamma/ElectronIdentification/python/Training"))


with open(join(out_dir_base, wp_file), 'r') as f:
    wp_dict = json.load(f)

# To add to the ntuplizer
ntp_cff_entries = []
ntp_eleMVAs_entries = []
ntp_eleMVALabels_entries = []
ntp_eleMVAValMaps_entries = []
ntp_eleMVAValMapLabels_entries = []

for idname in cfg["cmssw_cff"]:

    weight_files_cmssw_dir = join("RecoEgamma/ElectronIdentification/data/MVAWeightFiles", idname)
    data_dir = join(cfg["cmssw_dir"], cfg['submit_version'], "src", weight_files_cmssw_dir)

    cfg_cmssw = cfg["cmssw_cff"][idname]
    cfg_train = cfg["trainings"][idname]
    cfg_wp = cfg["working_points"][idname]

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for wp in cfg_wp:
        cats = cfg_wp[wp]["categories"]
        break

    n_cat = len(cats)

    for i in range(n_cat):
        weight_file_from = join(out_dir_base, idname, cats[i], weight_file_name)
        weight_file_to = join(data_dir, "{0}.weights.xml.gz".format(cats[i]))
        os.system("cp {0} {1}".format(weight_file_from, weight_file_to))

    ntp_cff_entries.append(cfg_cmssw['file_name'])

    with open(join(vid_dir, cfg_cmssw['file_name']), 'w') as cff:
        # imports
        cff.write('import FWCore.ParameterSet.Config as cms\n')
        cff.write('from RecoEgamma.ElectronIdentification.Identification.mvaElectronID_tools import *\n')
        cff.write('from os import path\n')
        cff.write('\n')
        # tag
        tag = cfg_cmssw['mvaTag']
        ntp_eleMVAValMaps_entries.append(tag + "Values")
        ntp_eleMVAValMaps_entries.append(tag + "RawValues")
        cff.write('mvaTag = "{0}"\n'.format(tag))
        cff.write('\n')

        # weight files
        cff.write('weightFileDir = "{0}"\n'.format(weight_files_cmssw_dir))
        cff.write('\n')
        cff.write('mvaWeightFiles = cms.vstring(\n')
        for i in range(n_cat):
            cff.write('     path.join(weightFileDir, "{0}.weights.xml.gz"), # {0}\n'.format(cats[i]))
        cff.write('     )\n')
        cff.write('\n')

        # categories
        cff.write('categoryCuts = cms.vstring(\n')
        for i in range(n_cat):
            cutstring = cfg["trainings"][idname][cats[i]]['cut']
            cutstring = cutstring.replace("ele_", "")
            cutstring = cutstring.replace("scl_", "superCluster.")
            cff.write('     "{0}", # {1}\n'.format(cutstring, cats[i]))
        cff.write('     )\n')
        cff.write('\n')


        # working points
        for wp in cfg_wp:
            ntp_eleMVAs_entries.append(wp)
            cff.write('{0}_container = EleMVARaw_WP(\n'.format(wp.replace("-","_")))
            cff.write('    idName = "{0}", mvaTag = mvaTag,\n'.format(wp))
            for i in range(n_cat):
                cat = cfg_wp[wp]["categories"][i]
                wpi = wp_dict[idname][wp][cat]
                if isinstance(wpi, dict):
                    cff.write('    cutCategory{0} = "{1} - exp(-pt / {2}) * {3}", # {4}\n'.format(i, wpi['c'], wpi['tau'], wpi['A'], cat))
                else:
                    cff.write('    cutCategory{0} = "{1}", # {2}\n'.format(i, wpi, cat))
            cff.write('    )\n')
            cff.write('\n')
        cff.write('\n')

        # Set up mva value producer
        cff.write('{} = cms.PSet(\n'.format(cfg_cmssw["produceer_config_name"]))
        cff.write('    mvaName             = cms.string(mvaClassName),\n'.format("mvaClassName"))
        cff.write('    mvaTag              = cms.string(mvaTag),\n')
        cff.write('    nCategories         = cms.int32({}),\n'.format(n_cat))
        cff.write('    categoryCuts        = categoryCuts,\n')
        cff.write('    weightFileNames     = mvaWeightFiles,\n')
        cff.write('    variableDefinition  = cms.string(mvaVariablesFile)\n')
        cff.write('    )\n')
        cff.write('\n')

        # set up the working points
        for wp in cfg_wp:
            cff.write('{0} = configureVIDMVAEleID( {0}_container )\n'.format(wp.replace("-","_")))
        cff.write('\n')

        # approve
        for wp in cfg_wp:
            cff.write('{0}.isPOGApproved = cms.untracked.bool(False)\n'.format(wp.replace("-","_")))

# Modify ntuplizer
with open(cfg['ntuplizer_cfg'], 'r') as ntp:
  lines = ntp.read().split('\n')
  with open(ntp_new_file, 'w') as ntp_new:
      for i, l in enumerate(lines):
          if i == len(lines) - 1:
              ntp_new.write(l)
          else:
              ntp_new.write(l+'\n')
          if "my_id_modules" in l and "=" in l:
              for x in ntp_cff_entries:
                  ntp_new.write("        'RecoEgamma.ElectronIdentification.Identification.{}',".format(x.replace(".py", ""))+'\n')
          if "eleMVAs" in l:
              for x in ntp_eleMVAs_entries:
                  ntp_new.write('                                          "egmGsfElectronIDs:{}",'.format(x)+'\n')
          if "eleMVALabels" in l:
              for x in ntp_eleMVAs_entries:
                  ntp_new.write('                                          "{}",'.format("".join(x.split("-")[1:]))+'\n')
          if "eleMVAValMaps" in l:
              for x in ntp_eleMVAValMaps_entries:
                  ntp_new.write('                                           "electronMVAValueMapProducer:ElectronMVAEstimatorRun2{}",'.format(x)+'\n')
          if "eleMVAValMapLabels" in l:
              for x in ntp_eleMVAValMaps_entries:
                  ntp_new.write('                                           "{}",'.format(x.replace("Values", "Vals"))+'\n')
