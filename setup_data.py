import os
from config import cfg
from ROOT import TFile
from root_numpy import tree2array
import numpy as np
import xgboost as xgb
import json
from os.path import join

ntuple_dir = join(cfg['ntuple_dir'], cfg['submit_version'])
ntuple_file = join(ntuple_dir, 'train_eval.root')

input_file = TFile.Open(ntuple_file, "read")
input_dir = input_file.Get("ntuplizer")
tree_name = "tree"
tree = input_dir.Get(tree_name)

dmatrix_dir = join(cfg['dmatrix_dir'], cfg['submit_version'])
out_dir = join(cfg["out_dir"], cfg['submit_version'])

num_ele = {}

if not os.path.exists(dmatrix_dir):
    os.makedirs(dmatrix_dir)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for idname in cfg["trainings"]:

    num_ele[idname] = {}

    for training_bin in cfg["trainings"][idname]:

        out_dir_full = join(out_dir, idname, training_bin)

        if not os.path.exists(out_dir_full):
            os.makedirs(out_dir_full)

        print("Processing data for {0} training bin {1}...".format(idname, training_bin))

        num_ele[idname][training_bin] = {}

        cut = cfg["trainings"][idname][training_bin]["cut"]
        variables = cfg["trainings"][idname][training_bin]["variables"]

        sel = cut + " && " + cfg["selection_base"]
        data = tree2array(tree, selection=sel, branches=set(variables + ["matchedToGenEle", "ele_pt", "scl_eta"]))

        n = len(data)

        X = np.zeros([n, len(variables)], dtype=np.float)
        for i, v in enumerate(variables):
            X[:,i] = data[v]

        y = data["matchedToGenEle"] == 1

        pt = data["ele_pt"]
        eta = data["scl_eta"]

        n_sig = np.sum(y)
        n_bkg = len(y) - n_sig

        print("True electrons : {}".format(n_sig))
        print("Fakes          : {}".format(n_bkg))

        num_ele[idname][training_bin]["sig"] = n_sig
        num_ele[idname][training_bin]["bkg"] = n_bkg

        # index where to split between training and eval
        s = int(n * cfg["train_size"])

        np.save(join(out_dir_full, 'y_train.npy'), y[:s])
        np.save(join(out_dir_full, 'y_eval.npy'), y[s:])

        np.save(join(out_dir_full, 'eta_train.npy'), eta[:s])
        np.save(join(out_dir_full, 'ta_eval.npy'), eta[s:])
        np.save(join(out_dir_full, 'pt_train.npy'), pt[:s])
        np.save(join(out_dir_full, 'pt_eval.npy'), pt[s:])

        dtrain = xgb.DMatrix(X[:s], label=y[:s])
        deval  = xgb.DMatrix(X[s:], label=y[s:])

        dtrain.save_binary(join(dmatrix_dir, idname + "_" + training_bin + "_train.DMatrix"))
        deval.save_binary(join(dmatrix_dir, idname + "_" + training_bin + "_eval.DMatrix"))

    with open(out_dir + '/num_ele.json', 'w') as fp:
        json.dump(num_ele, fp, indent=4)
