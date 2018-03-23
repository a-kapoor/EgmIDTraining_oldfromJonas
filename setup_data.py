import os
from config import cfg
from ROOT import TFile
from root_numpy import tree2array
import numpy as np
import xgboost as xgb
import json

ntuple_dir = cfg['ntuple_dir'] + '/' + cfg['submit_version']
ntuple_file = ntuple_dir + '/train_eval.root'

input_file = TFile.Open(ntuple_file, "read")
input_dir = input_file.Get("ntuplizer")
tree_name = "tree"
tree = input_dir.Get(tree_name)

dmatrix_dir = cfg['dmatrix_dir'] + '/' + cfg['submit_version']
out_dir = cfg["out_dir"] + '/' + cfg['submit_version']

num_ele = {}

if not os.path.exists(dmatrix_dir):
    os.makedirs(dmatrix_dir)

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

if not os.path.exists(out_dir + "/true"):
    os.makedirs(out_dir + "/true")

for name in cfg["trainings"]:

    if not os.path.exists(out_dir + "/" + name):
        os.makedirs(out_dir + "/" + name)

    print("Processing data for training bin {}...".format(name))

    num_ele[name] = {}

    cut = cfg["trainings"][name]["cut"]
    variables = cfg["trainings"][name]["variables"]

    sel = cut + " && " + cfg["selection_base"] + " && (" + cfg["selection_sig"] + " || " + cfg["selection_bkg"] + ")"
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

    num_ele[name]["sig"] = n_sig
    num_ele[name]["bkg"] = n_bkg

    # index where to split between training and eval
    s = int(n * cfg["train_size"])

    np.save(out_dir + '/' + name + '/y_train.npy'.format(name), y[:s])
    np.save(out_dir + '/' + name + '/y_eval.npy'.format(name), y[s:])

    np.save(out_dir + '/' + name + '/eta_train.npy'.format(name), eta[:s])
    np.save(out_dir + '/' + name + '/eta_eval.npy'.format(name), eta[s:])
    np.save(out_dir + '/' + name + '/pt_train.npy'.format(name), pt[:s])
    np.save(out_dir + '/' + name + '/pt_eval.npy'.format(name), pt[s:])

    dtrain = xgb.DMatrix(X[:s], label=y[:s])
    deval  = xgb.DMatrix(X[s:], label=y[s:])

    dtrain.save_binary(dmatrix_dir + "/" + name + "_train.DMatrix")
    deval.save_binary(dmatrix_dir + "/" + name + "_eval.DMatrix")

with open(out_dir + '/num_ele.json', 'w') as fp:
    json.dump(num_ele, fp, indent=4)
