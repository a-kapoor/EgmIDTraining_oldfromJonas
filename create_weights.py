#!/usr/bin/env python3

import ROOT
import numpy as np
import matplotlib.pyplot as plt
from root_numpy import tree2array, array2root
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--bkg_file', help='file to draw the background distribution from', required=True)
parser.add_argument('--sig_file', help='file where to weights are adapted too', required=True)
parser.add_argument('--pt_name', help='pt branch name', default="ele_pt")
parser.add_argument('--eta_name', help='eta branch name', default="scl_eta")
parser.add_argument('--bkg_sel', help='background selection', default="(matchedToGenEle == 0 || matchedToGenEle == 3) && vtxN > 1")
parser.add_argument('--sig_sel', help='signal selection', default="matchedToGenEle == 1 && vtxN > 1")
parser.add_argument('--weight_name', help='name of weight branch', default="weights")
parser.add_argument('--out_dir', help='output directory', default="weights")

# parser.print_help()

args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

# yedges = np.array([5,6,7,8,9] + list(range(10, 51))[::1] + [55.0,60.0,65.0,70.0,75.0,80.0,90.0,100.0,150.0,200.0,250.0])
# xedges = np.linspace(-2.5, 2.5, 100)
yedges = np.array(list(range(20, 51))[::1] + [55.0,60.0,65.0,70.0,75.0,80.0,90.0,100.0,150.0,200.0,250.0])
xedges = np.linspace(-2.5, 2.5, 50)
pt_min, pt_max = min(yedges), max(yedges)
eta_min, eta_max = min(xedges), max(xedges)

root_file_bkg = ROOT.TFile(args.bkg_file, 'READ')
root_dir_bkg = root_file_bkg.Get("ntuplizer")
root_tree_bkg = root_dir_bkg.Get("tree")

root_file_sig = ROOT.TFile(args.sig_file, 'READ')
root_dir_sig = root_file_sig.Get("ntuplizer")
root_tree_sig = root_dir_sig.Get("tree")

####
arr = tree2array(root_tree_sig, branches=["matchedToGenEle", "vtxN"]) # HARDCODED!
is_sig = np.logical_and(arr["matchedToGenEle"] == 1, arr["vtxN"] > 1) # HARDCODED!
####

sig = tree2array(root_tree_sig, selection=args.sig_sel, branches=[args.pt_name, args.eta_name])
bkg = tree2array(root_tree_bkg, selection=args.bkg_sel, branches=[args.pt_name, args.eta_name])
n     = len(arr)
n_sig = len(sig)
n_bkg = len(bkg)

H_sig, xedges, yedges = np.histogram2d(sig[args.eta_name], sig[args.pt_name], bins=(xedges, yedges))
H_bkg, xedges, yedges = np.histogram2d(bkg[args.eta_name], bkg[args.pt_name], bins=(xedges, yedges))

W = H_bkg / H_sig

W[W == np.inf] = 0
W = np.nan_to_num(W)

bkg_in_bin = np.logical_and.reduce([bkg[args.pt_name] > pt_min, bkg[args.pt_name] < pt_max,
                             bkg[args.eta_name] > eta_min, bkg[args.eta_name] < eta_max])
bkg_in_bins = bkg[bkg_in_bin]

in_bin = np.logical_and.reduce([sig[args.pt_name] > pt_min, sig[args.pt_name] < pt_max,
                             sig[args.eta_name] > eta_min, sig[args.eta_name] < eta_max])
sig_in_bins = sig[in_bin]
xinds = np.digitize(np.abs(sig_in_bins[args.eta_name]), xedges) - 1
yinds = np.digitize(sig_in_bins[args.pt_name], yedges) - 1

n_sig_in_bins = len(sig_in_bins)
n_sig_overflow = n_sig - n_sig_in_bins

weights = np.ones(n)
sig_weights = np.zeros(n_sig)
sig_weights[in_bin] = W[xinds, yinds]

weights[is_sig] = sig_weights

out_file = args.sig_file.replace(".root", "").split("/")[-1] + "_weights_from_" + args.bkg_file.replace(".root", "").split("/")[-1] + ".root"

x = np.array([(x) for x in weights], dtype=[(args.weight_name, 'f4')])
array2root(x, os.path.join(args.out_dir, out_file), treename="tree", mode='recreate')

############################
# Make some validation plots
############################

density=True

for var_name, edges in zip([args.pt_name, args.eta_name], [yedges, xedges]):

    bin_centers = edges[:-1] + np.diff(edges)/2.

    hist, _, _ = plt.hist(sig_in_bins[var_name], edges, histtype='step', label="signal")
    hist_reweighted_density, _, _ = plt.hist(sig_in_bins[var_name], edges, density=density, histtype='step', weights=weights[is_sig][in_bin])

    plt.figure()
    plt.hist(bkg_in_bins[var_name], edges, density=density, histtype='step', label="background")
    hist_density, _, _ = plt.hist(sig_in_bins[var_name], edges, density=density, histtype='step', label="signal")

    hist_err = np.sqrt(hist)
    hist_reweighted_density_err = hist_err * hist_reweighted_density / hist

    plt.errorbar(bin_centers, hist_reweighted_density, yerr=hist_reweighted_density_err, fmt='o', color='k', label="signal reweighted", markersize="3")
    plt.xlabel(var_name)
    plt.legend()
    fig_name = os.path.join(args.out_dir, "{0}_{1}".format(args.weight_name, var_name))
    plt.savefig(fig_name + ".png")
    plt.savefig(fig_name + ".pdf")
    plt.close()
