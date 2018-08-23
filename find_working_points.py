from config import cfg
import json
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

out_dir_base = join(cfg["out_dir"], cfg['submit_version'])

def wpfunc(x, c, tau, A):
    return c - np.exp(-x / tau) * A

d = {}

def load_df(idname, cat, score_col):
    df = pd.read_hdf(join(out_dir_base, idname, cat,'pt_eta_score.h5'))
    # For compatibility with older version of the training framework
    if not "y" in df.columns:
        df.eval('y = ({0}) + 2 * ({1}) - 1'.format(cfg["selection_bkg"], cfg["selection_sig"]), inplace=True)

    df.rename(index=str, columns={score_col: "score"}, inplace=True)

    return df

for idname in cfg['working_points']:

    if idname == "Fall17NoIsoV2":
        continue

    d[idname] = {}
    print("Processing {}...".format(idname))

    for wpname in cfg['working_points'][idname]:
        cfgwp = cfg['working_points'][idname][wpname]
        wptype = cfgwp['type']

        eff_boundaries = {}
        wp = {}

        print("    - working point {}...".format(wpname))

        for i, cat in enumerate(cfgwp['categories']):
            print("        category " + cat)

            df = load_df(idname, cat, "bdt_score_bo")

            y = df["y"].values

            if wptype == 'constant_cut_sig_eff_targets':

                wp[cat] = df.score[df.y == True].quantile(1-cfgwp['targets'][i])

                if 'match_boundary' in cfgwp and cfgwp['match_boundary']:
                    pt = df["ele_pt"].values
                    if '5' in cat:
                        sel = np.logical_and.reduce([y == True, pt >= 9.5, pt < 10.0])
                    elif '10' in cat:
                        sel = np.logical_and.reduce([y == True, pt >= 10.0, pt < 10.5])
                    eff_boundaries[cat] = np.sum(np.logical_and(sel, df.score > wp[cat]))*1./np.sum(sel)


            elif wptype == 'pt_scaling_cut_sig_eff_targets':
                pt = df["ele_pt"].values
                x = np.zeros(len(cfgwp['ptbins']))
                x[:] = np.nan
                pt_c = np.mean(cfgwp['ptbins'], axis=1)
                for k, ptbin in enumerate(cfgwp['ptbins']):
                    sel = np.logical_and.reduce([y == True, pt >= ptbin[0], pt < ptbin[1]])
                    if np.sum(sel) == 0:
                        continue
                    x[k] = np.percentile(df.score[sel], (1-cfgwp['targets'][i][k]) * 100)


                pt_c = pt_c[~np.isnan(x)]
                x = x[~np.isnan(x)]

                popt, pcov = curve_fit(wpfunc, pt_c, x, p0=[7, 20, 10], bounds=([0, 0, 0], [100, 100, 100]))

                wp[cat] = {'c': popt[0], 'tau': popt[1], 'A': popt[2]}

                if 'match_boundary' in cfgwp and cfgwp['match_boundary']:
                    if '5' in cat:
                        sel = np.logical_and.reduce([y == True, pt >= 9.5, pt < 10.0])
                    elif '10' in cat:
                        sel = np.logical_and.reduce([y == True, pt >= 10.0, pt < 10.5])
                    wps = wpfunc(pt, wp[cat]['c'], wp[cat]['tau'], wp[cat]['A'])
                    eff_boundaries[cat] = np.sum(np.logical_and(sel, df.score > wps))*1./np.sum(sel)

        if 'match_boundary' in cfgwp and cfgwp['match_boundary']:
            if wptype == 'constant_cut_sig_eff_targets':
                for i, cat in enumerate(cfgwp['categories']):
                    if '5' in cat:
                        df = load_df(idname, cat, "bdt_score_bo")
                        df.query("y == 1 & ele_pt >= 9.5 & ele_pt < 10.0", inplace=True)
                        wp[cat] = df.score.quantile(1-eff_boundaries[cat.replace('5', '10')])

            if wptype == 'pt_scaling_cut_sig_eff_targets':
                for i, cat in enumerate(cfgwp['categories']):
                    if '5' in cat:
                        df = load_df(idname, cat, "bdt_score_bo")
                        df.query("y == 1 & ele_pt >= 9.5 & ele_pt < 10.0", inplace=True)
                        wp_boundary = df.score.quantile(1-eff_boundaries[cat.replace('5', '10')])
                        wp[cat]['c'] = wp[cat]['c'] - wpfunc(9.75, wp[cat]['c'], wp[cat]['tau'], wp[cat]['A']) + wp_boundary

        d[idname][wpname] = wp

with open(join(out_dir_base, 'working_points.json'), 'w') as fp:
    json.dump(d, fp, indent=4)
