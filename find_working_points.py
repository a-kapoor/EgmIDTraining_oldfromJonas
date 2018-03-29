from config import cfg
import json
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

out_dir_base = join(cfg["out_dir"], cfg['submit_version'])

def wpfunc(x, c, tau, A):
    return c - np.exp(-x / tau) * A

d = {}

for idname in cfg['working_points']:
    d[idname] = {}
    print("Processing {}...".format(idname))

    for wpname in cfg['working_points'][idname]:
        cfgwp = cfg['working_points'][idname][wpname]
        wptype = cfgwp['type']

        eff_boundaries = {}
        wp = {}

        print("    - working point {}...".format(wpname))

        for i, cat in enumerate(cfgwp['categories']):

            out_dir = join(out_dir_base, idname, cat)
            y = np.load(join(out_dir, 'y_eval.npy'))
            y_pred = np.load(join(out_dir, 'y_bdt_raw_eval.npy'))

            if wptype == 'constant_cut_sig_eff_targets':

                wp[cat] = np.percentile(y_pred[y == True], (1-cfgwp['targets'][i]) * 100)

                if 'match_boundary' in cfgwp and cfgwp['match_boundary']:
                    pt = np.load(join(out_dir, 'pt_eval.npy'))
                    if '5' in cat:
                        sel = np.logical_and.reduce([y == True, pt >= 9.5, pt < 10.0])
                    elif '10' in cat:
                        sel = np.logical_and.reduce([y == True, pt >= 10.0, pt < 10.5])
                    eff_boundaries[cat] = np.sum(np.logical_and(sel, y_pred > wp[cat]))*1./np.sum(sel)


            if wptype == 'pt_scaling_cut_sig_eff_targets':
                pt = np.load(join(out_dir, 'pt_eval.npy'))
                x = np.zeros(len(cfgwp['ptbins']))
                x[:] = np.nan
                pt_c = np.mean(cfgwp['ptbins'], axis=1)
                for k, ptbin in enumerate(cfgwp['ptbins']):
                    sel = np.logical_and.reduce([y == True, pt >= ptbin[0], pt < ptbin[1]])
                    if np.sum(sel) == 0:
                        continue
                    x[k] = np.percentile(y_pred[sel], (1-cfgwp['targets'][i][k]) * 100)


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
                    eff_boundaries[cat] = np.sum(np.logical_and(sel, y_pred > wps))*1./np.sum(sel)

        if 'match_boundary' in cfgwp and cfgwp['match_boundary']:
            if wptype == 'constant_cut_sig_eff_targets':
                for i, cat in enumerate(cfgwp['categories']):
                    if '5' in cat:
                        out_dir = join(out_dir_base, idname, cat)
                        y = np.load(join(out_dir, 'y_eval.npy'))
                        y_pred = np.load(join(out_dir, 'y_bdt_raw_eval.npy'))
                        pt = np.load(join(out_dir, 'pt_eval.npy'))
                        sel = np.logical_and.reduce([y == True, pt >= 9.5, pt < 10.0])
                        wp_boundary = np.percentile(y_pred[sel], (1-eff_boundaries[cat.replace('5', '10')]) * 100)
                        wp[cat] = wp_boundary

            if wptype == 'pt_scaling_cut_sig_eff_targets':
                for i, cat in enumerate(cfgwp['categories']):
                    if '5' in cat:
                        out_dir = join(out_dir_base, idname, cat)
                        y = np.load(join(out_dir, 'y_eval.npy'))
                        y_pred = np.load(join(out_dir, 'y_bdt_raw_eval.npy'))
                        pt = np.load(join(out_dir, 'pt_eval.npy'))
                        sel = np.logical_and.reduce([y == True, pt >= 9.5, pt < 10.0])
                        wp_boundary = np.percentile(y_pred[sel], (1-eff_boundaries[cat.replace('5', '10')]) * 100)
                        wp[cat]['c'] = wp[cat]['c'] - wpfunc(9.75, wp[cat]['c'], wp[cat]['tau'], wp[cat]['A']) + wp_boundary

        d[idname][wpname] = wp

with open(join(out_dir_base, 'working_points.json'), 'w') as fp:
    json.dump(d, fp, indent=4)
