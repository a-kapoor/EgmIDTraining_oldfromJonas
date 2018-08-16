from config import cfg
import json
from os.path import join
import os

plot_dir = join("plots", cfg['submit_version'])

substitutes = {
        "PLOT_ROC_EB1_10": "roc/2017_EB1_10.pdf",
        "PLOT_ROC_EB2_10": "roc/2017_EB2_10.pdf",
        "PLOT_ROC_EE_10": "roc/2017_EE_10.pdf",
        "PLOT_ROC_EB1_5": "roc/2017_EB1_5.pdf",
        "PLOT_ROC_EB2_5": "roc/2017_EB2_5.pdf",
        "PLOT_ROC_EE_5": "roc/2017_EE_5.pdf",
        "PLOT_TURNON_EB1": "turnon/2017_EB1_wpAll.pdf",
        "PLOT_TURNON_EB2": "turnon/2017_EB2_wpAll.pdf",
        "PLOT_TURNON_EE": "turnon/2017_EE_wpAll.pdf",
        "PLOT_ETAEFF_5": "etaeff/5_wpAll.pdf",
        "PLOT_ETAEFF_10": "etaeff/10_wpAll.pdf",
        "PLOT_NVTX_EB1_10": "nvtx/2017_EB1_10_wpAll.pdf",
        "PLOT_NVTX_EB2_10": "nvtx/2017_EB2_10_wpAll.pdf",
        "PLOT_NVTX_EE_10": "nvtx/2017_EE_10_wpAll.pdf",
        "PLOT_NVTX_EB1_5": "nvtx/2017_EB1_5_wpAll.pdf",
        "PLOT_NVTX_EB2_5": "nvtx/2017_EB2_5_wpAll.pdf",
        "PLOT_NVTX_EE_5": "nvtx/2017_EE_5_wpAll.pdf",
        }

for key in substitutes:
    substitutes[key] = join(plot_dir, substitutes[key])

if not os.path.exists("slides"):
    os.makedirs("slides")

with open("slides_template.tex", 'r') as template:
    lines = template.readlines()
    with open(join("slides", cfg['submit_version']+".tex"), 'w') as slides:

        for l in lines:
            for s in substitutes:
                l = l.replace(s, substitutes[s])
            slides.write(l)
