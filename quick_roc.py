import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.gridspec as gridspec
from sklearn import metrics
import sys

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

def create_axes(yunits=4):
    #fig = plt.figure(figsize=(6.4, 4.6)) # the default figsize
    fig = plt.figure(figsize=(6.4, 4.8)) # the default figsize
    gs = gridspec.GridSpec(yunits, 1)
    ax1 = plt.subplot(gs[:2, :])
    ax2 = plt.subplot(gs[2:, :])
    axarr = [ax1, ax2]

    gs.update(wspace=0.025, hspace=0.075)

    plt.setp(ax1.get_xticklabels(), visible=False)

    ax1.grid()
    ax2.grid()

    return ax1, ax2, axarr

dirnames = sys.argv[1:]

ax1, ax2, axarr = create_axes(yunits=3)

xmin = 0.7

# Get the reference (first file)
y_eval = np.load(dirnames[0] + "/y_eval.npy")
y_bdt_eval = np.load(dirnames[0] + "/y_bdt_eval.npy")

fpr_ref, tpr_ref, _ = metrics.roc_curve(y_eval, y_bdt_eval, pos_label=1)

for i, dirname in enumerate(dirnames):

    y_eval = np.load(dirname + "/y_eval.npy")
    y_train = np.load(dirname + "/y_train.npy")
    y_bdt_eval = np.load(dirname + "/y_bdt_eval.npy")
    y_bdt_train = np.load(dirname + "/y_bdt_train.npy")

    label = dirname.split("/")[-1]

    rounds = np.loadtxt(dirname + "/rounds.txt")

    auc_train, auc_eval = rounds[-1]

    fpr_train, tpr_train, _ = metrics.roc_curve(y_train, y_bdt_train, pos_label=1)
    fpr_eval, tpr_eval, _ = metrics.roc_curve(y_eval, y_bdt_eval, pos_label=1)

    sel = tpr_eval > xmin
    ax1.semilogy(100 * tpr_eval[sel], 100 * fpr_eval[sel], color=colors[i], label="{0} eval AUC {1:f}".format(label, auc_eval))
    ax2.plot(100 * tpr_eval[sel], fpr_eval[sel] / np.interp(tpr_eval[sel], tpr_ref, fpr_ref), color=colors[i])

    sel = tpr_train > xmin
    ax1.semilogy(100 * tpr_train[sel], 100 * fpr_train[sel], color=colors[i], label="{0} train AUC {1:f}".format(label, auc_train), linestyle='--')
    ax2.plot(100 * tpr_train[sel], fpr_train[sel] / np.interp(tpr_train[sel], tpr_ref, fpr_ref), color=colors[i], linestyle='--')

# Styling the plot

ax1.set_xlabel(r'Signal efficiency [%]')
ax1.set_ylabel(r'Background efficiency [%]')

ax2.set_xlabel(r'Signal efficiency [%]')
ax2.set_ylabel(r'Ratio')

ax1.legend(loc="upper left", ncol=1)

ax1.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

ax1.set_xlim(xmin * 100, 100)
ax2.set_xlim(xmin * 100, 100)

plt.show()
