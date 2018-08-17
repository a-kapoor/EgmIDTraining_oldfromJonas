import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("out/20180813_EleMVATraining/Fall17NoIsoV2/EE_5/xgb_bo_results.csv")
df["rel_improv"] = (1 - df["test-auc-mean"][0])/(1 - df["test-auc-mean"])
df["rel_improv_err"] = (df["test-auc-std"]**2 + df["test-auc-std"][0]**2)**0.5

x = df.index.values
y = df["rel_improv"].values
y_err = [0]*len(y)#df["rel_improv_err"].values

plt.errorbar(x[6:] , y[6:],  yerr=y_err[6:], fmt='o-', label='bayes_opt')
plt.errorbar(x[1:6], y[1:6], yerr=y_err[1:6], fmt='o-', label='init')
plt.errorbar(x[0]  , y[0],   yerr=y_err[0], fmt='o-', label='default')
plt.legend(loc="lower right")
plt.xlim(-2, 60)
plt.xlabel("n_iter")
plt.ylabel("(1-AUC)/(1-AUC_default)")
plt.title("Electron ID in EB2_5")
plt.show()
# plt.savefig("bayes_opt.png")
