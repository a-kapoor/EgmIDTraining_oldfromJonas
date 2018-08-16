from sklearn import metrics
import pandas as pd
from config import cfg
import matplotlib.pyplot as plt

df = pd.read_hdf("/home/llr/cms/rembser/EgmIDTraining/out/20180813_EleMVATraining/Fall17NoIsoV2/EE_5/pt_eta_score.h5")

df = df.query(cfg["selection_base"])
df = df.query(cfg["trainings"]["Fall17NoIsoV2"]["EE_5"]["cut"])
df.eval('y = ({0}) + 2 * ({1}) - 1'.format(cfg["selection_bkg"], cfg["selection_sig"]))

print(df.shape)

fpr_ref, tpr_ref, _ = metrics.roc_curve(df["y"], df["Fall17NoIsoV2RawVals"])
fpr_bdt, tpr_bdt, _ = metrics.roc_curve(df["y"], df["bdt_score_default"])
fpr_bdt_bo, tpr_bdt_bo, _ = metrics.roc_curve(df["y"], df["bdt_score_bo"])

print(metrics.roc_auc_score(df["y"], df["Fall17NoIsoV2RawVals"]))
print(metrics.roc_auc_score(df["y"], df["bdt_score_default"]))
print(metrics.roc_auc_score(df["y"], df["bdt_score_bo"]))

plt.figure()
plt.plot(tpr_ref, fpr_ref)
plt.plot(tpr_bdt, fpr_bdt)
plt.plot(tpr_bdt_bo, fpr_bdt_bo)
plt.show()
