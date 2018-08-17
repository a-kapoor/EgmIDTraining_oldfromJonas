from sklearn import metrics
import pandas as pd
from config import cfg
import matplotlib.pyplot as plt
from utils import ROCPlot

df = pd.read_hdf("/home/llr/cms/rembser/EgmIDTraining/out/20180813_EleMVATraining/Fall17NoIsoV2/EE_5/pt_eta_score.h5")

df = df.query(cfg["selection_base"])
df = df.query(cfg["trainings"]["Fall17NoIsoV2"]["EE_5"]["cut"])
df.eval('y = ({0}) + 2 * ({1}) - 1'.format(cfg["selection_bkg"], cfg["selection_sig"]))

print(df.shape)
print(df.head())

plt.figure()
roc = ROCPlot(xlim=(0.5,1))
# roc.plot(df["matchedToGenEle"] == 1, df["Fall17NoIsoV2RawVals"])
roc.plot(df["matchedToGenEle"] == 1, df["bdt_score_default"])
roc.plot(df["matchedToGenEle"] == 1, df["bdt_score_bo"])
plt.show()
