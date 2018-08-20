from sklearn import metrics
import pandas as pd
from config import cfg
import matplotlib.pyplot as plt
from utils import ROCPlot

location = "EB2_10"

df = pd.read_hdf("/home/llr/cms/rembser/EgmIDTraining/out/20180813_EleMVATraining/Fall17IsoV2/{}/pt_eta_score.h5".format(location))

df = df.query(cfg["selection_base"])
df = df.query(cfg["trainings"]["Fall17IsoV2"][location]["cut"])
df.eval('y = ({0}) + 2 * ({1}) - 1'.format(cfg["selection_bkg"], cfg["selection_sig"]))

print(df.shape)
print(df.head())

df_tmva = pd.read_hdf("/home/llr/cms/rembser/EgmIDTraining/out/20180813_EleMVATraining/Fall17IsoV2/{}/legacy/pt_eta_score.h5".format(location))

print(df_tmva)


plt.figure()
roc = ROCPlot(xlim=(0.5,1), ylim=(0.0011, 1), logscale=True, grid=True, percent=True)
roc.plot(df_tmva["classID"] == 0, df_tmva["BDT"])
roc.plot(df["matchedToGenEle"] == 1, df["bdt_score_default"])
roc.plot(df["matchedToGenEle"] == 1, df["bdt_score_bo"])
# roc.plot(df["matchedToGenEle"] == 1, df["Fall17IsoV2RawVals"])
plt.show()
