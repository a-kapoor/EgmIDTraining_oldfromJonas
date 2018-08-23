from sklearn import metrics
import pandas as pd
from config import cfg
import matplotlib.pyplot as plt
from utils import ROCPlot
import uproot
import numpy as np
from sklearn import linear_model

for location in ["EB1_10", "EB1_5", "EB2_5", "EB2_10", "EE_10", "EE_5"]:
    # test_v2 = uproot.open('/home/llr/cms/rembser/data/Egamma/20180323_EleMVATraining/test.root')
    # test_v2 = uproot.open('/home/llr/cms/rembser/data/Egamma/20180323_EleMVATraining/train_eval.root')
    test_v2 = uproot.open('/home/llr/cms/rembser/data_home/Egamma/20180813_EleMVATraining/train_eval.root')
    df_v2 = test_v2["ntuplizer/tree"].pandas.df(["Fall17IsoV2Vals", "ele_pt", "scl_eta", "matchedToGenEle", "genNpu"], entrystop=None)

    df = pd.read_hdf("/home/llr/cms/rembser/EgmIDTraining/out/20180813_EleMVATraining/Fall17IsoV2/{}/pt_eta_score.h5".format(location))

    df_v2 = df_v2.query(cfg["selection_base"])
    df_v2 = df_v2.query(cfg["trainings"]["Fall17IsoV2"][location]["cut"])
    df_v2.eval('y = ({0}) + 2 * ({1}) - 1'.format(cfg["selection_bkg"], cfg["selection_sig"]), inplace=True)
    df_v2 = df_v2.query("y >= 0")

    # So we don't get what was used for V2 training
    # df_v2 = df_v2[int(len(df_v2*0.75)):]


    df_tmva = pd.read_hdf("/home/llr/cms/rembser/EgmIDTraining/out/20180813_EleMVATraining/Fall17IsoV2/{}/legacy/pt_eta_score.h5".format(location))
    # df_tmva_noiso = pd.read_hdf("/home/llr/cms/rembser/EgmIDTraining/out/20180813_EleMVATraining/Fall17NoIsoV2/{}/legacy/pt_eta_score.h5".format(location))

    # ea = pd.read_csv("effAreaElectrons_cone03_pfNeuHadronsAndPhotons_94X.txt", comment="#", delim_whitespace=True, header=None, names=["eta_min", "eta_max", "ea"])
    # df_tmva_noiso["ea"] = 0.0
    # for i in range(len(ea))[::-1]:
        # df_tmva_noiso.at[abs(df_tmva_noiso["scl_eta"]) < ea.iloc[i]["eta_max"], "ea"] = ea.iloc[i]["ea"]

    # def add_hzz_iso(df):
        # df["hzz_iso"] = (df["ele_pfChargedHadIso"] + np.clip(df["ele_pfNeutralHadIso"] + df["ele_pfPhotonIso"] - df["rho"]*df["ea"], 0, None)) / df["ele_pt"]
        # return df

    # df_tmva_noiso = add_hzz_iso(df_tmva_noiso)
    # df_tmva_noiso["hzz_iso"].hist(bins=200)
    # plt.show()

    # regr = linear_model.LogisticRegression()
    # regr.fit(df_tmva_noiso[["hzz_iso", "BDT"]], df_tmva_noiso["classID"])
    # print('Coefficients: \n', regr.coef_)
    # df_tmva_noiso["hzz_seq"] = regr.predict_proba(df_tmva_noiso[["hzz_iso", "BDT"]])[:,1]
    # print(df_tmva_noiso)

    plt.figure()
    roc = ROCPlot(xlim=(0.6,1), ylim=(0.0011, 1), logscale=True, grid=True, percent=True, height_ratios=[1,1], ncol=2, rlim=(0.75, 1.15))
    # roc.plot(df_tmva_noiso["classID"] == 1, df_tmva_noiso["hzz_seq"], label="TMVA + iso seq.", color='k')
    roc.plot(df_tmva["classID"] == 0, df_tmva["BDT"], label="TMVA")
    roc.plot(df_v2["y"] == 1, df_v2["Fall17IsoV2Vals"], label="Fall17V2")
    roc.plot(df["matchedToGenEle"] == 1, df["bdt_score_default"], label="xgb default")
    roc.plot(df["matchedToGenEle"] == 1, df["bdt_score_bo"], label="xgb bayes_opt")
    # plt.show()
    plt.savefig("plots/bayes_opt/roc_{}.pdf".format(location))
    plt.savefig("plots/bayes_opt/roc_{}.png".format(location), dpi=300)
