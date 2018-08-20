import sys
import os
from ROOT import TMVA, TFile, TCut
from config import cfg, variables_iso_only
from os.path import join
import uproot
import root_pandas

TMVA.Tools.Instance()
(TMVA.gConfig().GetVariablePlotting()).fNbinsXOfROCCurve = 400

out_dir_base = join(cfg["out_dir"], cfg['submit_version'])

max_n_per_class = 200000

for idname in cfg["trainings"]:

    for training_bin in cfg["trainings"][idname]:

        print("Process training pipeline for {0} {1}".format(idname, training_bin))

        out_dir = join(out_dir_base, idname, training_bin, "legacy")

        if not os.path.exists(out_dir):
            os.makedirs(join(out_dir))

        feature_cols = cfg["trainings"][idname][training_bin]["variables"]

        outfileName = join(out_dir, "TMVA.root")
        print("---> Working with OutfileName = " + outfileName);

        outputFile = TFile.Open(outfileName, "RECREATE")

        mva_name = "EIDmva_"+idname+"_"+training_bin
        factory = TMVA.Factory(mva_name, outputFile, ":".join([
                               "!V", "!Silent", "Color", "DrawProgressBar=False",
                               "Transformations=I;D;P;G", "AnalysisType=Classification"
                               ]))

        for variable in feature_cols:
            factory.AddVariable(variable, 'F')

        factory.AddSpectator("ele_pt", 'F')
        factory.AddSpectator("scl_eta", 'F')
        factory.AddSpectator("genNpu", 'F')

        # Add isolation variables as spectators
        spectators = []
        for v in set(variables_iso_only).difference(set(feature_cols)):
            spectators.append(v)
            factory.AddSpectator(v, 'F')

        # -----------------------------
        #  Input File & Tree
        # -----------------------------
        ntuple_dir = join(cfg['ntuple_dir'], cfg['submit_version'])
        ntuple_file = join(ntuple_dir, 'train_eval.root')
        root_file = uproot.open(ntuple_file)
        tree = root_file["ntuplizer/tree"]

        df = tree.pandas.df(feature_cols + spectators + ["ele_pt", "scl_eta", "matchedToGenEle", "genNpu"], entrystop=None)

        df = df.query(cfg["selection_base"])
        df = df.query(cfg["trainings"][idname][training_bin]["cut"])

        df_sig = df.query(cfg["selection_sig"])
        df_bkg = df.query(cfg["selection_bkg"])

        df_sig.to_root(join(out_dir, 'sig.root'), key='tree')
        df_bkg.to_root(join(out_dir, 'bkg.root'), key='tree')

        sig_file = TFile.Open(join(out_dir, 'sig.root'), "read")
        bkg_file = TFile.Open(join(out_dir, 'bkg.root'), "read")

        sig_tree = sig_file.Get("tree")
        bkg_tree = bkg_file.Get("tree")

        # You can add an arbitrary number of signal or background trees
        factory.AddSignalTree(sig_tree)
        factory.AddBackgroundTree(bkg_tree)

        # ---------------------------
        #  Training
        # ---------------------------

        # Get number of events
        n_sig = sig_tree.GetEntries()
        n_bkg = bkg_tree.GetEntries()
        n_sig_train = min(int(n_sig * cfg["train_size"]), max_n_per_class)
        n_bkg_train = min(int(n_bkg * cfg["train_size"]), max_n_per_class)
        n_sig_test  = n_sig - n_sig_train
        n_bkg_test  = n_bkg - n_bkg_train

        prepare_nevents = ":".join([
                                    "nTrain_Signal=" + str(n_sig_train),
                                    "nTrain_Background=" + str(n_bkg_train),
                                    "nTest_Signal=" + str(n_sig_test),
                                    "nTest_Background=" + str(n_bkg_test),
                                    "SplitMode=Random",
                                    "NormMode=NumEvents",
                                    "!V"
                                   ])
        print(prepare_nevents)

        factory.PrepareTrainingAndTestTree(TCut(""), TCut(""), prepare_nevents)

        # The settings from the Sprng16 Electron MVA
        # see https://github.com/cms-data/RecoEgamma-ElectronIdentification
        factory.BookMethod(TMVA.Types.kBDT, "BDT", ":".join([
                                             "!H",
                                             "!V",
                                             "NTrees=1000", #"NTrees=2000",
                                             "BoostType=Grad",
                                             "Shrinkage=0.10",
                                             "!UseBaggedGrad",
                                             "nCuts=2000",
                                             #"nEventsMin=100",
                                             #"NNodesMax=5",
                                             #"UseNvars=4",
                                             "MinNodeSize=1.5%", #"MinNodeSize=0.1%",
                                             "PruneStrength=5",
                                             "PruneMethod=CostComplexity",
                                             "MaxDepth=5", #"MaxDepth=6",
                                             "CreateMVAPdfs",
                                             #"NegWeightTreatment=PairNegWeightsGlobal",
                                                            ]))

        factory.TrainAllMethods()
        factory.TestAllMethods()
        factory.EvaluateAllMethods()

        # Save the output
        outputFile.Close()

        print("==> Wrote root file: " + outputFile.GetName())
        print("==> TMVAClassification is done!")

        # move weight files
        os.system("mv weights/"+mva_name+"_BDT.weights.xml "+join(out_dir, "BDT.weights.xml"))
        os.system("cd "+out_dir+" && gzip "+join(out_dir, "BDT.weights.xml"))
        os.system("mv weights/"+mva_name+"_BDT.class.C "+join(out_dir, "BDT.class.C"))
        os.system("rmdir weights")

        # Convert train tree to pandas data frame saved in hdf
        root_file = uproot.open(join(out_dir, "TMVA.root"))
        tree = root_file["TestTree"]
        df = tree.pandas.df(["ele_pt", "scl_eta", "genNpu", "classID", "BDT"] + list(variables_iso_only), entrystop=None)
        df.to_hdf(join(out_dir,'pt_eta_score.h5'), key="TestTree")
