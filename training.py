from config import cfg
import os
import numpy as np
from os.path import join
import uproot
import pandas as pd

from sklearn.model_selection import train_test_split

from xgbo import XgboClassifier

out_dir_base = join(cfg["out_dir"], cfg['submit_version'])

for idname in cfg["trainings"]:

    for training_bin in cfg["trainings"][idname]:

        print("Process training pipeline for {0} {1}".format(idname, training_bin))

        out_dir = join(out_dir_base, idname, training_bin)

        if not os.path.exists(out_dir):
            os.makedirs(join(out_dir))

        feature_cols = cfg["trainings"][idname][training_bin]["variables"]

        print("Reading data...")
        ntuple_dir = join(cfg['ntuple_dir'], cfg['submit_version'])
        ntuple_file = join(ntuple_dir, 'train_eval.root')
        root_file = uproot.open(ntuple_file)
        tree = root_file["ntuplizer/tree"]

        df = tree.pandas.df(feature_cols + ["ele_pt", "scl_eta", "matchedToGenEle", "genNpu"], entrystop=10000)

        df = df.query(cfg["selection_base"])
        df = df.query(cfg["trainings"][idname][training_bin]["cut"])
        df.eval('y = ({0}) + 2 * ({1}) - 1'.format(cfg["selection_bkg"], cfg["selection_sig"]), inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(df[feature_cols],
                                                            df["y"],
                                                            random_state=99,
                                                            test_size=1. - cfg["train_size"])


        # Entries from the class with more entries are discarded. This is because
        # classifier performance is usually bottlenecked by the size of the
        # dataset for the class with fewer entries. Having one class with extra
        # statistics usually just adds computing time.
        n_per_class = min(y_train.value_counts())

        # The number of entries per class might also be limited by a parameter
        # in case the dataset is just too large for this algorithm to run in a
        # reasonable time.

        selection = np.concatenate([y_train[y_train == 0].head(n_per_class).index.values,
                                    y_train[y_train == 1].head(n_per_class).index.values])

        X_train = X_train.loc[selection]
        y_train = y_train.loc[selection]

        import xgboost as xgb

        xgtrain = xgb.DMatrix(X_train, label=y_train)
        xgtest  = xgb.DMatrix(X_test , label=y_test )

        print("Running bayesian optimized training...")
        xgbo_classifier = XgboClassifier(out_dir=out_dir)

        xgbo_classifier.optimize(xgtrain, init_points=1, n_iter=1, acq='ei')

        xgbo_classifier.fit(xgtrain, model="default")
        xgbo_classifier.fit(xgtrain, model="optimized")

        xgbo_classifier.save_model(feature_cols, model="default")
        xgbo_classifier.save_model(feature_cols, model="optimized")

        preds_default   = xgbo_classifier.predict(xgtest, model="default")
        preds_optimized = xgbo_classifier.predict(xgtest, model="optimized")

        print("Saving reduced data frame...")
        # Create a data frame with bdt outputs and kinematics to calculate the working points
        df_reduced = df.loc[y_test.index, ["ele_pt", "scl_eta", "y", "Fall17NoIsoV2RawVals", "genNpu"]]
        df_reduced["bdt_score_default"]   = preds_default
        df_reduced["bdt_score_optimized"] = preds_optimized
        df_reduced.to_hdf(join(out_dir,'pt_eta_score.h5'), key='pt_eta_score')
