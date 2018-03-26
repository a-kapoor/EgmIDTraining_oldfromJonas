import xgboost as xgb
from xgboost2tmva import convert_model
from config import cfg
import json
import os
import numpy as np
from os.path import join

dmatrix_dir = join(cfg['dmatrix_dir'], cfg['submit_version'])
out_dir_base = join(cfg["out_dir"], cfg['submit_version'])

with open(join(out_dir_base, 'num_ele.json'), 'r') as f:
    num_ele = json.load(f)

for idname in cfg["trainings"]:

    for training_bin in cfg["trainings"][idname]:

        out_dir = join(out_dir_base, idname, training_bin)

        dtrain = xgb.DMatrix(join(dmatrix_dir, idname + "_" + training_bin + "_train.DMatrix"))
        deval  = xgb.DMatrix(join(dmatrix_dir, idname + "_" + training_bin + "_train.DMatrix"))

        params = cfg["trainings"][idname][training_bin]["params"]
        variables = cfg["trainings"][idname][training_bin]["variables"]

        if 'balance_sample' in params:
            if params['balance_sample']:
                params['scale_pos_weight'] = 1. * num_ele[idname][training_bin]["bkg"] / num_ele[idname][training_bin]["sig"]

        params['silent'] = 0
        # params['objective'] = 'binary:logistic'
        params['objective'] = 'binary:logitraw'
        params['nthread'] = 8
        params['eval_metric'] = 'auc'

        evallist = [(deval, 'eval'), (dtrain, 'train')]

        num_round = 1000
        eval_dict = {}

        bst = xgb.train(params, dtrain, num_round, evallist, early_stopping_rounds=10, evals_result=eval_dict)

        # print("Saving xgboost model...")
        # bst.save_model(out_dir + "/xgb.model")

        print("Saving TMVA model for {0} {1}...".format(idname, training_bin))
        model = bst.get_dump()
        variables_with_type = list(zip(variables, len(variables)*['F']))
        tmvafile = join(out_dir, "weights.xml")
        convert_model(model,input_variables=variables_with_type,output_xml=tmvafile)
        os.system("xmllint --format {0} > {0}.tmp".format(tmvafile))
        os.system("mv {0}.tmp {0}".format(tmvafile))
        os.system("cd "+ out_dir + " && gzip -f weights.xml")

        # Save the auc during all the rounds
        eval_arr = np.array([eval_dict[u'train'][u'auc'], eval_dict[u'eval'][u'auc']]).T
        np.savetxt(join(out_dir, "rounds.txt"), eval_arr, fmt='%f %f',
                header='train_auc eval_auc')

        # Saving predictions
        label_eval = deval.get_label()
        label_train = dtrain.get_label()
        y_pred_raw_eval = bst.predict(deval, ntree_limit=bst.best_ntree_limit)
        y_pred_raw_train = bst.predict(dtrain, ntree_limit=bst.best_ntree_limit)

        y_pred_eval = 2.0/(1.0+np.exp(-2.0*y_pred_raw_eval))-1
        y_pred_train = 2.0/(1.0+np.exp(-2.0*y_pred_raw_train))-1

        np.save(join(out_dir, 'y_bdt_train.npy'), y_pred_train)
        np.save(join(out_dir, 'y_bdt_eval.npy'), y_pred_eval)

        np.save(join(out_dir, 'y_bdt_raw_train.npy'), y_pred_raw_train)
        np.save(join(out_dir, 'y_bdt_raw_eval.npy'), y_pred_raw_eval)
