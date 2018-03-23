import xgboost as xgb
from xgboost2tmva import convert_model
from config import cfg
import json
import os
import numpy as np

dmatrix_dir = cfg['dmatrix_dir'] + '/' + cfg['submit_version']
out_dir = cfg["out_dir"] + '/' + cfg['submit_version']

with open(out_dir + '/num_ele.json', 'r') as f:
    num_ele = json.load(f)

for name in cfg["trainings"]:

    if not os.path.exists(out_dir + "/" + name):
        os.makedirs(out_dir + "/" + name)

    dtrain = xgb.DMatrix(dmatrix_dir + "/" + name + "_train.DMatrix")
    deval  = xgb.DMatrix(dmatrix_dir + "/" + name + "_eval.DMatrix")

    params = cfg["trainings"][name]["params"]
    variables = cfg["trainings"][name]["variables"]

    if 'balance_sample' in params:
        if params['balance_sample']:
            params['scale_pos_weight'] = 1. * num_ele[name]["bkg"] / num_ele[name]["sig"]

    params['silent'] = 0
    params['objective'] = 'binary:logistic'
    params['nthread'] = 8
    params['eval_metric'] = 'auc'

    evallist = [(deval, 'eval'), (dtrain, 'train')]

    num_round = 1000
    eval_dict = {}

    bst = xgb.train(params, dtrain, num_round, evallist, early_stopping_rounds=10, evals_result=eval_dict)

    # print("Saving xgboost model...")
    # bst.save_model(out_dir + "/" + name + "/xgb.model")

    print("Saving TMVA model...")
    model = bst.get_dump()
    variables_with_type = list(zip(variables, len(variables)*['F']))
    tmvafile = out_dir + "/" + name + "/weights.xml"
    convert_model(model,input_variables=variables_with_type,output_xml=tmvafile)
    os.system("xmllint --format {0} > {0}.tmp".format(tmvafile))
    os.system("mv {0}.tmp {0}".format(tmvafile))
    os.system("cd "+ out_dir + "/" + name + " && gzip weights.xml")

    # Save the auc during all the rounds
    eval_arr = np.array([eval_dict[u'train'][u'auc'], eval_dict[u'eval'][u'auc']]).T
    np.savetxt(out_dir + "/" + name + "/rounds.txt", eval_arr, fmt='%f %f',
            header='train_auc eval_auc')

    # Saving predictions
    label_eval = deval.get_label()
    label_train = dtrain.get_label()
    y_pred_eval = bst.predict(deval, ntree_limit=bst.best_ntree_limit)
    y_pred_train = bst.predict(dtrain, ntree_limit=bst.best_ntree_limit)

    np.save(out_dir + '/' + name + '/y_bdt_train.npy'.format(name), y_pred_train)
    np.save(out_dir + '/' + name + '/y_bdt_eval.npy'.format(name), y_pred_eval)
