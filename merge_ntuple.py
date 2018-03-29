from config import cfg
import os
import argparse

parser = argparse.ArgumentParser(description='Submit crab jobs.')
parser.add_argument('--train', action='store_true', default=False, help='merge ntuplizer for training.')
parser.add_argument('--test', action='store_true', default=False, help='merge ntuplizer for testing.')

args = parser.parse_args()

out_dir = cfg['ntuple_dir'] + '/' + cfg['submit_version']

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

if args.train:
    # Reconstruct the directory where crab stored the ntuples
    crab_dir = cfg["crab_output_dir_full"] + "/train_eval/" + \
               cfg["train_eval_sample"].split('/')[1] + "/crab_" + \
               cfg["train_eval_sample_request_name"]
    out_file = out_dir + '/train_eval.root'

    crab_job_paths = os.popen("xrdfs polgrid4.in2p3.fr ls -u {}".format(crab_dir)).read()

    # Get the date from the most recent crab job
    crab_dir = crab_dir + "/" + crab_job_paths.split("/")[-1].strip() + "/0000"

    file_list = os.popen("xrdfs polgrid4.in2p3.fr ls -u {}".format(crab_dir)).read().split("\n")
    file_list = [x for x in file_list if '.root' in x]

    os.system("source root-v6-06-00-el6-gcc48-thisroot.sh && hadd " + out_file + " " + " ".join(file_list))

if args.test:
    # Reconstruct the directory where crab stored the ntuples
    crab_dir = cfg["crab_output_dir_full"] + "/test/" + \
               cfg["test_sample"].split('/')[1] + "/crab_" + \
               cfg["test_sample_request_name"]
    out_file = out_dir + '/test.root'

    crab_job_paths = os.popen("xrdfs polgrid4.in2p3.fr ls -u {}".format(crab_dir)).read()

    # Get the date from the most recent crab job
    crab_dir = crab_dir + "/" + crab_job_paths.split("/")[-1].strip() + "/0000"

    file_list = os.popen("xrdfs polgrid4.in2p3.fr ls -u {}".format(crab_dir)).read().split("\n")
    file_list = [x for x in file_list if '.root' in x]

    os.system("source root-v6-06-00-el6-gcc48-thisroot.sh && hadd " + out_file + " " + " ".join(file_list))
