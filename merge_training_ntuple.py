from config import cfg
import os

out_dir = cfg['ntuple_dir'] + '/' + cfg['submit_version']
out_file = out_dir + '/train_eval.root'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Reconstruct the directory where crab stored the ntuples
crab_dir = cfg["crab_output_dir_full"] + "/train_eval/" + \
           cfg["train_eval_sample"].split('/')[1] + "/crab_" + \
           cfg["train_eval_sample_request_name"]

crab_job_paths = os.popen("xrdfs polgrid4.in2p3.fr ls -u {}".format(crab_dir)).read()

# Get the date from the most recent crab job
crab_dir = crab_dir + "/" + crab_job_paths.split("/")[-1].strip() + "/0000"

file_list = os.popen("xrdfs polgrid4.in2p3.fr ls -u {}".format(crab_dir)).read().split("\n")
file_list = [x for x in file_list if '.root' in x]

os.system("source root-v6-06-00-el6-gcc48-thisroot.sh && hadd " + out_file + " " + " ".join(file_list))
