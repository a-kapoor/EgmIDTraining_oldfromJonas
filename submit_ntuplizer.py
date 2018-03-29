from CRABClient.UserUtilities import config, getUsernameFromSiteDB
from config import cfg
import sys
import argparse

parser = argparse.ArgumentParser(description='Submit crab jobs.')
parser.add_argument('--train', action='store_true', default=False, help='make ntuplizer for training.')
parser.add_argument('--test', action='store_true', default=False, help='make ntuplizer for testing.')

args = parser.parse_args()

config = config()

submitVersion = cfg["submit_version"]
mainOutputDir = cfg["crab_output_dir"]

config.General.transferLogs = False

config.JobType.pluginName = 'Analysis'
config.JobType.psetName = cfg['ntuplizer_cfg']
config.JobType.sendExternalFolder = True

config.Data.inputDBS = 'global'
config.Data.publication = False

config.Site.storageSite = cfg["storage_site"]
if __name__ == '__main__':

    from CRABAPI.RawCommand import crabCommand
    from CRABClient.ClientExceptions import ClientException
    from httplib import HTTPException

    # We want to put all the CRAB project directories from the tasks we submit here into one common directory.
    # That's why we need to set this parameter (here or above in the configuration file, it does not matter, we will not overwrite it).
    config.General.workArea = 'crab_%s' % submitVersion

    def submit(config):
        try:
            crabCommand('submit', config = config)
        except HTTPException as hte:
            print "Failed submitting task: %s" % (hte.headers)
        except ClientException as cle:
            print "Failed submitting task: %s" % (cle)

    ##### submit MC
    config.Data.splitting     = 'FileBased'
    config.Data.unitsPerJob   = 8

    if args.train:
        config.Data.outLFNDirBase = '%s/%s/' % (mainOutputDir,'train_eval')
        config.Data.inputDataset    = cfg["train_eval_sample"]
        config.General.requestName  = cfg["train_eval_sample_request_name"]
        submit(config)

    if args.test:
        config.Data.outLFNDirBase = '%s/%s/' % (mainOutputDir,'test_eval')
        config.Data.inputDataset    = cfg["test_sample"]
        config.General.requestName  = cfg["test_sample_request_name"]
        submit(config)
