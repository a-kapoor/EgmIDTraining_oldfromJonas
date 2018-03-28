from CRABClient.UserUtilities import config, getUsernameFromSiteDB
from config import cfg
import sys

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
    config.Data.outLFNDirBase = '%s/%s/' % (mainOutputDir,'test')
    config.Data.splitting     = 'FileBased'
    config.Data.unitsPerJob   = 8

    config.Data.inputDataset    = cfg["test_sample"]
    config.General.requestName  = cfg["test_sample_request_name"]
    submit(config)
