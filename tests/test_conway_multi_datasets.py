import sys
sys.path.append('../conway') # for multi_datasets.py
sys.path.append('../../') # for NDNT and NTDatasets

import torch
import numpy as np
import scipy.io as sio
import multi_datasets as multidata

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

datadir = '/home/dbutts/ColorV1/Data/'
dirname = '/home/dbutts/ColorV1/CLRworkspace/'

class Experiment(object):
    def __init__(self, name):
        self.name = name
        self.filename = name+'/'+name+'_CC_CC_v08'
        self.cell_list = None
        self.eLLsNULL = None
        self.eLLsGLM = None
        self.edrift = None
        self.eRFcenter = None
        self.tc = None
        self.shift = None
        self.metric = None
        self.DFdat = None

    def load(self, datadir):
        matdat = sio.loadmat(datadir+"%s/%s_UT_stimpos.mat"%(self.name, self.name))
        self.cell_list = np.array(matdat['valUT'], dtype=np.int64).squeeze()
        self.eLLsNULL = np.array(matdat['LLsNULL'], dtype=np.float32).squeeze()
        self.eLLsGLM = np.array(matdat['LLsGLM'], dtype=np.float32).squeeze()
        self.edrift = np.array(matdat['drift_terms'], dtype=np.float32)
        self.eRFcenter = np.array(matdat['RFcenters'], dtype=np.int64)
        self.tc = np.array(matdat['top_corner'], dtype=np.int64).squeeze()

        # Shifts
        matdat = sio.loadmat(datadir+"%s/%s_CC_CC_shifts_best.mat"%(self.name, self.name))
        self.shift = np.array(matdat['ETshifts'], dtype=np.int64)
        self.metric = np.array(matdat['ETmetrics'], dtype=np.float32).squeeze()

        # get the updated DFs
        self.DFdat = sio.loadmat(datadir + self.name + '/' + self.name + '_CC_CC_DFupdate.mat')


# load data for all tests
num_lags = 16
L = 60
expt_names = ['J220715','J220722','J220801','J220808']
# load the experiments
experiments = []
for expt_name in expt_names:
    experiment = Experiment(expt_name)
    experiment.load(datadir)
    experiments.append(experiment)

# load the data
data = multidata.MultiClouds(
    datadir=datadir, filenames=[e.filename for e in experiments], eye_config=3, drift_interval=16,
    luminance_only=True, binocular=False, include_MUs=True, num_lags=num_lags,
    cell_lists=[e.cell_list for e in experiments],
    trial_sample=True)

# update the DFs
for ee, expt in enumerate(experiments):
    data.updateDF(ee, expt.DFdat['XDF'])

for ee, expt in enumerate(experiments):
    data.build_stim(ee, top_corner=expt.tc-(15,15), L=L, shifts=expt.shift)
data.assemble_stim()


def test_load_data():
    assert data is not None
    assert data.NT == 630240
    assert data.NC == 585

