import ipdb

import time
import os

import numpy as np

import ujson as uj

from config import config

if not os.path.exists(config.pgm_model_folder + 'beliefs/'): os.mkdir(config.pgm_model_folder + 'beliefs/')
if not os.path.exists(config.pgm_model_folder + 'priors/'): os.mkdir(config.pgm_model_folder + 'priors/')
if not os.path.exists(config.pgm_model_folder + 'messages/'): os.mkdir(config.pgm_model_folder + 'messages/')
if not os.path.exists(config.pgm_model_folder + 'connections/'): os.mkdir(config.pgm_model_folder + 'connections/')


def _store_factor_parameters(group, fname, parameters):
    fname = config.pgm_model_folder + group + '/' + fname
    with open(fname, 'w') as filehandler:
        uj.dump(parameters, filehandler)

def _read_factor_parameters(group, fname):
    fname = config.pgm_model_folder + group + '/' + fname
    with open(fname, 'r') as filehandler:
        parameters = uj.load(filehandler)
    return parameters

def store_beliefs(fname, parameters):
    _store_factor_parameters('beliefs', fname, parameters)

def read_beliefs(fname):
    return _read_factor_parameters('beliefs', fname)

def store_priors(fname, parameters):
    _store_factor_parameters('priors', fname, parameters)
    
def read_priors(fname):
    return _read_factor_parameters('priors', fname)

def store_messages(fname1, fname2,  parameters):
    mname = fname1 + '->' + fname2
    _store_factor_parameters('messages', mname, parameters)
    store_connections(fname1, fname2)

def read_messages(fname1, fname2):
    mname = fname1 + '->' + fname2
    return _read_factor_parameters('messages', mname)

def store_connections(fname1, fname2):
    connections = read_connections(fname1)
    connections.append(fname2)
    connections = np.unique(connections).tolist()
    _store_factor_parameters('connections', fname1, connections)

def read_connections(fname):
    try:
        connections = _read_factor_parameters('connections', fname)
    except:
        connections = []

    return connections
