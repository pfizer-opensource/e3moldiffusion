import os
import numpy as np
import torch
import pickle
path = "/hpfs/userws/let55/experiments/e3moldiffusion/crossdocked/tmp"

if not os.path.exists(path):
    os.makedirs(path)
    
x = np.random.randn(100,)
with open(path + "/test.pkl", "wb") as f:
    pickle.dump(x, f)