# Evaluation: Time

import json

import numpy as np
from tensorflow.python.framework.config import list_physical_devices, set_memory_growth

from utils.utils import time_analysis
from workflow import load_model_and_scale

print(list_physical_devices('GPU'))
gpu_devices = list_physical_devices('GPU')
for device in gpu_devices:
    set_memory_growth(device, True)

batch_size = 20

# Load data hyperparameters.
with open('../hyperparam.json', 'r') as hfile:
    param = json.load(hfile)

# Prepare input data.
path = '/home/moderato/Documents/iDoTa/data/test/'
name = 'unity_005'

# Evaluate the first 177 slices
testIDs = [*range(177)]

# Load normalization constants
transformer, scale = load_model_and_scale(name)

## Define and load the transformer model.
transformer.summary()

times = time_analysis(
    model=transformer,
    testIDs=testIDs,
    path=path,
    scale=scale,
    batch_size=batch_size
)
np.savez('./eval/time_analysis.npz', times)
