# Evaluation: Accuracy
# Difficulty to use the pymedphys library

import json
import numpy as np

from utils.utils import gamma_analysis, error_analysis
from workflow import load_model_and_scale

# Load model and data hyperparameters
with open('../hyperparam.json', 'r') as hfile:
    param = json.load(hfile)

# Prepare input data.
path_test = '/home/moderato/Documents/iDoTa/data/test/'
name = 'unity_005'

# Evaluate the first 177 slices
testIDs = [*range(177)]

# Load normalization constants
transformer, scale = load_model_and_scale(name)

# Gamma evaluation.
indexes_gamma, gamma_pass_rate, gamma_dist = gamma_analysis(
    model=transformer,
    testIDs=testIDs,
    path=path_test,
    scale=scale,
    num_sections=1,
    cutoff=0.1
)
np.savez('./eval/gamma_005.npz', indexes_gamma, gamma_pass_rate, gamma_dist)

# Error evaluation.
indexes_error, errors, error_dist, rmse = error_analysis(
    model=transformer,
    testIDs=testIDs,
    path=path_test,
    scale=scale,
    num_sections=0,
    cutoff=0.1
)
np.savez('./eval/error_005.npz', indexes=indexes_error, error=errors, error_quad=error_dist, rmse=rmse)
