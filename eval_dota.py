import json
import os

from utils.utils import infer
from src.models import idota
from utils.plot import plot_slice, plot_beam
from utils.utils import set_gpu

set_gpu(0)

# Load model and data hyperparameters.
with open("./hyperparam.json", "r") as hfile:
    param = json.load(hfile)

# Prepare input data.
path = "./data/test/"

checkpoints_dir = './checkpoints/'
name = 'unity'
expr_dir = os.path.join(checkpoints_dir, name)

path_weights = os.path.join(expr_dir, 'weights.ckpt')
testIDs = [0, 2, 31]

# Load normalization constants.
if os.path.exists(os.path.join(expr_dir, 'scale.json')):
    with open(os.path.join(expr_dir, 'scale.json'), 'r') as fp:
        scale = json.load(fp)
else:
    scale = {'y_min':0, 'y_max':0.05,
            'r_min':0, 'r_max':0.05,
            'x_min':-1000, 'x_max':3000}

transformer = idota(
    inshape=param['inshape'],
    steps=param['num_levels'],
    enc_feats=param['enc_feats'],
    num_heads=param['num_heads'],
    num_transformers=param['num_transformers'],
    kernel_size=param['kernel_size']
)
transformer.summary()

# Load weights from checkpoint.
transformer.load_weights(path_weights)

inputs, prediction, ground_truth = infer(transformer, testIDs[0], path, scale)
plot_beam(inputs, ground_truth, prediction,  slices=10, gamma_evaluation=False)

inputs, prediction, ground_truth = infer(transformer, testIDs[0], path, scale)
prediction[ground_truth==0] = 0
plot_slice(inputs, ground_truth, prediction, scale, slice_number=64, gamma_slice=False)
