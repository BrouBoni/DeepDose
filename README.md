# Dose calculation via Transformers #

This is a repository to predict dose distribution using a deep learning model (iDoTa) based on the patient geometry and the general beam shape.

This repo is created for the purpose of the collaboration between Elekta and the Institut Jules Bordet.

This work is based on Pastor-Serrano, O., & Perkó, Z. (2022). Millisecond speed deep learning based proton dose calculation with Monte Carlo accuracy.
<https://doi.org/10.1088/1361-6560/ac692e>.


## Usage/Examples

### Hyperparameters file format
Hyperparam files are in `.json` format:
```javascript
{
    "inshape": [128, 96, 128, 1],
    "kernel_size": 3, 
    "num_heads": 4,
    "num_transformers": 1,
    "enc_feats": 10,
    "num_levels": 4,
}
```

### Training

Once the data are generated and stored in `data/train/`, you can run `train_dota.py`.

### Evaluation

It is possible to evaluate quickly the model with `eval_dota.py`. You need to put the samples in the folder `data/test/`.

### Workflow

To predict a full dose distribution, you will need an excel file similar like the one described in the procedure in `.preprocessing/`.
Based on this file, the code will look after the dicoms, make the preprocessing, then the prediction for each segment.
The doses predicted will be repositioned in the conventional orientation and then accumulated.

Run with `workflow.py`. 

## Requirements ##

* Tensorflow 2.11.0
* Torchio 0.20.4
* tensorflow-addons 0.21.0

## Notes on Performance

34GB is needed to train the model (size 128 x 96 x 128) and 10 hours for 1600 samples.
I would consider using the patch aggregator of Torchio.