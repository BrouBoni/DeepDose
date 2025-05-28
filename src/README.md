
# src 

The core module

## _generators.py_

Defines the dataloader for training and inference.

The samples are `.npz` file with at least 3 arrays. The `vol` for the CT, the `ray` for the first dose estimate and the `dose` for the dose.
The volume in _HU_ and the doses in _Gy/MU_.
These arrays will be normalised according to the scale dictionary defined in the `train_dota.py`.

To keep track of the patient info, the `subject`, `segment`, `mu` and `gantry` are also saved.

## _blocks.py_

Defines the building blocks (layers) needed to build the iDoTa model.

## _model.py_

Defines the iDoTa model.
