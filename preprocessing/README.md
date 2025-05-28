
# Preprocessing 

This module generates the inputs and outputs of the neural network.

## _align.py_

To avoid unnecessary copy of raw data, the `align.py` file is used to generate a database in Excel format. 
The generated file contains the absolute paths, the gantry, the MU and the dose grid scaling.

It works with the folder structure of Moderato by matching the beam ID of the ray and the dose. 
  ```
  treatments/
  │
  ├── subject_0
  │   ├── ct 
  │   │   ├── ct_0.dcm
  │   │   ├── ct_1.dcm
  │   │   └── ...
  │   └── doses
  │       ├── task_0 - each task corresponds to a segment
  │       │   ├── doses.dcm
  │       │   ├── manifest.json - segment info
  │       │   └── settings.json - patient and beam info
  │       └── ...
  └── ...

  ```

## _generate.py_

Generates data for Tensorflow, each sample is stored inside a .npz file. `
Torchio library is used to load and preprocess the dicoms.
