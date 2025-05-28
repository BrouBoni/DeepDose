import os
import time
import numpy as np
import pandas as pd
import torch
import torchio as tio

from utils.utils import rotations_to_target_range, load_model_and_scale
from utils.utils import set_gpu

set_gpu(0)

def generate_inputs(path, transformer, scale, cutoff=0.5):

    df = pd.read_excel(path)
    segments_number = len(df)
    print(segments_number)

    r_total, y_total, predicted_total = 3 *[torch.zeros(1, 128, 128, 96)]
    length = 0
    for index, row in df.iterrows():
            print(index)
            gantry = row['gantry']
            mu = row['MU']
            dose_grid_scaling = row['dose_grid_scaling']
            ray_grid_scaling = row['ray_grid_scaling']

            x = tio.ScalarImage(row['ct_path'])
            r = tio.ScalarImage(row['ray_path'])
            y = tio.ScalarImage(row['dose_path'])

            r.data = r.data * ray_grid_scaling
            y.data = y.data * dose_grid_scaling

            rotations = rotations_to_target_range(int(gantry))

            HOUNSFIELD_AIR, HOUNSFIELD_BONE = -1000, 3000

            transform = tio.Compose([
                tio.ToCanonical(),
                tio.Resample('dose'),
                tio.Clamp(out_min=HOUNSFIELD_AIR, out_max=HOUNSFIELD_BONE),
                tio.EnsureShapeMultiple(8, method='crop'),
                tio.CropOrPad((128, 128, 96)),
            ])

            s = tio.Subject(
                ct = x,
                ray = r,
                dose = y,
            )

            s = transform(s)

            # foward
            ct = np.rot90(s.ct.numpy()[0], 1)
            ct = np.fliplr(ct)

            ray = np.rot90(s.ray.numpy()[0], 1)
            ray = np.fliplr(ray)

            dose = np.rot90(s.dose.numpy()[0], 1)
            dose = np.fliplr(dose)

            x_left = np.rot90(ct, k=rotations, axes=(0, 1))
            r_left = np.rot90(ray, k=rotations, axes=(0, 1))
            y_left = np.rot90(dose, k=rotations, axes=(0, 1))

            r_left = np.clip(r_left, 0, 30) / mu
            y_left = np.clip(y_left, 0, 30) / mu

            x_left =np.rot90(x_left, axes=(1, 2))
            r_left = np.rot90(r_left, axes=(1, 2))
            y_left = np.rot90(y_left, axes=(1, 2))

            geometry = np.expand_dims(x_left, axis=(0, -1))
            ct_vol = (geometry - scale['x_min']) / (scale['x_max'] - scale['x_min'])
            raytrace = np.expand_dims(r_left, axis=(0, -1))
            ray_tr = (raytrace - scale['r_min']) / (scale['r_max'] - scale['r_min'])

            start = time.time()

            prediction = transformer.predict([ct_vol, ray_tr], verbose=0)
            length = length + time.time() - start

            prediction = prediction * (scale['y_max'] - scale['y_min']) + scale['y_min']
            prediction[prediction < (cutoff / 100) * scale['y_max']] = 0

            prediction = np.squeeze(prediction)

            # backward
            y_right = np.rot90(prediction, axes=(2, 1))
            y_right = y_right * mu
            y_right = np.rot90(y_right, k=rotations, axes=(1, 0))
            y_right = np.fliplr(y_right)
            y_right = np.rot90(y_right, -1)
            y_right = np.expand_dims(y_right, 0).copy()

            s.add_image(tio.ScalarImage(tensor=torch.tensor(y_right)), 'predicted')

            r_total = r_total + s.ray.data
            y_total = y_total + s.dose.data
            predicted_total = predicted_total + s.predicted.data

    return s.ct, r_total, y_total, predicted_total

if __name__ == '__main__':

    name = 'unity'
    transformer, scale = load_model_and_scale(name)

    # Change ID
    subject_id = "010101"
    path = os.path.join("/home/moderato/Documents/iDoTa/prediction/", subject_id)
    path_xlsx = os.path.join(path, subject_id+".xlsx")
    ct, r_total, y_total, predicted_total = generate_inputs(path_xlsx, transformer, scale)

    s = tio.Subject(
        ct = ct,
        ray = tio.ScalarImage(tensor = r_total),
        dose = tio.ScalarImage(tensor = y_total),
        predicted = tio.ScalarImage(tensor=predicted_total),
    )

    s.plot()
    s.ray.affine = s.ct.affine
    s.dose.affine = s.ct.affine
    s.predicted.affine = s.ct.affine

    s.ct.save(os.path.join(path, 'ct.nii.gz'))
    s.ray.save(os.path.join(path, 'ray.nii.gz'))
    s.dose.save(os.path.join(path, 'dose.nii.gz'))
    s.predicted.save(os.path.join(path, 'predicted.nii.gz'))
