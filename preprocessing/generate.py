import numpy as np
import pandas as pd
import torchio as tio

from utils.utils import rotations_to_target_range

if __name__ == '__main__':

    path = '/home/moderato/Documents/iDoTa/preprocessing/output.xlsx'
    df_all = pd.read_excel(path, sheet_name=None)

    ID = 0

    for subject, df in df_all.items():
        print(subject)
        for index, row in df.iterrows():
                # Load attributes
                gantry = row['gantry']
                mu = row['MU']
                dose_grid_scaling = row['dose_grid_scaling']
                ray_grid_scaling = row['ray_grid_scaling']

                # Loading the data
                x = tio.ScalarImage(row['ct_path'])
                r = tio.ScalarImage(row['ray_path'])
                y = tio.ScalarImage(row['dose_path'])

                # Dose grid dcaling
                r.data = r.data * ray_grid_scaling
                y.data = y.data * dose_grid_scaling

                # Number of rotations to be performed in order to be between 225° and 316°
                rotations = rotations_to_target_range(int(gantry))

                HOUNSFIELD_AIR, HOUNSFIELD_BONE = -1000, 3000

                transform = tio.Compose([
                    tio.ToCanonical(),  # RAS+ orientation
                    tio.Resample('dose'),   # Resample the CT to the dose physical space
                    tio.Clamp(out_min=HOUNSFIELD_AIR, out_max=HOUNSFIELD_BONE), # Clamp HU values
                    tio.EnsureShapeMultiple(8, method='crop'), # Ensure that shape is divisible by 8
                    tio.CropOrPad((128, 128, 96)),  # Modify the target view by cropping to the target shaoe.
                ])

                # Data structure used in Torchio
                s = tio.Subject(
                    ct = x,
                    ray = r,
                    dose = y,
                )

                # Apply the previous list of transforms
                s = transform(s)

                # Beam always travels in the same direction along the first dimension D=128.
                ct = np.rot90(s.ct.numpy()[0], 1)
                ct = np.fliplr(ct)

                ray = np.rot90(s.ray.numpy()[0], 1)
                ray = np.fliplr(ray)

                dose = np.rot90(s.dose.numpy()[0], 1)
                dose = np.fliplr(dose)

                x_left = np.rot90(ct, k=rotations, axes=(0, 1))
                r_left = np.rot90(ray, k=rotations, axes=(0, 1))
                y_left = np.rot90(dose, k=rotations, axes=(0, 1))

                x_left =np.rot90(x_left, axes=(1, 2))
                r_left = np.rot90(r_left, axes=(1, 2))
                y_left = np.rot90(y_left, axes=(1, 2))

                # Dose per MU ( 30 Gy is considered the max per segment)
                r_left = np.clip(r_left, 0, 30) / mu
                y_left = np.clip(y_left, 0, 30) / mu

                # For visualization (uncomment following lines)
                # x_left_s = np.expand_dims(x_left, 0).copy()
                # r_left_s = np.expand_dims(r_left, 0).copy()
                # y_left_s = np.expand_dims(y_left, 0).copy()
                #
                # s_left = tio.Subject(
                #     ct=tio.ScalarImage(tensor=torch.tensor(x_left_s)),
                #     ray=tio.ScalarImage(tensor=torch.tensor(r_left_s)),
                #     dose=tio.ScalarImage(tensor=torch.tensor(y_left_s)),
                # )
                #
                # s_left.plot()

                # Save each sample
                np.savez('/home/moderato/Documents/iDoTa/data/train/'+str(ID), vol=x_left, ray=r_left, dose=y_left, mu=mu, gantry=gantry, subject=subject, segment=row['beam'])
                print('saved '+str(ID)+'.npz')
                ID += 1