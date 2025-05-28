import json
import os

import numpy as np
import pandas as pd
import pydicom

if __name__ == '__main__':

    path = '/home/moderato/Documents/moderato/release/traitments'

    subjects = os.listdir(path)
    subjects = [subject for subject in subjects if not subject.startswith('.') and subject != '000000' and subject != '401402829' and subject != '105528' and subject != '401766861' and subject != '401733561' and subject != '260471' and subject != '401410426' and subject != '4001469145 cluster2' and subject != '400149021~b' and subject != '401361361']

    segments_subjects = []
    df_all = []

    for subject in subjects:
        print("Subject #" + subject)

        subject_path = os.path.join(path, subject)
        ct_path = os.path.join(subject_path, 'ct')

        doses_path = os.path.join(subject_path, 'doses')
        doses = os.listdir(doses_path)
        doses = [dose for dose in doses if dose.startswith('task')]

        df = pd.DataFrame(np.nan, index=range(0, int(len(doses)/2)), columns=['subject_id', 'beam', 'ct_path', 'ray_path', 'dose_path', 'gantry', 'MU'])

        for dose in doses:
            dose_path = os.path.join(doses_path, dose, 'doses.dcm')
            settings_path = os.path.join(doses_path, dose, 'settings.json')
            with open(settings_path, 'r') as f:
                settings = json.load(f)
                subject_id = settings['traitment']
                beam_id = settings['beam']
                no_secondary_e = settings['ext']['noelectron']

            manifest_path = os.path.join(doses_path, dose, 'manifest.json')
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

                gantry = manifest['beam']['data']['gantry']
                mu = manifest['beam']['data']['weight']

            df.at[beam_id, 'subject_id'] = int(subject_id)
            df.at[beam_id, 'beam'] = int(beam_id)
            df.at[beam_id, 'ct_path'] = str(ct_path)
            df.at[beam_id, 'MU'] = float(mu)
            df.at[beam_id, 'gantry'] = int(gantry)

            dose_grid_scaling = pydicom.dcmread(dose_path).DoseGridScaling
            if no_secondary_e:
                df.at[beam_id, 'ray_path'] = str(dose_path)
                df.at[beam_id, 'ray_grid_scaling'] = dose_grid_scaling
            else:
                df.at[beam_id, 'dose_path'] = str(dose_path)
                df.at[beam_id, 'dose_grid_scaling'] = dose_grid_scaling


        df['subject_id'] = df['subject_id'].astype(int)
        df['beam'] = df['beam'].astype(int)
        df['ct_path'] = df['ct_path'].astype(str)
        df['MU'] = df['MU'].astype(float)
        df['gantry'] = df['gantry'].astype(float)
        df['dose_path'] = df['dose_path'].astype(str)
        df['ray_path'] = df['ray_path'].astype(str)
        df_all.append(df)

    with pd.ExcelWriter('/home/moderato/Documents/iDoTa/preprocessing/output.xlsx') as writer:
        for index, df in enumerate(df_all):
            print(str(df.at[0, 'subject_id']))
            df.to_excel(writer, sheet_name=str(df.at[0, 'subject_id']))
