"""dataset.py

    Author: Dany Vohl (2020-2021)
"""
from .utils import load
import numpy as np
import pandas as pd

def load_dataframe(input_filterbanks_repository, state_variable_name):
    print ('Reloading dataframe %s' % (state_variable_name))
    df_R3 = load(state_variable_name)

    if 'detection_isot' not in df_R3.keys():
        try:
            df_R3['detection_isot'] = pd.to_datetime(df_R3['observation_datetime'], format='%Y-%m-%dT%H:%M:%S.%f') + pd.to_timedelta(df_R3['detection_time'], unit='s')
        except:
            print ('Setting detection_isot did not complete.')

    if input_filterbanks_repository not in df_R3.iloc[0]['file_location']:
        if input_filterbanks_repository[-1] != '/':
            input_filterbanks_repository += '/'

        for i, row in df_R3.iterrows():
            df_R3.loc[df_R3['detection_isot'] == row['detection_isot'], 'file_location'] = "%s/%s" % (
                input_filterbanks_repository,
                row['file_location'].split('/')[-1]
            )

    return df_R3
