import pandas as pd
import numpy as np
import re

# Read in raw data
# source: https://data.seattle.gov/Permitting/Special-Events-Permits/dm95-f8w5
data_folder = '../data/'
data_file_raw = 'Special_Events_Permits.csv'
df_raw = pd.read_csv(data_folder + data_file_raw)


# Switch column names to lower_case_with_underscores
def standardize_name(cname):
    cname = re.sub(r'[-\.]', ' ', cname)
    cname = cname.strip().lower()
    cname = re.sub(r'\s+', '_', cname)
    return cname

df_raw.columns = df_raw.columns.map(standardize_name)


# Filter to 2016 events
df_raw['event_start_date1'] = pd.to_datetime(df_raw.event_start_date)
df = df_raw[np.logical_and(df_raw.event_start_date1 >= '2016-01-01',
                           df_raw.event_start_date1 <= '2016-12-31')]
df = df.drop('event_start_date1', axis=1)


# Export data
data_file = 'Special_Events_Permits_2016.csv'
df.to_csv(data_folder + data_file, index=False)


#
