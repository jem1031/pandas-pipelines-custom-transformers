import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


# SET UP

# Read in data
data_folder = '../data/'
data_file = 'Special_Events_Permits_2016.csv'
df = pd.read_csv(data_folder + data_file)

# Set aside 25% as test data
df_train, df_test = train_test_split(df, random_state=4321)


# EXPLORATORY DATA ANALYSIS

# application_date
appdate_null_ct = df_train.application_date.isnull().sum()
print(appdate_null_ct)  # 3
# Comments:
# - few missing values

# permit_status - Outcome
status_cts = df_train.permit_status.value_counts(dropna=False)
print(status_cts)
# Complete      358
# Cancelled      31
# In Process      7
# Comments:
# - Complete v not (Cancelled or In Process) as binary outcome

# permit_type
type_cts = df_train.permit_type.value_counts(dropna=False)
print(type_cts)
# Special Event     325
# Valet Parking      61
# Charter Vessel     10
# Comments:
# - mostly Special Events
# - no missing values

# event_category
cat_cts = df_train.event_category.value_counts(dropna=False)
# Community            156
# NaN                   71
# Commercial            68
# Athletic              55
# Free Speech           40
# Mixed Free Speech      3
# Citywide               3
type_cat_cts = (
    df_train
    .groupby([df_train.permit_type, df_train.event_category.isnull()])
    .size())
print(type_cat_cts)
# permit_type     event_category
# Charter Vessel  True               10
# Special Event   False             325
# Valet Parking   True               61
# Comments:
# - present iff Special Event

# event_sub_category
subcat_cts = df_train.event_sub_category.value_counts(dropna=False)
print(subcat_cts)
# NaN         341
# Run/Walk     38
# Water         9
# Cycling       5
# Other         3
cat_subcat_cts = (
    df_train
    .groupby([df_train.event_category, df_train.event_sub_category.isnull()])
    .size())
print(cat_subcat_cts)
# event_category     event_sub_category
# Athletic           False                  55
# Citywide           True                    3
# Commercial         True                   68
# Community          True                  156
# Free Speech        True                   40
# Mixed Free Speech  True                    3
# Comments:
# - present iff Athletic Special Event

# name_of_event
name_null_ct = df_train.name_of_event.isnull().sum()
print(name_null_ct)  # 0
name_cts = df_train.name_of_event.value_counts(dropna=False)
print(len(name_cts))  # 392
# Comments:
# - no missing values
# - almost all unique

# year_month_app
yma_null_ct = df_train.year_month_app.isnull().sum()
print(yma_null_ct)  # 0
yma_cts = df_train.year_month_app.value_counts(dropna=False)
print(len(yma_cts))  # 396
# Comments:
# - no missing values
# - all unique

# event_start_date
startdate_null_ct = df_train.event_start_date.isnull().sum()
print(startdate_null_ct)  # 0
# Comments:
# - no missing values

# event_end_date
enddate_null_ct = df_train.event_end_date.isnull().sum()
print(enddate_null_ct)  # 0
multiday_ct = (df_train.event_start_date != df_train.event_end_date).sum()
print(multiday_ct)  # 49
# Comments:
# - no missing values
# - about 10% multi-day events

# event_location_park
park_cts = df_train.event_location_park.value_counts(dropna=False)
print(park_cts)
# NaN                                    364
# Magnuson Park                            8
# Gas Works Park                           5
# Occidental Park                          3
# Greenlake Park                           2
# Volunteer Park                           2
# Seattle Center                           1
# Seward Park                              1
# Anchor Park                              1
# Madison Park                             1
# OTHER                                    1
# Myrtle Edwards Park                      1
# Martin Luther King Jr Memorial Park      1
# Hamilton Viewpoint Park                  1
# Ballard Commons Park                     1
# Lake Union Park                          1
# Judkins Park                             1
# Bell Street Park                         1
# Comments:
# - about 90% missing values
# - could be new values in test data
# - Note: there are 400+ parks in Seattle

# event_location_neighborhood
neighborhood_null_ct = df_train.event_location_neighborhood.isnull().sum()
print(neighborhood_null_ct)  # 0
neighborhood_cts = df_train.event_location_neighborhood.value_counts(dropna=False)
print(len(neighborhood_cts))  # 42
print(neighborhood_cts.head())
# Downtown                  71
# Pioneer Square            41
# Capitol Hill              26
# Queen Anne                26
# Wallingford               22
# ...
# Comments:
# - no missing values
# - could be new values in test data
# - Note: there are ?? neighborhoods in Seattle

# council_district
district_cts = df_train.council_district.value_counts(dropna=False)
print(district_cts)
# 7                128
# 3                 81
# 2                 61
# 4                 53
# 6                 29
# 1                 14
# 3;4                8
# 5                  5
# 4;6                3
# 3;7                3
# 2;3                3
# 1;2;3;4;6;7        1
# 4;7                1
# 3;4;7              1
# 4;6;7              1
# 2;3;4              1
# 2;3;4;7            1
# 1;2;3;4;5;6;7      1
# 2;7                1
# Comments:
# - no missing values
# - combinations separated by semi-colon
# - could be new combinations in test data

# precinct
precinct_cts = df_train.precinct.value_counts(dropna=False)
print(precinct_cts)
# West                     186
# North                     92
# South                     51
# East                      51
# Southwest                  6
# North;South                4
# East;West                  3
# East;South                 1
# North;West                 1
# East;North;South;West      1
# Comments:
# - no missing values
# - combinations separated by semi-colon
# - could be new combinations in test data

# organization
org_null_ct = df_train.organization.isnull().sum()
print(org_null_ct)  # 1
org_cts = df_train.organization.value_counts(dropna=False)
print(len(org_cts))  # 245
# Comments:
# - few missing values
# - many different values
# - could be new values in test data

# attendance
attendance_null_ct = df_train.attendance.isnull().sum()
print(attendance_null_ct)  # 3
print(df_train.attendance.describe())
# count       393.000000
# mean       3716.913486
# std       16097.152814
# min          15.000000
# 25%         200.000000
# 50%         640.000000
# 75%        1800.000000
# max      204000.000000
# Histogram of Attendance:
x = df_train.attendance[df_train.attendance > 0]
n, bins, patches = plt.hist(x, 50, edgecolor='black', facecolor='gray', alpha=0.75)
plt.xlabel('Attendance')
plt.ylabel('# Events')
plt.title('Histogram of Attendance - Raw Scale')
plt.show()
# Histogram of Log(Attendance):
x = np.log(x)
n, bins, patches = plt.hist(x, 50, edgecolor='black', facecolor='gray', alpha=0.75)
plt.xlabel('Log(Attendance)')
plt.ylabel('# Events')
plt.title('Histogram of Attendance - Log Scale')
plt.show()
# Comments:
# - few missing values
# - better on log-scale


#
