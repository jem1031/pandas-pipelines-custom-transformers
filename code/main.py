import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from custom_transformers import ColumnExtractor, DFStandardScaler, DFFeatureUnion, DFImputer
from custom_transformers import DummyTransformer, Log1pTransformer, ZeroFillTransformer
from custom_transformers import DateFormatter, DateDiffer, MultiEncoder


# SET UP

# Read in data
# source: https://data.seattle.gov/Permitting/Special-Events-Permits/dm95-f8w5
data_folder = '../data/'
data_file = 'Special_Events_Permits_2016.csv'
df = pd.read_csv(data_folder + data_file)

# Set aside 25% as test data
df_train, df_test = train_test_split(df, random_state=4321)

# Take a look
df_train.head()


# SIMPLE MODEL

# Binary outcome
y_train = np.where(df_train.permit_status == 'Complete', 1, 0)
y_test = np.where(df_test.permit_status == 'Complete', 1, 0)

# Single feature
X_train_1 = df_train[['attendance']].fillna(value=0)
X_test_1 = df_test[['attendance']].fillna(value=0)

# Fit model
model_1 = LogisticRegression(random_state=5678)
model_1.fit(X_train_1, y_train)
y_pred_train_1 = model_1.predict(X_train_1)
p_pred_train_1 = model_1.predict_proba(X_train_1)[:, 1]

# Evaluate model
# baseline: always predict the average
p_baseline_test = [y_train.mean()]*len(y_test)
auc_baseline = roc_auc_score(y_test, p_baseline_test)
print(auc_baseline)  # 0.5
y_pred_test_1 = model_1.predict(X_test_1)
p_pred_test_1 = model_1.predict_proba(X_test_1)[:, 1]
auc_test_1 = roc_auc_score(y_test, p_pred_test_1)
print(auc_test_1)  # 0.576553672316


# MODEL W/PIPELINE

# Group columns by type of preprocessing needed
OUTCOME = 'permit_status'
NEAR_UNIQUE_FEATS = ['name_of_event', 'year_month_app', 'organization']
CAT_FEATS = [
    'permit_type', 'event_category', 'event_sub_category',
    'event_location_park', 'event_location_neighborhood']
MULTI_FEATS = ['council_district', 'precinct']
DATE_FEATS = ['application_date', 'event_start_date', 'event_end_date']
NUM_FEATS = ['attendance']

# Preprocessing with a Pipeline
pipeline = Pipeline([
    ('features', DFFeatureUnion([
        ('categoricals', Pipeline([
            ('extract', ColumnExtractor(CAT_FEATS)),
            ('dummy', DummyTransformer())
        ])),
        ('numerics', Pipeline([
            ('extract', ColumnExtractor(NUM_FEATS)),
            ('zero_fill', ZeroFillTransformer()),
            ('log', Log1pTransformer())
        ]))
    ])),
    ('scale', DFStandardScaler())
])
pipeline.fit(df_train)
X_train_2 = pipeline.transform(df_train)
X_test_2 = pipeline.transform(df_test)

# Fit model
model_2 = LogisticRegression(random_state=5678)
model_2.fit(X_train_2, y_train)
y_pred_train_2 = model_2.predict(X_train_2)
p_pred_train_2 = model_2.predict_proba(X_train_2)[:, 1]

# Evaluate model
p_pred_test_2 = model_2.predict_proba(X_test_2)[:, 1]
auc_test_2 = roc_auc_score(y_test, p_pred_test_2)
print(auc_test_2)  # 0.705084745763


# MODEL W/EVEN MORE FEATURES

# Preprocessing with a Pipeline
pipeline3 = Pipeline([
    ('features', DFFeatureUnion([
        ('dates', Pipeline([
            ('extract', ColumnExtractor(DATE_FEATS)),
            ('to_date', DateFormatter()),
            ('diffs', DateDiffer()),
            ('mid_fill', DFImputer(strategy='median'))
        ])),
        ('categoricals', Pipeline([
            ('extract', ColumnExtractor(CAT_FEATS)),
            ('dummy', DummyTransformer())
        ])),
        ('multi_labels', Pipeline([
            ('extract', ColumnExtractor(MULTI_FEATS)),
            ('multi_dummy', MultiEncoder(sep=';'))
        ])),
        ('numerics', Pipeline([
            ('extract', ColumnExtractor(NUM_FEATS)),
            ('zero_fill', ZeroFillTransformer()),
            ('log', Log1pTransformer())
        ]))
    ])),
    ('scale', DFStandardScaler())
])
pipeline3.fit(df_train)
X_train_3 = pipeline3.transform(df_train)
X_test_3 = pipeline3.transform(df_test)

# Fit model
model_3 = LogisticRegression(random_state=5678)
model_3.fit(X_train_3, y_train)
y_pred_train_3 = model_3.predict(X_train_3)
p_pred_train_3 = model_3.predict_proba(X_train_3)[:, 1]

# Evaluate model
p_pred_test_3 = model_3.predict_proba(X_test_3)[:, 1]
auc_test_3 = roc_auc_score(y_test, p_pred_test_3)
print(auc_test_3)  # 0.680790960452  # too many features -> starting to overfit


#
