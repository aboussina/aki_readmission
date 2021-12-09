#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
Program: get_readmission_features.ipynb

Purpose:
1.  Query measurements and observations for the AKI cohort
2.  Perform initial feature selection
3.  Preprocess data
4.  Train models

Author: Aaron Boussina
'''


# In[ ]:


import pandas as pd
import subprocess
import os
import datetime
import numpy as np

import statsmodels.api as sm
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    confusion_matrix, accuracy_score, roc_auc_score
)

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras import regularizers


import matplotlib.pyplot as plt


# In[ ]:


'''
Initialize Constants
'''
my_bucket = os.getenv('WORKSPACE_BUCKET')
dataset = os.getenv('WORKSPACE_CDR')


# In[ ]:


aki_cohort = pd.read_csv("aki_cohort.csv")
planned_visits =  pd.read_csv("planned_visits.csv")


# In[ ]:


all_adults = pd.read_csv('All_Adults_df.csv')


# In[ ]:


all_features = pd.read_csv('all_features.csv')


# In[ ]:


'''
Import the aki_cohort dataset.  
This contains all AKI encounters and the readmission labels
'''
for infile in ["planned_visits.csv"]:
    args = ["gsutil", "cp", f"{my_bucket}/data/{infile}", infile]
    output = subprocess.run(args, capture_output=True)
    print(output.stderr)

aki_cohort = pd.read_csv("aki_cohort.csv")
planned_visits =  pd.read_csv("planned_visits.csv")


# In[ ]:


for infile in ["aki_cohort.csv"]:
    args = ["gsutil", "cp", infile, f"{my_bucket}/data/{infile}"]
    output = subprocess.run(args, capture_output=True)
    print(output.stderr)


# In[ ]:


visit_ids = tuple(
    aki_cohort.visit_occurrence_id.unique()
)

person_ids = tuple(
    aki_cohort.person_id.unique()
)

num_persons = len(person_ids)
num_visits = len(visit_ids)


# ##  Extract Observations, Measurements, Procedures, Conditions and Demographics

# In[ ]:


'''
Identify features that are well populated
'''

common_observations_query = f'''
    SELECT
        observation_concept_id,
        COUNT(*)/{num_persons} as obs_freq
    FROM (
        SELECT DISTINCT
            person_id,
            observation_concept_id
        FROM
            {dataset}.observation
        WHERE
            person_id IN {person_ids}
            AND (
                value_as_string IS NOT NULL
                OR value_as_concept_id IS NOT NULL
                OR value_as_number IS NOT NULL
            )
    ) obs
    GROUP BY
        observation_concept_id
    HAVING
        obs_freq >= 0.3
'''

common_measurements_query = f'''
    SELECT
        measurement_concept_id,
        COUNT(*)/{num_visits} as meas_freq
    FROM (
        SELECT DISTINCT
            visit_occurrence_id,
            measurement_concept_id
        FROM
            {dataset}.measurement
        WHERE
            visit_occurrence_id IN {visit_ids}
            AND (
                value_as_concept_id IS NOT NULL
                OR value_as_number IS NOT NULL
            )
    ) obs
    GROUP BY
        measurement_concept_id
    HAVING
        meas_freq >= 0.30
'''

common_procedures_query = f'''
    SELECT
        procedure_concept_id,
        COUNT(*)/{num_visits} as proc_freq
    FROM (
        SELECT DISTINCT
            visit_occurrence_id,
            procedure_concept_id
        FROM
            {dataset}.procedure_occurrence
        WHERE
            visit_occurrence_id IN {visit_ids}
    ) procs
    GROUP BY
        procedure_concept_id
    HAVING
        proc_freq >= 0.1
'''

common_comorbidities_query = f'''
    SELECT
        condition_concept_id,
        COUNT(*)/{num_visits} as cond_freq
    FROM (
        SELECT DISTINCT
            visit_occurrence_id,
            condition_concept_id
        FROM
            {dataset}.condition_occurrence
        WHERE
            visit_occurrence_id IN {visit_ids}
    ) conds
    GROUP BY
        condition_concept_id
    HAVING
        cond_freq >= 0.1
'''


# In[ ]:


x = f'''
    SELECT
        COUNT(distinct person_id)
    FROM 
        {dataset}.person
'''


# In[ ]:


all_features


# In[ ]:


common_observations = pd.read_gbq(common_observations_query)


# In[ ]:


common_measurements = pd.read_gbq(common_measurements_query)


# In[ ]:


common_procedures = pd.read_gbq(common_procedures_query)


# In[ ]:


common_conditions = pd.read_gbq(common_comorbidities_query)


# In[ ]:


common_observation_ids = tuple(
    common_observations.observation_concept_id
)

common_measurement_ids = tuple(
    common_measurements.measurement_concept_id
)

common_procedure_ids = tuple(
    common_procedures.procedure_concept_id
)

common_condition_ids = tuple(
    common_conditions.condition_concept_id
)


# In[ ]:


observations_query = f'''
    SELECT
        person_id,
        observation_concept_id,
        observation_datetime,
        value_as_number,
        value_as_string,
        value_as_concept_id,
        unit_concept_id,
        visit_occurrence_id
    FROM
        {dataset}.observation
    WHERE
        person_id IN {person_ids}
        AND observation_concept_id IN {common_observation_ids}
        AND (
            value_as_string IS NOT NULL
            OR value_as_concept_id IS NOT NULL
            OR value_as_number IS NOT NULL
        )
'''

measurements_query = f'''
    SELECT
        person_id,
        measurement_concept_id,
        measurement_datetime,
        value_as_number,
        value_as_concept_id,
        unit_concept_id,
        visit_occurrence_id
    FROM
        {dataset}.measurement
    WHERE
        visit_occurrence_id IN {visit_ids}
        AND measurement_concept_id IN {common_measurement_ids}
        AND (
            value_as_concept_id IS NOT NULL
            OR value_as_number IS NOT NULL
        )
'''

procedures_query = f'''
    SELECT
        person_id,
        procedure_concept_id,
        procedure_datetime,
        visit_occurrence_id
    FROM
        {dataset}.procedure_occurrence
    WHERE
        visit_occurrence_id IN {visit_ids}
        AND procedure_concept_id IN {common_procedure_ids}
'''


conditions_query = f'''
    SELECT
        person_id,
        condition_concept_id,
        condition_start_datetime,
        condition_end_datetime,
        visit_occurrence_id
    FROM
        {dataset}.condition_occurrence
    WHERE
        visit_occurrence_id IN {visit_ids}
        AND condition_concept_id IN {common_condition_ids} 
'''


dm_query = f'''
    SELECT
        person_id,
        gender_concept_id,
        year_of_birth,
        race_concept_id,
        ethnicity_concept_id,
        location_id
    FROM
        {dataset}.person
    WHERE
        person_id IN {person_ids}
'''


# In[ ]:


observations = pd.read_gbq(observations_query)


# In[ ]:


measurements = pd.read_gbq(measurements_query)


# In[ ]:


procedures = pd.read_gbq(procedures_query)


# In[ ]:


conditions = pd.read_gbq(conditions_query)


# In[ ]:


dm = pd.read_gbq(dm_query)


# In[ ]:


'''
Preserve Intermediate Datasets
'''
observations.to_csv('observations.csv', index=False)
measurements.to_csv('measurements.csv', index=False)
procedures.to_csv('procedures.csv', index=False)
conditions.to_csv('conditions.csv', index=False)
dm.to_csv('dm.csv', index=False)


# In[ ]:


measurements = pd.read_csv('measurements.csv')
observations = pd.read_csv('observations.csv')
dm = pd.read_csv('dm.csv')
procedures = pd.read_csv('procedures.csv')
conditions =  pd.read_csv('conditions.csv')


# In[ ]:


aki_cohort = pd.read_csv('aki_cohort.csv')


# In[ ]:


for infile in [
    'observations.csv', 'measurements.csv', 'procedures.csv', 'conditions.csv', 'dm.csv'
]:
    args = ["gsutil", "cp", infile, f"{my_bucket}/data/{infile}"]
    output = subprocess.run(args, capture_output=True)


# In[ ]:


for infile in [
    'aki_cohort.csv'
]:
    args = ["gsutil", "cp", f"{my_bucket}/data/{infile}", infile]
    output = subprocess.run(args, capture_output=True)


# In[ ]:


aki_cohort = pd.read_csv('aki_cohort.csv')


# In[ ]:


'''
Reimport if needed

for infile in [
    'observations.csv', 'measurements.csv', 'dm.csv'
]:
    args = ["gsutil", "cp", f"{my_bucket}/data/{infile}", infile]
    output = subprocess.run(args, capture_output=True)

measurements = pd.read_csv('measurements.csv')
observations = pd.read_csv('observations.csv')
dm = pd.read_csv('dm.csv')
'''


# ## Preprocess Data

# For categorical measurements, preserve the last value and the counts for each value.
# 
# For continuous measurements, aggregate the mean, min, max and standard deviation.
# 
# For observations, preserve the last recorded value for the encounter.

# In[ ]:


encounters = aki_cohort.get([
    'Encounter', 'person_id', 'visit_occurrence_id', 'hospital_discharge_time'
])

def get_categorical_counts(df, prefix='count_', meas=True):
    '''
    Takes a tall & skinny dataframe and returns a wide frame
    with each column corresponding to a specific categorical value
    for a given observation and encounter
    '''
    if meas:
        concept_id = 'measurement_concept_id'
    else:
        concept_id = 'observation_concept_id'
    
    cat_counts = (
        df
        .groupby(['Encounter', concept_id])
        .value_as_concept_id
        .value_counts()
        .reset_index('Encounter')
        .rename(
            {'value_as_concept_id': 'value_count'},
            axis=1
        )
        .reset_index()
    )

    cat_counts['column'] = (
        prefix
        + cat_counts[concept_id].astype(str) 
        + '_' 
        + cat_counts.value_as_concept_id.astype(str)
    )

    enc_cat = (
        cat_counts
        .pivot(index='Encounter', columns='column', values='value_count')
        .fillna(0)
        .reset_index('Encounter')
    )
    
    return enc_cat


# In[ ]:


def get_continuous_stats(df):
    '''
    Takes a small & skinny dataframe and returns a wide frame
    with the first, last, mean, standard deviation, minimum, and maximum
    for a given obersvation and encounter
    '''
    def give_prefix(x, prefix):
        return (
            prefix
            + x['measurement_concept_id'].astype(str)
        )
    
    firsts = (
        df.drop_duplicates(
            ['Encounter', 'measurement_concept_id']
        )
        .assign(column = lambda x: give_prefix(x, 'first_'))
        .get(['Encounter', 'column', 'value_as_number'])
    )
    
    lasts = (
        df.drop_duplicates(
            ['Encounter', 'measurement_concept_id'],
            keep='last'
        )
        .assign(column = lambda x: give_prefix(x, 'last_'))
        .get(['Encounter', 'column', 'value_as_number'])
    )
        
    values = (
        df
        .groupby(['Encounter', 'measurement_concept_id'])
        .value_as_number
    )

    means = values.mean().reset_index().assign(
        column =  lambda x: give_prefix(x, 'mean_')
    )
    stds = values.std().reset_index().assign(
        column =  lambda x: give_prefix(x, 'std_')
    )
    mins = values.min().reset_index().assign(
        column =  lambda x: give_prefix(x, 'min_')
    )            
    maxs = values.max().reset_index().assign(
        column =  lambda x: give_prefix(x, 'max_')
    )
    counts = values.size().reset_index().assign(
        column =  lambda x: give_prefix(x, 'count_')
    )
    
        
    summary = pd.concat(
        [firsts, lasts, means, stds, mins, maxs, counts]
    )
    
    ret = summary.pivot(
        index='Encounter',
        columns='column',
        values='value_as_number'
    )
    
    return(ret.reset_index('Encounter'))


# ### Categorical Measurements

# In[ ]:


categorical_measurements = (
     measurements
     .merge(encounters, on='visit_occurrence_id')
     .query(
         '''
         value_as_number != value_as_number \
         and value_as_concept_id != 0.0
         '''
     )
)


# In[ ]:


enc_cat_meas = get_categorical_counts(categorical_measurements)


# In[ ]:


cat_meas_last = (
    categorical_measurements
    .sort_values([
        'Encounter', 
        'measurement_concept_id',
        'measurement_datetime'
    ])
    .drop_duplicates(
        ['Encounter', 'measurement_concept_id'],
        keep='last'
    )
)

enc_cat_meas_last = get_categorical_counts(cat_meas_last, prefix='last_')


# In[ ]:


enc_cat_meas = (
    enc_cat_meas
    .merge(
        enc_cat_meas_last,
        on='Encounter'
    )
)


# ### Continuous Measurements

# In[ ]:


continuous_measurements = (
    measurements
    .merge(encounters, on='visit_occurrence_id')
    .query("value_as_number == value_as_number")
    .sort_values([
        'Encounter', 
        'measurement_concept_id',
        'measurement_datetime'
    ])
)


# In[ ]:


enc_cont_meas = get_continuous_stats(continuous_measurements)


# ### Observations

# In[ ]:


last_obs = (
    observations
    .merge(
        encounters, 
        on='person_id'
    )
    .query(
        '''
        @pd.to_datetime(observation_datetime).dt.tz_localize(None) 
        <= @pd.to_datetime(hospital_discharge_time)
        '''
        .replace('\n', '')
    )
    .sort_values([
        'Encounter', 'observation_concept_id', 'observation_datetime'
    ])
    .drop_duplicates(
        ['Encounter', 'observation_concept_id'],
        keep='last'
    )
)


# In[ ]:


enc_cat_obs = get_categorical_counts(
    last_obs.query(         
        '''
        value_as_number != value_as_number \
        and value_as_concept_id != 0.0
        '''
    ),
    prefix='last_', 
    meas=False
)


# In[ ]:


enc_cont_obs = (
    last_obs.query("value_as_number == value_as_number")
    .assign(
        column = lambda x: "last_" + x['observation_concept_id'].astype(str)
    )
    .pivot(
        index='Encounter',
        columns='column',
        values='value_as_number'
    )
    .reset_index('Encounter')
)


# ### Demographics

# In[ ]:


dm['age'] = datetime.datetime.now().year - dm['year_of_birth']
races = pd.get_dummies(dm.race_concept_id)
ethnicities = pd.get_dummies(dm.ethnicity_concept_id)
genders = pd.get_dummies(dm.gender_concept_id)


# In[ ]:


dm_features = pd.concat(
        [dm.get(["person_id", "age"]), races, ethnicities, genders],
        axis=1
)


# In[ ]:


enc_info = (
    aki_cohort
    .drop_duplicates('Encounter', keep='last')
    .assign(
        length_of_stay = lambda x: (
            (
                pd.to_datetime(x['hospital_discharge_time'])
                - pd.to_datetime(x['hospital_admission_time'])
            )
            .dt
            .total_seconds()
            /3600
        ),
        
        num_units = (
            aki_cohort.groupby('Encounter').person_id.size().values
        ),
        
        num_previous_encounters = (
            aki_cohort.drop_duplicates('Encounter').groupby('person_id').cumcount().values
        )    
    )
    .eval("readmission = readmission*(visit_occurrence_id not in @planned_visits.visit_occurrence_id)")
    .get([
        'Encounter',
        'person_id',
        'length_of_stay', 
        'num_units', 
        'num_previous_encounters',
        'careunit',
        'Hospital_admission',
        'KDIGO',
        'readmission'
    ])
)

enc_info = (
    pd.concat(
        [enc_info, pd.get_dummies(enc_info.careunit)],
        axis=1
    )
    .drop('careunit', axis=1)
)


# ### Procedures

# In[ ]:


enc_proc = (
    procedures
    .merge(encounters, on='visit_occurrence_id')
    .groupby(['Encounter', 'procedure_concept_id'])
    .size()
    .reset_index()
    .rename(columns={0: 'count'})
    .pivot(
        index='Encounter',
        columns='procedure_concept_id',
        values='count'
    )
)


# ### Conditions

# In[ ]:


enc_conds = (
    conditions
    .merge(encounters, on='visit_occurrence_id')
    .groupby(['Encounter', 'condition_concept_id'])
    .size()
    .reset_index()
    .rename(columns={0: 'count'})
    .pivot(
        index='Encounter',
        columns='condition_concept_id',
        values='count'
    )
)


# ### Concatenate all features

# In[ ]:


all_continuous_features = (
    enc_info.get(['Encounter'])
    .merge(enc_cont_obs, on='Encounter', how='left')
    .merge(enc_cont_meas, on='Encounter', how='left')
    .merge(enc_cat_meas, on='Encounter', how='left')
)

all_continuous_features.dropna(axis=1, how='all', inplace=True)


all_continuous_features.fillna(
    all_continuous_features.mean(),
    inplace=True
)

# Do not normalize the encounter number
all_continuous_features.iloc[:, 1:] = (
    (all_continuous_features.iloc[:, 1:] - all_continuous_features.iloc[:, 1:].mean())
    / all_continuous_features.iloc[:, 1:].std()
)

all_categorical_features = (
    enc_info.get(['Encounter'])
    .merge(enc_cat_obs, on='Encounter', how='left')
    .merge(enc_proc, on='Encounter', how='left')  
    .merge(enc_conds, on='Encounter', how='left')   
)

all_categorical_features.fillna(0, inplace=True)

all_features = (
    enc_info
    .merge(dm_features, on='person_id')
    .merge(all_continuous_features, on='Encounter', how='left')
    .merge(all_categorical_features, on='Encounter', how='left')
)

def normalize_features(df, cols):
    for col in cols:
        df[col] = (df[col] - df[col].mean())/df[col].std()
        
    return(df)

all_features = normalize_features(
    all_features,
    ['age', 'length_of_stay', 'num_units']
)

all_features.dropna(axis=1, how='all', inplace=True)


# In[ ]:


'''
Preserve intermediate datasset
'''
outfile = 'all_features.csv'
all_features.to_csv(outfile, index=False)

args = ["gsutil", "cp", outfile, f"{my_bucket}/data/{outfile}"]
output = subprocess.run(args, capture_output=True)

print(output.stderr)


# ## Perform Feature Selection

# In[ ]:


'''
Run a logistic regression model using each variable.
If the variable is significantly correlated, then preserve.
'''
X = all_features.drop(['Encounter', 'person_id', 'readmission'], axis=1)
y = all_features.readmission


# In[ ]:


keep_features = []
for feature in X.columns:
    try:
        logit_fit = sm.Logit(y, X[[feature]]).fit(disp=False)
        
        if logit_fit.pvalues[0] <= 0.001 and not pd.isnull(logit_fit.pvalues[0]):   
            keep_features.append(feature)
    except:
        continue
    


# In[ ]:


'''
Remove features that are highly correlated
'''

corr_matrix = X[keep_features].corr().abs()


# In[ ]:


drop_columns = []
for i, row in enumerate(corr_matrix.values):
    for j, cov in enumerate(row):
        if j == i:
            break
        elif cov >= 0.95 and corr_matrix.columns[i] not in drop_columns:
            drop_columns.append(corr_matrix.columns[j])
        


# In[ ]:


X_final = X[keep_features].drop(drop_columns, axis=1)


# In[ ]:


'''
Use Random Forest for Feature Selection
'''

X_train, X_test, y_train, y_test = train_test_split(
    X_final2, y, test_size = 0.2
)


# In[ ]:


selector = SelectFromModel(estimator=RandomForestClassifier()).fit(X_train, y_train)


# In[ ]:


X_final2 = X_final.iloc[:, selector.get_support()]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
    X_final2 , y, test_size = 0.2
)


# In[ ]:


logit_fit = LogisticRegression().fit(
    X_train, y_train.values.ravel()
)


# In[ ]:



x = list(np.sort(selector.estimator_.feature_importances_))


# In[ ]:


import numpy as np
fi = selector.estimator_.feature_importances_
top_150 = np.sort(fi)[len(fi)-:]


# In[ ]:


X_train.columns[s2]


# In[ ]:


print("Top 10 features:", top_150[0:10])


# In[ ]:


y_predict_logit = logit_fit.predict_proba(X_test)
y_predict_rf = RandomForestClassifier().predict_proba(X_test)


# In[ ]:


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras import regularizers

import tensorflow as tf

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, roc_auc_score, roc_curve
)
from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


X = pd.read_csv('X.csv')
y= pd.read_csv('y.csv')


# In[ ]:


sfs = SequentialFeatureSelector(LogisticRegression()).fit(X_final, y)


# In[ ]:


'''
Neural Network
'''

rr = 0.1
dr = 0.30

model = Sequential()
model.add(
    Dense(
        100, 
        input_shape=[X_train.shape[1]], 
        activation='relu',
        kernel_regularizer=regularizers.l2(rr)
    )
)

model.add(Dropout(dr))

model.add(
    Dense(
        1000, 
        input_shape=[X_train.shape[1]], 
        activation='relu',
        kernel_regularizer=regularizers.l2(rr)
    )
)

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train.values, y_train.values, epochs=80)


# In[ ]:


X_train


# In[ ]:


y_predict_nn = model.predict(X_test)


# In[ ]:


roc_auc_score(y_test, y_predict_nn)


# In[ ]:


fpr, tpr, threshold = roc_curve(y_test, y_predict_nn)


# In[ ]:


fpr2, tpr2, threshold2 = roc_curve(y_test, y_predict2[:,1])


# In[ ]:


fpr3, tpr3, threshold3 = roc_curve(y_test, y_predict_logit[:,1])


# In[ ]:


roc_auc_score(y_test, y_predict[:,1])


# In[ ]:


model.summary()


# ## Generate Outputs and Export Dataset

# In[ ]:


X_final2.to_csv("X.csv", index=False)


# In[ ]:


args = ["gsutil", "cp", "X.csv", f"{my_bucket}/data/X.csv"]
output = subprocess.run(args, capture_output=True)

print(output.stderr)


# In[ ]:


y.to_csv("y.csv", index=False)

args = ["gsutil", "cp", "y.csv", f"{my_bucket}/data/y.csv"]
output = subprocess.run(args, capture_output=True)

print(output.stderr)


# In[ ]:


'''
Show the covariance matrix
'''

sns.heatmap(corr_matrix)


# In[ ]:


'''
Plot ROC
'''

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'g', label = 'Neural Net AUC = %0.2f' % roc_auc_score(y_test, y_predict))
plt.plot(fpr2, tpr2, 'b', label = 'Random Forest AUC = %0.2f' % roc_auc_score(y_test, y_predict2[:,1]))
plt.plot(fpr3, tpr3, 'r', label = 'Logistic Regression AUC = %0.2f' % roc_auc_score(y_test, y_predict3[:,1]))
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

