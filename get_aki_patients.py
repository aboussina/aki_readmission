#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
Program: get_aki_patients.ipynb

Purpose: 
Build initial cohort of AKI patients and
label readmission encounters

Author: Aaron Boussina
'''


# In[2]:


import subprocess
import os
import pandas as pd


# In[4]:


my_bucket = os.getenv('WORKSPACE_BUCKET')


print(subprocess.check_output(f"gsutil ls -r {my_bucket}", shell=True).decode('utf-8'))


# In[5]:


'''
Initialize Constants
'''
my_bucket = os.getenv('WORKSPACE_BUCKET')
dataset = os.getenv('WORKSPACE_CDR')

aki_concept_ids = tuple([
    4126305, # Acute renal impairment
    197320, # Acute renal failure syndrome  
    3661748, # Acute kidney injury due to disease caused by Severe acute respiratory syndrome coronavirus 2
    36716182, # Acute kidney injury due to circulatory failure
    36716183, # Acute kidney injury due to hypovolemia
    36716312, # Acute kidney injury due to sepsis
    37111531, # Acute kidney injury due to trauma
    37116430, # Acute kidney failure stage 1
    37116431, # Acute kidney failure stage 2
    37116432, # Acute kidney failure stage 3
    37395514, # Acute kidney injury due to acute tubular necrosis due to hypovolaemia
    37395516, # Acute kidney injury due to acute tubular necrosis due to circulatory failure
    37395517, # Acute kidney injury due to acute tubular necrosis due to sepsis
    37395518, # Acute kidney injury due to acute tubular necrosis with histological evidence
    37395519, # Acute kidney injury due to acute tubular necrosis due to hypovolaemia with histological evidence
    37395520, # Acute kidney injury due to acute tubular necrosis due to sepsis with histological evidence
    37395521, # Acute kidney injury due to acute tubular necrosis due to circulatory failure with histological evidence
    44809061, # Acute kidney injury stage 1
    44809062, # Acute kidney injury stage 2
    44809063, # Acute kidney injury stage 3
    761083, # Acute injury of kidney
    45757442, # Acute nontraumatic kidney injury
    4232873, # Acute postoperative renal failure
    37016366, # Acute renal failure caused by contrast agent
    197329, # Acute renal failure due to acute cortical necrosis
    4151112, # Acute renal failure due to crush syndrome
    4311129, # Acute renal failure due to ischemia
    44808744, # Acute renal failure due to non-traumatic rhabdomyolysis
    4180453, # Acute renal failure due to obstruction
    4264681, # Acute renal failure due to procedure
    44808128, # Acute renal failure due to traumatic rhabdomyolysis
    196490, # Acute renal failure following labor AND/OR delivery
    4215648, # Acute renal failure following molar AND/OR ectopic pregnancy
    45757398, # Acute renal failure on dialysis
    4160274, # Acute renal failure with oliguria
    432961, # Acute renal papillary necrosis with renal failure
    4128067, # Acute-on-chronic renal failure
    4126424, # Nephrotoxic acute renal failure
    4200639, # Post-renal renal failure
    4139414, # Transient acute renal failure
])


# In[6]:


for infile in ["All_Adults_df.csv"]:
    args = ["gsutil", "cp", f"{my_bucket}/data/{infile}", infile]
    output = subprocess.run(args, capture_output=True)
    print(output.stderr)


# In[7]:


'''
Import the all_patients dataset.  
This contains information on all inpatient encounters.
'''
all_patients = (
    pd.read_csv("./All_Adults_df.csv")
    .eval(
        '''
        hospital_admission_time = @pd.to_datetime(hospital_admission_time)
        hospital_discharge_time = @pd.to_datetime(hospital_discharge_time)
        '''
    )
)


# In[20]:


person_ids = tuple(aki_cohort.person_id.unique())


# In[101]:


'''
Query AKI condition occurrences
'''

aki_query = f'''
SELECT
    *
FROM
    {dataset}.condition_occurrence co
WHERE
    person_id IN {person_ids}
    AND
    condition_concept_id IN {aki_concept_ids}
'''


# In[102]:


aki_conditions = pd.read_gbq(aki_query, dialect='standard')


# In[18]:


aki_visits = aki_cohort.drop_duplicates('visit_occurrence_id').get(['Encounter', 'visit_occurrence_id', 'KDIGO'])


# In[104]:


'''
Identify the encounters where AKI was diagnosed 
'''
aki_visits = (
    all_patients
    .merge(aki_conditions, on='person_id')
    .eval(
        '''
        condition_start_datetime = @pd.to_datetime(condition_start_datetime).dt.tz_localize(None)
        '''
    )
    .query(
        '''
        visit_occurrence_id_x == visit_occurrence_id_y
        or (
            condition_start_datetime >= hospital_admission_time
            and condition_start_datetime <= hospital_discharge_time
        )
        '''
        .replace('\n', '')
    )
    .get(['Encounter'])
    .drop_duplicates('Encounter')
)


# In[14]:


'''
Identify instances of readmission
'''
readmissions = (
    all_patients
    .sort_values(
        ['person_id', 'Encounter'], 
        ascending=[True, False]
    )
    .query(
        '''
        Encounter != Encounter.shift()
        and person_id == person_id.shift()
        and 30 >= (
            (hospital_admission_time.shift() - hospital_admission_time)
            .dt
            .days
        )
        '''
        .replace('\n', '')
    )
    .get(['Encounter'])
    .eval('readmission = 1')
)


# In[19]:


'''
Create the AKI cohort 
'''
aki_cohort = (
    all_patients
    .merge(readmissions, on='Encounter', how='left')
    .merge(aki_visits, on='Encounter')
    .fillna({'readmission': 0})
)


# In[3]:


aki_cohort = pd.read_csv('aki_cohort.csv')


# In[21]:


'''
Remove encounters where the patient died.
These are irrelevant for prediction of readmission.
'''
death_query = f'''
    SELECT
        person_id,
        death_datetime
    FROM
        {dataset}.death
    WHERE
        person_id IN {person_ids}
'''


# In[22]:


deaths = pd.read_gbq(death_query, dialect='standard')


# In[23]:


death_visits = (
    aki_cohort
    .merge(deaths, on='person_id')
    .eval(
        '''
        death_datetime = @pd.to_datetime(death_datetime).dt.tz_localize(None)
        '''
    )
    .query(
        '''
        death_datetime >= hospital_admission_time
        and death_datetime <= hospital_discharge_time
        '''
        .replace('\n', '')
    )
    .get(['Encounter'])
    .drop_duplicates('Encounter')
)


# In[24]:


aki_cohort = (
    aki_cohort
    .query('Encounter not in @death_visits.Encounter')
)


# In[27]:


aki_cohort = aki_cohort.drop('visit_occurrence_id_x', axis=1).rename({'visit_occurrence_id_y': 'visit_occurrence_id'}, axis=1)


# In[28]:


aki_cohort


# In[29]:


outfile = 'aki_cohort.csv'
aki_cohort.to_csv(outfile, index=False)

args = ["gsutil", "cp", outfile, f"{my_bucket}/data/{outfile}"]
output = subprocess.run(args, capture_output=True)
output.stderr

