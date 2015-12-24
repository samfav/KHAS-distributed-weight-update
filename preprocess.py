#!/usr/local/bin/python3.5

import numpy as np
from sklearn import preprocessing
import pandas as pd


df = pd.read_csv('./../dataset/dataset_diabetes/diabetic_data.csv', sep=',',
                 na_values='?', low_memory=False)
df.replace('No', 0, inplace=True)
df.replace('Yes', 1, inplace=True)
df.readmitted.replace(['NO', '<30'], 0, inplace=True)
df.readmitted.replace('>30', 1, inplace=True)

#nearest neighbours :
n_neighbors = 100

columns_names = [u'encounter_id', u'patient_nbr', u'race', u'gender', u'age',
                 u'weight', u'admission_type_id', u'discharge_disposition_id',
                 u'admission_source_id', u'time_in_hospital', u'payer_code',
                 u'medical_specialty', u'num_lab_procedures', u'num_procedures',
                 u'num_medications', u'number_outpatient', u'number_emergency',
                 u'number_inpatient', u'diag_1', u'diag_2', u'diag_3', u'number_diagnoses',
                 u'max_glu_serum', u'A1Cresult', u'metformin', u'repaglinide',
                 u'nateglinide', u'chlorpropamide', u'glimepiride', u'acetohexamide',
                 u'glipizide', u'glyburide', u'tolbutamide', u'pioglitazone',
                 u'rosiglitazone', u'acarbose', u'miglitol', u'troglitazone',
                 u'tolazamide', u'examide', u'citoglipton', u'insulin',
                 u'glyburide-metformin', u'glipizide-metformin', u'glimepiride-pioglitazone',
                 u'metformin-rosiglitazone', u'metformin-pioglitazone', u'change', u'diabetesMed']
print(df.head())
