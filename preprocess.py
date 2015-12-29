#!/usr/local/bin/python3.5

import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse

import pandas as pd
import cPickle as pickle


df = pd.read_csv('./../dataset/dataset_diabetes/diabetic_data.csv', sep=',',
                 na_values='?', low_memory=False)
df.replace('No', 0, inplace=True)
df.replace('Yes', 1, inplace=True)
df.readmitted.replace(['NO', '<30'], 0, inplace=True)
df.readmitted.replace('>30', 1, inplace=True)

#remove some columns
df.drop(['encounter_id', 'weight', 'payer_code', u'medical_specialty', u'patient_nbr'], axis=1, inplace=True)

columns_names = [u'race', u'gender', u'admission_type_id', u'discharge_disposition_id',
                 u'admission_source_id', u'diag_1', u'diag_2', u'diag_3', u'max_glu_serum', u'A1Cresult', u'metformin', u'repaglinide',
                 u'nateglinide', u'chlorpropamide', u'glimepiride', u'acetohexamide',
                 u'glipizide', u'glyburide', u'tolbutamide', u'pioglitazone',
                 u'rosiglitazone', u'acarbose', u'miglitol', u'troglitazone',
                 u'tolazamide', u'examide', u'citoglipton', u'insulin',
                 u'glyburide-metformin', u'glipizide-metformin', u'glimepiride-pioglitazone',
                 u'metformin-rosiglitazone', u'metformin-pioglitazone', u'change', u'diabetesMed']


#maybe convert diagnosis to drop specific (250".8") ??
df = df.apply(preprocessing.LabelEncoder().fit_transform)
enc = OneHotEncoder(categorical_features='all', dtype='float', handle_unknown='error', n_values='auto', sparse=False)
encoded_data = enc.fit_transform(df[columns_names])


classes = df.readmitted


c = [u'race', u'gender', u'age',
     u'admission_type_id', u'discharge_disposition_id',
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

z = list(set(c) - set(columns_names))
df_not_transformed = df.as_matrix(columns=z)

scaler = preprocessing.StandardScaler().fit(df_not_transformed)
df_transformed = scaler.transform(df_not_transformed)

processed_array = np.concatenate((df_transformed, encoded_data), axis=1)
Asp = scipy.sparse.csr_matrix(processed_array)

with open('processed_hospital_sparse.dat', 'wb') as outfile:
    pickle.dump(Asp, outfile, pickle.HIGHEST_PROTOCOL)

print(df.size)
