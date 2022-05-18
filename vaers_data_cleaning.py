#from pandas_profiling import ProfileReport
pip install pandas
import pandas as pd

df = pd.read_csv("2021VAERSDATA.csv") #encoding = "ISO-8859-1")
#profile = ProfileReport(df, title="VAERS profiling")
#profile.to_file("VAERS Profile.html")
#remove variables with high correlation, high missing values 
# create y variable (merging symptoms and dead column)
# derive variables with high cardinality (no of different values)

# bag of words

list_of_words=["severe", "rapid", "anaphylaxis", "anaphylactic", "Anaphylactic", "tachychardia", "heart", "nerve", "nervous", "spine", "axillary lymphadenopathy", 
               "vomiting", "diarrhea", "death", "dead", 
               "died", "blood clots", "clotting", "chest pain", "swelling", "tightness", "palpitations", "allergic", "erythematous", "Bell's Palsy", 
               "parasthesia", "deceased", "thyroiditis", "erythema", "Vertigo", "vertigo", ]

# removing variables with high missing values and correlation
# dropping unrequired columns first 
df.drop("VAERS_ID", axis=1, inplace=True)
df.drop("RECVDATE", axis=1, inplace=True)
df.drop("CAGE_MO", axis=1, inplace=True)
df.drop("RPT_DATE", axis=1, inplace=True)
df.drop("DATEDIED", axis=1, inplace=True)
df.drop("RECOVD", axis=1, inplace=True)
df.drop("ONSET_DATE", axis=1, inplace=True)
df.drop("NUMDAYS", axis=1, inplace=True)
df.drop("LAB_DATA", axis=1, inplace=True)
df.drop("SPLTTYPE", axis=1, inplace=True)
df.drop("TODAYS_DATE", axis=1, inplace=True)
#removing variables with large missing values/correlation
df.drop("STATE", axis=1, inplace=True)

#merging columns that require merging 


import numpy as np
# joining Symptoms_text and Died into "AE" (Adverse Event)
df['AE'] = np.where((df['DIED']=='Y')|(df['SYMPTOM_TEXT'].str.contains('|'.join(list_of_words),case=False)),1,0)

df.drop(["DIED", "SYMPTOM_TEXT"], axis=1, inplace=True) #deleting symptoms_text and died columns
df.drop("HOSPITAL", axis=1, inplace=True) #deleting hospitalized column
df['HOSPDAYS'].fillna(0,inplace=True) #to fill blank cells with 0
df.drop("VAX_DATE", axis=1, inplace=True) #deleting vax_date column

# create list of unknowns and convert other_meds column into a dummy variable column with 0 and 1
list_of_unknowns = ["none", "unkown", "no", "na", "n/a", "ukn"]
df['OM'] = np.where((df['OTHER_MEDS'].str.contains('|'.join(list_of_unknowns),case=False))|(df['OTHER_MEDS'].isna()),0,1)            
df.drop("OTHER_MEDS", axis=1, inplace=True)        # drop other_meds column

df['Diabetes'] = np.where(df['HISTORY'].str.contains('diabetes',case=False),1,0)
df['Obesity'] = np.where(df['HISTORY'].str.contains('obesity|obese', case=False),1,0)
df['Asthma'] = np.where(df['HISTORY'].str.contains('asthma', case=False),1,0)
df['Hypertension'] = np.where(df['HISTORY'].str.contains('hypertension', case=False),1,0)
df['Coronary Art Dis'] = np.where(df['HISTORY'].str.contains('coronary', case=False),1,0)
df['High BP'] = np.where(df['HISTORY'].str.contains('pressure|bp', case=False),1,0)
disease = ['diabetes','obesity','obese', 'asthma', 'hypertension', 'pressure', 'coronary'
           , 'bp', 'none', 'no', 'NA', 'n/a', 'ukn', 'unknown' ]
df['Other Comoridities'] = np.where((df['HISTORY'].str.contains('|'.join(disease),case=False))|(df['HISTORY'].isna()),0,1)

#creating columns for adverse effects
df['anaphylaxis'] = np.where(df['SYMPTOM_TEXT'].str.contains('anaphylaxis', 'Anaphylactic', 'anaphylactic', case=False),1,0)
df['axillary lymphadenopathy'] = np.where(df['SYMPTOM_TEXT'].str.contains('axillary lymphadenopathy', 'Axillary lymphadenopathy', case=False),1,0)
df["Bell's Pallsy"] = np.where(df['SYMPTOM_TEXT'].str.contains("Bell's Palsy", "bell's pallsy", "Bell's pallsy", case=False),1,0)
df['thyroiditis'] = np.where(df['SYMPTOM_TEXT'].str.contains('thyroiditis', 'Thyroiditis', case=False),1,0)
df['blood clots'] = np.where(df['SYMPTOM_TEXT'].str.contains('Blood clots', 'blood clots', 'Blood Clots', 'clotting', 'Clotting', case=False),1,0)
df['erythema'] = np.where(df['SYMPTOM_TEXT'].str.contains('Erythema', 'erythema', 'erythematous', 'Erythematous', case=False),1,0)
df['Vertigo'] = np.where(df['SYMPTOM_TEXT'].str.contains('Vertigo', 'vertigo', case=False),1,0)
df['parasthesia'] = np.where(df['SYMPTOM_TEXT'].str.contains('parasthesia', 'Parasthesia', case=False),1,0)
df['tachychardia'] = np.where(df['SYMPTOM_TEXT'].str.contains('tachychardia', 'Tachychardia', case=False),1,0)
df['palpitations'] = np.where(df['SYMPTOM_TEXT'].str.contains('palpitations', 'Palpitations', case=False),1,0)
df['chest pain'] = np.where(df['SYMPTOM_TEXT'].str.contains('Chest pain', 'chest pain', case=False),1,0)


df.drop("HISTORY", axis=1, inplace=True)
df['ER_VISIT'] = np.where(df['ER_VISIT']=='Y',1,0) #dummy variables for ER_VISIT  
df['X_STAY'] = np.where(df['X_STAY']=='Y',1,0) #creating dummy variables for X_Stay
df['DISABLE'] = np.where(df['DISABLE']=='Y',1,0) #creating dummy variables for Disability
df.drop("V_FUNDBY", axis=1, inplace=True) #delelting because 99.8% missing values and known values are "unkown" (mostly)
# dummy variables: converting unkowns into 0 and knowns into 1
df['PRIOR_VAX'] = np.where((df["PRIOR_VAX"].str.contains('|'.join(list_of_unknowns),case=False))|(df['PRIOR_VAX'].isna()),0,1)
df['CUR_ILL'] = np.where((df["CUR_ILL"].str.contains('|'.join(list_of_unknowns),case=False))|(df['CUR_ILL'].isna()),0,1)

#dummies for V_ADMINBY
# convert everrything else iinto 0 and 1 (sex, office visit, er_ed_visit, birth_defect, allergies)
# allergies - yes/no

df['V_ADMINBY'] = np.where((df["V_ADMINBY"].str.contains('|'.join(list_of_unknowns),case=False))|(df['V_ADMINBY'].isna()),0,1) 
#dummies for adminby
df['SEX'] = np.where(df['SEX']=='M',1,0) # converting sex into 1 and 0
df['OFC_VISIT'] = np.where(df['OFC_VISIT']=='Y',1,0) # converting ofc visit into 1 and 0
df['ER_ED_VISIT'] = np.where(df['ER_ED_VISIT']=='Y',1,0) # convertin er_ed_visit into 1 and 0
df['BIRTH_DEFECT'] = np.where(df['BIRTH_DEFECT']=='Y',1,0) # converting birth_defect into 1 and 0
df['L_THREAT'] = np.where(df['L_THREAT']=='Y',1,0) #converting L_THREAT into 1 and 0
df['ALLERGIES'] = np.where((df["ALLERGIES"].str.contains('|'.join(list_of_unknowns),case=False))|(df['ALLERGIES'].isna()),0,1) 
#converting allergies to 1 and 0
df.drop("CAGE_YR", axis=1, inplace=True)

#manage missing ages - convert to age brackets? Or just add mean age to missing age values?

df['AGE_YRS'].fillna(df['AGE_YRS'].median(), inplace=True)


df.to_csv("./2021VAERSDATA_CLEANED1.csv",index=False)


            
            
            



