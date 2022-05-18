import pandas as pd

df=pd.read_csv("2021VAERSDATA_CLEANED.csv")

from sklearn.ensemble import RandomForestClassifier 
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

train,test = train_test_split(df,test_size = 0.2)
clf = RandomForestClassifier()
X_var = ['AGE_YRS', 'SEX', 'L_THREAT', 'ER_VISIT', 'HOSPDAYS', 'X_STAY',
       'DISABLE', 'V_ADMINBY', 'CUR_ILL', 'PRIOR_VAX', 'FORM_VERS',
       'BIRTH_DEFECT', 'OFC_VISIT', 'ER_ED_VISIT', 'ALLERGIES', 'OM',
       'Diabetes', 'Obesity', 'Asthma', 'Hypertension', 'Coronary Art Dis',
       'High BP', 'Other Comoridities']


clf.fit(train[X_var],train['AE'])

from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix, recall_score


accuracy_score(train['AE'],clf.predict(train[X_var]))   #accuracy of training data

accuracy_score(test['AE'],clf.predict(test[X_var]))     # accuracy for testing data (overfitting)

confusion_matrix(train['AE'],clf.predict(train[X_var])) #confusion matrix for training data
confusion_matrix(test['AE'],clf.predict(test[X_var]))   # confusion matrix for testing data

#recall score
recall_score(train['AE'], clf.predict(train[X_var])) 
#0.6115965774622246
recall_score(test['AE'], clf.predict(test[X_var]))
#0.36485023457235655
# overfitting - needs to be tuned to be fixed 
# without tuning - the best model for prediction seems to be naive bayes (between these two)
