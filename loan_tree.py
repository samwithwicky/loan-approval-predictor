import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

#Read dataset
df = pd.read_csv("loan_approval_dataset.csv")
#print(df.head())

#EDA + preprocessing
df[' education'] = df[' education'].map({' Graduate':1, ' Not Graduate': 0})
df[' self_employed'] = df[' self_employed'].map({' Yes':1, ' No': 0})
df[' loan_status'] = df[' loan_status'].map({' Approved':1, ' Rejected': 0})
sns.heatmap(df.corr(),annot= True)
#plt.show()
df.drop(columns= [' no_of_dependents',' education',' self_employed',' loan_term'],inplace= True)# no or low correlation with other features

#Train test split
X = df.drop(' loan_status', axis =1)
y = df[' loan_status']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
model =  DecisionTreeClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

#Accuracy metrics
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Matrix:",confusion_matrix(y_test,y_pred))
print("f1 score:", f1_score(y_test,y_pred))
