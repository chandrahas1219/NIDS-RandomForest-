#importing modules
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline 
import warnings 
warnings.filterwarnings('ignore')

data_train = pd.read_csv('Train_data.csv') 
data_test pd.read_csv('Test_data.csv')

data_train.head(5) 
data_train.shape 
data_train.info() 
data_train.describe()

#Data profiling
data_train.info() 
data_train.isnull().sum()


#Column unique values
for col in data_train.columns: 
print(data_train[col].nunique())


#Data visualization
sns.countplot(data=data_train, x='class', palette='PRGn') sns.countplot(data=data_train, x="protocol_type", hue='class', palette='copper')

plt.figure(figsize=(20,10)) sns.countplot(data=data_train, x='flag', hue='class', palette='Pastel2')

plt.figure(figsize=(40,30)) 
sns.heatmap(data_train.corr(), annot=True)


#Data encoding
from sklearn import preprocessing

def encoding(df): 
for col in df.columns: 
if df[col].dtype == 'object': 
label_encoder = preprocessing.LabelEncoder() 
df[col] = label_encoder.fit_transform(df[col])

encoding(data_train)
data_train.head(2)

#Feature selection using mutual information
from sklearn.feature_selection import mutual_info_classif, SelectKBest

X = data_train.drop(['class'], axis=1) y = data_train['class']
mutual_info = mutual_info_classif(X, y) 

mutual_info = pd.Series(mutual_info) 
mutual_info.index = X.columns

selected_features = mutual_info.sort_values(ascending=False).head(25).index X = X[selected_features]

#Data splitting and scaling
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(X, y) 
sc = StandardScaler() 
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test)

#importing random forest and calculation metrics
import sklearn.linear_model as lm 
import sklearn.ensemble as ensemble 
import sklearn.naive_bayes as nb 
import xgboost as xgb 
import sklearn.metrics as metrics
random_forest = ensemble.RandomForestClassifier()
accuracy = metrics.accuracy_score 
classification_report = metrics.classification_report 
confusion_matrix_display = metrics.ConfusionMatrixDisplay 
precision_score = metrics.precision_score 
recall_score = metrics.recall_score 
f1_score = metrics.f1_score 
roc_auc_score = metrics.roc_auc_score 
roc_curve = metrics.roc_curve 
confusion_matrix = metrics.confusion_matrix


#Evaluation
def classalgo_test(x_train,x_test,y_train,y_test): #classification
      
    rfc=RandomForestClassifier()
    lr=LogisticRegression()
    
    algos = [rfc,lr]
    algo_names = ['RandomForestClassifier','LogisticRegression']
    Train_acc=[]
    Train_precsc=[]
    Train_fsc=[]
    Train_Recall=[]
    Test_acc=[]
    Test_precsc=[]
    Test_fsc=[]
    Test_Recall=[]
    Test_AUC=[]
    
    result = pd.DataFrame(index = algo_names)
    
    for algo in algos:
    
        algo.fit(x_train,y_train)
        y_train_pred = algo.predict(x_train)
        y_test_pred = algo.predict(x_test)
        Train_acc.append(accuracy_score(y_train,y_train_pred))
        Train_precsc.append(precision_score(y_train,y_train_pred))
        Train_fsc.append(f1_score(y_train,y_train_pred))
        Train_Recall.append(recall_score(y_train,y_train_pred,average='micro'))
        
        
        Test_acc.append(accuracy_score(y_test,y_test_pred))
        Test_precsc.append(precision_score(y_test,y_test_pred))
        Test_fsc.append(f1_score(y_test,y_test_pred))
        Test_Recall.append(recall_score(y_test,y_test_pred,average='micro'))
        Test_AUC.append(roc_auc_score(y_test,y_test_pred))
        
    
    result['Train_Accuracy Score'] = Train_acc
    result['Train_Precision Score'] = Train_precsc
    result['Train_F1Score']= Train_fsc
    result['Train_Recall']= Train_Recall    
    result['Test_Accuracy Score'] = Test_acc
    result['Test_Precision Score'] = Test_precsc
    result['Test_F1Score']= Test_fsc
    result['Test_Recall']= Test_Recall
    result['Accuracy_Score']= Test_AUC
        
    return result.sort_values('Test_Accuracy Score', ascending=False)

classalgo_test(X_train,X_test,y_train,y_test)
    
rf=RandomForestClassifier()
rf.fit(X_train,y_train)
rf_predict=rf.predict(X_test)
print(classification_report(y_test,rf_predict))
print(confusion_matrix(y_test,rf_predict))
