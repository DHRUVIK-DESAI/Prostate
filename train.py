#https://www.kaggle.com/datasets/sajidsaifi/prostate-cancer/data
import pandas as pd
df=pd.read_csv("Prostate_Cancer.csv")
print(df.isnull().sum())
df=df.dropna()
print(df.isnull().sum())
print(df.dtypes)
df=df.drop(["id"], axis=1)
category_colums=["diagnosis_result"]
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df[category_colums] = df[category_colums].apply(encoder.fit_transform)
X=df.iloc[:,1:10]
y=df.iloc[:,0]
X=X.to_numpy()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)


import warnings
warnings.filterwarnings("ignore")
names = ["K-Nearest Neighbors", "SVM",
          "Decision Tree", "Random Forest",
          "Naive Bayes","ExtraTreesClassifier","VotingClassifier"]
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import VotingClassifier

classifiers = [
    KNeighborsClassifier(),
    LinearSVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GaussianNB(),
    ExtraTreesClassifier(),
    VotingClassifier(estimators=[('DT', DecisionTreeClassifier()), ('rf', RandomForestClassifier()), ('et', ExtraTreesClassifier())], voting='hard')]

clfF=[]
for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    print(name)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print('--------------------------------------------------------------')
    clfF.append(clf)

import pickle
import bz2
sfile = bz2.BZ2File("model.pkl", 'wb')
pickle.dump(clfF, sfile)  
pickle.dump(encoder, open("encoder.pkl",'wb'))    
    
    
    
    
    
    
    
    
    
    
















