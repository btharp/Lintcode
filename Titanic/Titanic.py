#%%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

tr0=pd.read_csv('train.csv')

# %%

#tr0[tr0['Sex']=='male']=1
#tr0[tr0['Sex']=='female']=0
Sex=pd.get_dummies(tr0['Sex'],drop_first=True)
Embarked=pd.get_dummies(tr0['Embarked'],drop_first=True)
Fare=tr0['Fare']
NFare=(Fare-Fare.min())/(Fare.max()-Fare.min())
Pclass=pd.get_dummies(tr0['Pclass'],drop_first=True)
SibSp=tr0['SibSp']
Parch=tr0['Parch']
Age=tr0['Age']
Age.fillna(int(Age.mode()),inplace=True)
NAge=(Age-Age.min())/(Age.max()-Age.min())

X=pd.concat([Sex,Embarked,Fare,Pclass,SibSp,Parch,NAge],axis=1)
y=tr0['Survived']


# %%
ts0=pd.read_csv('test.csv')

Sex1=pd.get_dummies(ts0['Sex'],drop_first=True)
Embarked1=pd.get_dummies(ts0['Embarked'],drop_first=True)
Fare1=ts0['Fare']
NFare=(Fare1-Fare1.min())/(Fare1.max()-Fare1.min())
Fare1.fillna(int(Fare.mode()),inplace=True)
Pclass1=pd.get_dummies(ts0['Pclass'],drop_first=True)
SibSp1=ts0['SibSp']
Parch1=ts0['Parch']
Age1=ts0['Age']
Age1.fillna(int(Age.mode()),inplace=True)
NAge=(Age1-Age1.min())/(Age1.max()-Age1.min())


X1=pd.concat([Sex1,Embarked1,Fare1,Pclass1,SibSp1,Parch1,Age1],axis=1)




#%%
clf=LogisticRegression(random_state=0).fit(X,y)
clf.score(X,y)

clf_tree=DecisionTreeClassifier(random_state=0).fit(X,y)
clf_tree.score(X,y)

clf_svm=LinearSVC(random_state=0, tol=1e-5).fit(X,y)
clf_svm.score(X,y)

clf_random=RandomForestClassifier(max_depth=10, random_state=0).fit(X,y)
clf_random.score(X,y)

clf_ada = AdaBoostClassifier(random_state=0,n_estimators=100).fit(X,y)
clf_ada.score(X,y)






# %%
y1=clf_ada.predict(X1)
gender=pd.read_csv('gender_submission.csv')
gender['Survived']=y1

#%%
gender.to_csv('submission.csv',index=None)

# %%
