#%%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
sns.set(style='white', context='notebook', palette='deep')
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV


# %%
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")

# %%
train.head(5)

# %%
train.info()

# %%
train_len=len(train)
dataset=pd.concat(objs=[train,test],axis=0).reset_index(drop=True)

# %%
train.describe()

# %%
dataset.isnull().sum()

# %%
dataset = pd.get_dummies(dataset, columns = ["Col_3"], drop_first=True)
dataset = pd.get_dummies(dataset, columns = ["Col_4"], drop_first=True)
dataset = pd.get_dummies(dataset, columns = ["Col_8"], drop_first=True)
dataset = pd.get_dummies(dataset, columns = ["Col_9"], drop_first=True)
dataset = pd.get_dummies(dataset, columns = ["Col_11"], drop_first=True)
dataset = pd.get_dummies(dataset, columns = ["Col_12"], drop_first=True)
dataset = pd.get_dummies(dataset, columns = ["Col_15"], drop_first=True)
dataset = pd.get_dummies(dataset, columns = ["Col_17"], drop_first=True)
dataset = pd.get_dummies(dataset, columns = ["Col_20"], drop_first=True)
dataset = pd.get_dummies(dataset, columns = ["Col_22"], drop_first=True)
dataset = pd.get_dummies(dataset, columns = ["Col_24"], drop_first=True)
dataset = pd.get_dummies(dataset, columns = ["Col_25"], drop_first=True)
dataset = pd.get_dummies(dataset, columns = ["Col_27"], drop_first=True)
dataset = pd.get_dummies(dataset, columns = ["Col_28"], drop_first=True)
dataset = pd.get_dummies(dataset, columns = ["Col_29"], drop_first=True)
dataset = pd.get_dummies(dataset, columns = ["Col_32"], drop_first=True)


#%%
dataset.drop(['Id'],axis=1,inplace=True)



# Modeling
#%% Separate train dataset and test dataset
tr=dataset[:train_len]
ts=dataset[train_len:]
ts.drop(labels=["Score"],axis = 1,inplace=True)

#%% Separate train features and label
tr["Score"]=tr["Score"].astype(int)
Y_train=tr["Score"]
X_train=tr.drop(labels = ["Score"],axis = 1)

#%% Modelig with TPOT
x_train, x_test, y_train, y_test = train_test_split(X_train,Y_train, train_size=0.75, test_size=0.25)

tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2)
tpot.fit(x_train, y_train)
print(tpot.score(x_test, y_test))
tpot.export('tpot_CarInsurance_pipeline.py')

#%% predicting with linearSVC, the best model from tpot
# with a final score of about 0.84813 in public board 
svm = LinearSVC(C=10.0, dual=False, loss="squared_hinge", penalty="l1", tol=0.1)
clf = CalibratedClassifierCV(svm) 
clf.fit(X_train, Y_train)
clf.score(X_train,Y_train) # 0.8937074829931972
y_proba = clf.predict_proba(ts)