import pandas as pd
import numpy as np
import seaborn as sns
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix

df = pd.read_excel('xcel file.xlsx')
X = df.iloc[:, :-1].values
Y = df.iloc[:, 10].values
print X, Y
# changing column strings into numerical datatype (float)
ft = ColumnTransformer(transformers=[('one_hot_encoder',OneHotEncoder(categories='auto'),[8,9])], remainder='passthrough')
X = np.array(ft.fit_transform(X),dtype=  np.float)

# spliting the dataset into two
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)
print X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

# predicting test set result
rfc = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
rfc.fit(X_train, Y_train)
Ypred = rfc.predict(X_test)
print Ypred
# describing the performance of random forest on the dataset
cmtr = confusion_matrix(Y_test, Ypred)
print cmtr
# heatmap
seamap = sns.heatmap(cmtr, xticklabels=['1','0'], yticklabels=['1', '0'])
plt.show(seamap)



