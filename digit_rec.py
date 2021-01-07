import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

df_test = pd.read_csv('mnist_test.csv')
print(df_test.shape)
df_test.head()
x_test = df_test.iloc[:, 1 :]
y_test = df_test['7']
y_test.head()

df_train = pd.read_csv('train.csv')
print(df_train.shape)


# Checking datasets and spliting
print(df_train.isnull().sum())
print(df_test.isnull().sum())


from sklearn.ensemble import RandomForestClassifier
x_train = df_train.iloc[:, 1 :]
y_train = df_train['label']
clf = RandomForestClassifier(n_estimators=100)
clf.fit(x_train, y_train)

from sklearn.metrics import classification_report
y_predicted = clf.predict(x_test)
y_predicted

from sklearn.metrics import accuracy_score
print(classification_report(y_test,y_predicted))
print(f"Accuracy is : {accuracy_score(y_test, y_predicted)}")

'''
import random
digit = random.randint(1, 999)
image = x_test.iloc[digit, :]
image = np.array(image, dtype='float')
pixels = image.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()
print(y_predicted[digit])
'''


#from joblib import dump, load
#dump(clf, 'digitRecModel.joblib')