import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer


# importing the dataset and Extracting the Independent
file_path = ("C:/Users/user/Documents/MyTest/1000_Companies.csv")
companies = pd.read_csv (file_path)
X = companies.iloc[:,:-1].values
y = companies.iloc[:, 4].values
print(companies.head())

sns.heatmap(companies.corr())
plt.show()

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])

ct = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder = 'passthrough')
X = ct.fit_transform(X)
X= X[:, 1:]

#Spliting data set into train and test set...
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 0)

#fitting multiple linear regression model 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the test set results
y_pred = regressor.predict(X_test)
print(y_pred)

# calculation coefficients and intercept
print(regressor.coef_)
print(regressor.intercept_)

#claculation the r squared value
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))
