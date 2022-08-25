import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import statsmodels.formula.api as smf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.pipeline import make_pipeline
from sklearn import metrics

#1. Load dataset
#df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv')
df_raw = pd.read_csv('../data/raw/medical_insurance_cost.csv')
#2. Transform 
df_transformed=df_raw.copy()
columns_names = df_transformed.columns.values
df_transformed.drop_duplicates(subset=columns_names.tolist(), keep='last').shape
df_transformed = pd.get_dummies(df_raw, drop_first=True)
#3. Split
df=df_transformed.copy()
X = df.drop(['charges'], axis=1)
y= df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=15)
df_train=pd.concat([X_train, y_train], axis=1)
num_2=['age', 'children', 'bmi', 'charges']
def normalize(b, z):
    return (z-df_train[b].min())/(df_train[b].max()-df_train[b].min())
for i in num_2:
    df_train[f"{i}_N"] = df_train.apply(lambda x: normalize(i, x[i]), axis=1)
    df_train = df_train.drop([f"{i}"], axis=1)
#Model3
def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree),
                         LinearRegression(**kwargs))
param_grid = {'polynomialfeatures__degree': np.arange(4), # polynomial (0,1,2,3)
              'linearregression__fit_intercept': [True, False], # with and without intercept
              'linearregression__normalize': [True, False]} # normalize and not normalize

grid = GridSearchCV(PolynomialRegression(), param_grid) # 5 folds
grid.fit(X_train, y_train)
model3 = grid.best_estimator_
#save model to disk
filename='../models/final_model.sav'
pickle.dump(model3, open(filename, 'wb'))