#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 12:29:03 2022

@author: christian
"""

import pandas as pd
fed_files = ["/Users/christian/Downloads/MORTGAGE30US.csv", "/Users/christian/Downloads/RRVRUSQ156N.csv", "/Users/christian/Downloads/CPIAUCSL.csv"]
dfs = [pd.read_csv(f, parse_dates = True, index_col = 0) for f in fed_files]
dfs[0]

fed_data = pd.concat(dfs, axis = 1)
fed_data

fed_data = fed_data.ffill().dropna()
fed_data

zillow_files = ["/Users/christian/Downloads/Metro_median_sale_price_uc_sfrcondo_week.csv", "/Users/christian/Downloads/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_month.csv"]
dfs = [pd.read_csv(f) for f in zillow_files]
dfs[0]
dfs[1]

dfs = [pd.DataFrame(df.iloc[0,5:]) for df in dfs]
for df in dfs:
    df.index = pd.to_datetime(df.index)
    df["month"] = df.index.to_period("M")
dfs[0]

price_data = dfs[0].merge(dfs[1], on = "month")
price_data.index = dfs[0].index
price_data

del price_data["month"]
price_data.columns = ["price", "value"]
price_data

fed_data.index = fed_data.index + pd.Timedelta(days = 2)
fed_data.tail(10)

price_data = fed_data.merge(price_data, left_index = True, right_index = True)
price_data

price_data.columns = ["interest", "vacancy", "cpi", "price", "value"]
price_data

price_data["adj_price"] = price_data["price"]/price_data["cpi"]*100
price_data["adj_value"] = price_data["value"]/price_data["cpi"]*100

price_data.plot.line(y = "price", use_index = True)

price_data.plot.line(y = "adj_price", use_index = True)

price_data["next_quarter"] = price_data["adj_price"].shift(-13)
price_data.dropna(inplace = True)
price_data

price_data["change"] = (price_data["next_quarter"] > price_data["adj_price"]).astype(int)
price_data

price_data["change"].value_counts()

from numpy import arange
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

print(price_data.shape)

scatter_matrix(price_data)
pyplot.show()

#split out validation dataset
array = price_data.values
X = array[:,1:8]
Y = array[:,8]
validation_size = .2
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=(seed))

#test options and evaluation metric
num_folds = 10
seed = 7
scoring = 'neg_mean_squared_error'

#spot check algorithms
models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))

#evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=(seed), shuffle=(True))
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
#compare algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

#standardize the dataset
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))
pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))
results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=(seed), shuffle=(True))
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
#compare scaled algorithms
fig = pyplot.figure()
fig.suptitle('Scaled Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

import numpy as np

#KNN Algorithm Tuning
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
k_values = np.array([1,3,5,7,9,11,13,15,17,19,21])
param_grid = dict(n_neighbors=k_values)
model = KNeighborsRegressor()
kfold = KFold(n_splits=num_folds, random_state=(seed), shuffle=(True))
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, params))
    
#ensembles
ensembles = []
ensembles.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()),('AB', AdaBoostRegressor())])))
ensembles.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])))
ensembles.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF', RandomForestRegressor())])))
ensembles.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('ET', ExtraTreesRegressor())])))
results = []
names = []
for name, model in ensembles:
    kfold = KFold(n_splits=num_folds, random_state=(seed), shuffle=(True))
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
#compare scaled ensembles
fig = pyplot.figure()
fig.suptitle('Scaled Ensemble Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

#Extra Trees Wins
predictors = ["interest", "vacancy", "adj_price", "adj_value"]
response = "change"

START = 260
STEP = 52

from sklearn.ensemble import ExtraTreesClassifier

def predict(train, test, predictors, response):
    et = ExtraTreesClassifier()
    et.fit(train[predictors], train[response])
    preds = et.predict(test[predictors])
    return preds

def backtest(data, predictors, response):
    all_preds = []
    for i in range(START, data.shape[0], STEP):
        train = price_data.iloc[:i]
        test = price_data.iloc[i:(i+STEP)]
        all_preds.append(predict(train, test, predictors, response))
        
    preds = np.concatenate(all_preds)
    return preds, accuracy_score(data.iloc[START:][response], preds)

preds, accuracy = backtest(price_data, predictors, response)

preds

accuracy

yearly = price_data.rolling(52, min_periods=1).mean()

yearly_ratios = [p + "_year" for p in predictors]
price_data[yearly_ratios] = price_data[predictors] / yearly[predictors]
price_data

preds, accuracy = backtest(price_data, predictors + yearly_ratios, response)
accuracy

pred_match = (preds == price_data[response].iloc[START:])
pred_match[pred_match == True] = "green"
pred_match[pred_match == False] = "red"

import matplotlib.pyplot as plt
plot_data = price_data.iloc[START:].copy()
plot_data.reset_index().plot.scatter(x="index", y="adj_price", color=pred_match)

from sklearn.inspection import permutation_importance

et = ExtraTreesClassifier()
et.fit(price_data[predictors], price_data[response])
result = permutation_importance(et, price_data[predictors], price_data[response])

result["importances_mean"]
predictors
