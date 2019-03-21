#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 20:34:24 2019

@author: bryancruz

Working Directory: /Users/bryancruz/Desktop/Machine Learning

Purpose: This script is for the Best Regression Model 

"""

# Importing new libraries
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor 
from sklearn.tree import export_graphviz 
from sklearn.externals.six import StringIO 
import statsmodels.api as sm
from IPython.display import Image 
import pydotplus 
import seaborn as sns 
from scipy.stats import skew 
from matplotlib import style

# Importing other libraries
import pandas as pd
import matplotlib.pyplot as plt


file_new = 'Birthweight_EDA.xlsx'
bweight_new = pd.read_excel(file_new)

#########################
### Regression Analysis
#########################

# Preparing data and target sets
bw_data = bweight_new.loc[:,['mage',
                             'monpre',
                             'feduc',
                             'cigs',
                             'drink',
                             'male',
                             'mwhte',
                             'mblck',
                             'moth',
                             'fwhte',
                             'fblck',
                             'foth',
                             'out_monpre',
                             'out_npvis',
                             'out_fage',
                             'out_fmaps']]

bw_target = bweight_new.loc[:,'bwght']
    
# Train Test Split Proper
X_train, X_test, y_train, y_test = train_test_split(
                                                    bw_data,
                                                    bw_target,
                                                    test_size=0.1,
                                                    random_state=508)

#######
# KNN Regressor Model

# Getting Optimal k for KNN Regressor
training_accuracy = []
test_accuracy = []


neighbors_settings = range(1, 51)


for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsRegressor(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)
    
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))



plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

test_accuracy
max(test_accuracy)
test_accuracy.index(max(test_accuracy))
# -- reults show best n is n_neighbor = 5

# KNN Modeling Proper
knn_reg_bw = KNeighborsRegressor(algorithm='auto', n_neighbors=5)

knn_reg_bw_fit = knn_reg_bw.fit(X_train, y_train)

# Scoring the model
y_score_knn_bw_opt = knn_reg_bw.score(X_test, y_test)


# Generating Predictions based on the optimal KNN model
knn_reg_bw_opt_pred = knn_reg_bw_fit.predict(X_test)


#######
# OLS: Linear Regression() by sklearn

# Preparing the Model
lr_bw = LinearRegression(fit_intercept=True)

# Fitting the model
lr_bw_fit = lr_bw.fit(X_train, y_train)

# Predictions
lr_bw_pred = lr_bw_fit.predict(X_test)

# Scoring the model
y_score_ols_bw_opt = lr_bw_fit.score(X_test, y_test)



#######
# Ridge Regression

# Preparing the Model
ridge = Ridge(alpha = 0.1, normalize = True)

# Fitting the model
ridge.fit(X_train, y_train)

# Predictions
ridge_pred = ridge.predict(X_test)

# Scoring the model
ridge_bw_score = ridge.score(X_test, y_test)



######
# OLS by statsmodels.api

# Adding intercept since sm.OLS does not add intercept automatically
X_int_train = sm.add_constant(X_train)

# Preparing Model
mod = sm.OLS(y_train, X_int_train)

# Fitting Model
res = mod.fit()

# Predictions
X_int_test = sm.add_constant(X_test)
y_smols_pred = res.predict(X_int_test)

# Getting coefficients and R-squared, checking p-values 
res.summary()

smols_rsquared = res.rsquared.round(3)


######
# Comparing all R-Squared & Scores from Different Regression Model Types

print(f"""
          R-Squared & Scores Comparison:
              
          Optimal model KNN score   : {y_score_knn_bw_opt.round(3)}
          sklearn LR_OLS score      : {y_score_ols_bw_opt.round(3)}
          Ridge Regression score    : {ridge_bw_score.round(3)}
          sm.ols test set R-Squared : {smols_rsquared.round(3)}
          """)

# -- Using bweight_new dataset, sm.ols by statsmodels.api shows highest 
# -- R-squared. Thus, this will be our chosen model. 

print(res.summary())


#######
# Storing Predictions of Best Model to Excel 
model_pred_df = pd.DataFrame({'Actual' : y_test,
                              'OLS(statsmodels)_Pred': y_smols_pred})


model_pred_df.to_excel("Birthweight_ols_Model_Predictions.xlsx")



#########################################################################

#################################
### Decision Tree 
#################################

# Modeling
tree_full = DecisionTreeRegressor(random_state=508)

tree_full.fit(X_train, y_train)

print('Training Score', tree_full.score(X_train, y_train).round(4))

print('Testing Score:', tree_full.score(X_test, y_test).round(4))

# Creating a tree with only two levels.
tree_2 = DecisionTreeRegressor(max_depth=2,
                               random_state=508)

tree_2_fit = tree_2.fit(X_train, y_train)


print('Training Score', tree_2.score(X_train, y_train).round(4))
print('Testing Score:', tree_2.score(X_test, y_test).round(4))


# Visualizing the tree
dot_data = StringIO()
    
export_graphviz(decision_tree = tree_2,
                out_file = dot_data,
                filled=True,
                rounded=True,
                special_characters=True,
                feature_names = bw_data.columns)


graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

# See Image
Image(graph.create_png(),
      height=500,
      width=800)

tree_leaf_50 = DecisionTreeRegressor(criterion='mse',
                                     min_samples_leaf=50,
                                     random_state=508)

tree_leaf_50.fit(X_train, y_train)

print('Training Score', tree_leaf_50.score(X_train, y_train).round(4))
print('Testing Score:', tree_leaf_50.score(X_test, y_test).round(4))


# Defining a function to visualize feature importance
def plot_feature_importances(model, train = X_train, export=False):
    fig, ax = plt.subplots(figsize=(12,9))
    n_features = X_train.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")


plot_feature_importances(tree_leaf_50,
                         train = X_train,
                         export=True)
plt.show()


plot_feature_importances(tree_full,
                         train = X_train,
                         export=False)
plt.show()


# Residual Plot

style.use('bmh')
plt.scatter(lr_bw_pred, y_test-lr_bw_pred)
plt.show()





############################################################################
