#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 10:15:42 2019

@author: bryancruz

Working Directory: /Users/bryancruz/Desktop/Machine Learning

Purpose: Linear Modeling for birthweight dataset

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


file = 'birthweight_feature_set.xlsx'

bw = pd.read_excel(file)

# Making copies of original dataset for Exploration
bweight = pd.DataFrame.copy(bw)
bweight.to_excel('bweight.xlsx')

bweight = pd.DataFrame.copy(bweight)
bweight.to_excel('bweight.xlsx')



#################################
### Initial Dataset Exploration
#################################

# Dataset Information
bw.info()
bw.shape
bw.count()
bw.describe().round(2)

# Checking for missing values
bw.isnull().any()
bw.isnull().sum()

# Initial Correlation (w/out imputations)
df_corr = bw.corr().round(2)
sns.heatmap(df_corr, square=True, cmap='RdYlGn')



#######################
### Initial Box Plots
#######################

for col in enumerate(bweight):
        print (col)
        bweight.boxplot(column=[col[1]], vert=True, patch_artist=True,
                        meanline=True, showmeans=True)
        plt.tight_layout()
        plt.show()



########################
### Distribution Plots
########################
        
bweight_copy = pd.DataFrame.copy(bweight)

for hist in bweight_copy.iloc[:,:]:
    bweight_copy = pd.DataFrame.copy(bweight)
    bwt = bweight_copy[hist].dropna().round(2)
    sns.distplot(bwt)
    plt.xlabel(hist)
    plt.tight_layout()
    print(skew(bwt))
    plt.show()



###################################
### Imputation for Missing Values
###################################
    
# Create columns for MV Flagging (0s and 1s)
for col in bweight:    
    if bweight[col].isnull().any():
        bweight['m_'+col] = bweight[col].isnull().astype(int)
        
# Imputation Proper
bweight_copy = pd.DataFrame.copy(bweight)

for col in bweight_copy:
    bweight_copy = pd.DataFrame.copy(bweight)
    col_mean = bweight[col].mean()
    col_median = bweight[col].median()
    if bweight_copy[col].isnull().any():
        bwt = bweight_copy[col].dropna().round(2)
        skew_rate = skew(bwt)
        if abs(skew_rate) <= 1:
            bweight[col] = bweight[col].fillna(col_mean).round(2)
        else:
            bweight[col] = bweight[col].fillna(col_median).round(2)
            
# Checking the overall dataset for remaining MVs
print(bweight.isnull().any().any())



########################
### Flagging Outliers
########################

# Boxplots after imputation
for col in enumerate(bweight):
        print (col)
        bweight.boxplot(column=[col[1]], vert=True, patch_artist=True,
                        meanline=True, showmeans=True)
        plt.tight_layout()
        plt.show()

""" 

After looking into boxplots, the following features have outliers:
    
    mage
    monpre
    npvis
    fage
    feduc
    omaps
    fmaps
    drink
    bwght
    
We will then create outlier flagging columns for each feature mentioned.
This will be done using Interquantile Ranges(IQR) method. 

"""
        
# Outlier Flagging Proper
for column in bweight:
    print(column)
    

## Outlier for 'mage'
bweight['out_mage'] = 0

Q1_df = pd.DataFrame(bweight['mage'].quantile([0.25]))
Q3_df = pd.DataFrame(bweight['mage'].quantile([0.75]))
Q1 = Q1_df.iloc[0]['mage']
Q3 = Q3_df.iloc[0]['mage']
IQR = abs(Q3) - abs(Q1)
Min = Q1 - 1.5*IQR
Max = Q3 + 1.5*IQR
for flag in enumerate(bweight.loc[:, 'mage']):
    if flag[1] > Max:
        bweight.loc[flag[0], 'out_mage'] = 1
    if flag[1] < Min:
        bweight.loc[flag[0], 'out_mage'] = -1


## Outlier for 'monpre'
bweight['out_monpre'] = 0

Q1_df = pd.DataFrame(bweight['monpre'].quantile([0.25]))
Q3_df = pd.DataFrame(bweight['monpre'].quantile([0.75]))
Q1 = Q1_df.iloc[0]['monpre']
Q3 = Q3_df.iloc[0]['monpre']
IQR = abs(Q3) - abs(Q1)
Min = Q1 - 1.5*IQR
Max = Q3 + 1.5*IQR
for flag in enumerate(bweight.loc[:, 'monpre']):
    if flag[1] > Max:
        bweight.loc[flag[0], 'out_monpre'] = 1
    if flag[1] < Min:
        bweight.loc[flag[0], 'out_monpre'] = -1


## Outlier for 'npvis'
bweight['out_npvis'] = 0

Q1_df = pd.DataFrame(bweight['npvis'].quantile([0.25]))
Q3_df = pd.DataFrame(bweight['npvis'].quantile([0.75]))
Q1 = Q1_df.iloc[0]['npvis']
Q3 = Q3_df.iloc[0]['npvis']
IQR = abs(Q3) - abs(Q1)
Min = Q1 - 1.5*IQR
Max = Q3 + 1.5*IQR
for flag in enumerate(bweight.loc[:, 'npvis']):
    if flag[1] > Max:
        bweight.loc[flag[0], 'out_npvis'] = 1
    if flag[1] < Min:
        bweight.loc[flag[0], 'out_npvis'] = -1


## Outlier for 'fage'
bweight['out_fage'] = 0

Q1_df = pd.DataFrame(bweight['fage'].quantile([0.25]))
Q3_df = pd.DataFrame(bweight['fage'].quantile([0.75]))
Q1 = Q1_df.iloc[0]['fage']
Q3 = Q3_df.iloc[0]['fage']
IQR = abs(Q3) - abs(Q1)
Min = Q1 - 1.5*IQR
Max = Q3 + 1.5*IQR
for flag in enumerate(bweight.loc[:, 'fage']):
    if flag[1] > Max:
        bweight.loc[flag[0], 'out_fage'] = 1
    if flag[1] < Min:
        bweight.loc[flag[0], 'out_fage'] = -1
        

## Outlier for 'feduc'
bweight['out_feduc'] = 0

Q1_df = pd.DataFrame(bweight['feduc'].quantile([0.25]))
Q3_df = pd.DataFrame(bweight['feduc'].quantile([0.75]))
Q1 = Q1_df.iloc[0]['feduc']
Q3 = Q3_df.iloc[0]['feduc']
IQR = abs(Q3) - abs(Q1)
Min = Q1 - 1.5*IQR
Max = Q3 + 1.5*IQR
for flag in enumerate(bweight.loc[:, 'feduc']):
    if flag[1] > Max:
        bweight.loc[flag[0], 'out_feduc'] = 1
    if flag[1] < Min:
        bweight.loc[flag[0], 'out_feduc'] = -1


## Outlier for 'omaps'
bweight['out_omaps'] = 0

Q1_df = pd.DataFrame(bweight['omaps'].quantile([0.25]))
Q3_df = pd.DataFrame(bweight['omaps'].quantile([0.75]))
Q1 = Q1_df.iloc[0]['omaps']
Q3 = Q3_df.iloc[0]['omaps']
IQR = abs(Q3) - abs(Q1)
Min = Q1 - 1.5*IQR
Max = Q3 + 1.5*IQR
for flag in enumerate(bweight.loc[:, 'omaps']):
    if flag[1] > Max:
        bweight.loc[flag[0], 'out_omaps'] = 1
    if flag[1] < Min:
        bweight.loc[flag[0], 'out_omaps'] = -1


## Outlier for 'fmaps'
bweight['out_fmaps'] = 0

Q1_df = pd.DataFrame(bweight['fmaps'].quantile([0.25]))
Q3_df = pd.DataFrame(bweight['fmaps'].quantile([0.75]))
Q1 = Q1_df.iloc[0]['fmaps']
Q3 = Q3_df.iloc[0]['fmaps']
IQR = abs(Q3) - abs(Q1)
Min = Q1 - 1.5*IQR
Max = Q3 + 1.5*IQR
for flag in enumerate(bweight.loc[:, 'fmaps']):
    if flag[1] > Max:
        bweight.loc[flag[0], 'out_fmaps'] = 1
    if flag[1] < Min:
        bweight.loc[flag[0], 'out_fmaps'] = -1


## Outlier for 'drink'
bweight['out_drink'] = 0

Q1_df = pd.DataFrame(bweight['drink'].quantile([0.25]))
Q3_df = pd.DataFrame(bweight['drink'].quantile([0.75]))
Q1 = Q1_df.iloc[0]['drink']
Q3 = Q3_df.iloc[0]['drink']
IQR = abs(Q3) - abs(Q1)
Min = Q1 - 1.5*IQR
Max = Q3 + 1.5*IQR
for flag in enumerate(bweight.loc[:, 'drink']):
    if flag[1] > Max:
        bweight.loc[flag[0], 'out_drink'] = 1
    if flag[1] < Min:
        bweight.loc[flag[0], 'out_drink'] = -1


## Outlier for 'bwght'
bweight['out_bwght'] = 0

Q1_df = pd.DataFrame(bweight['bwght'].quantile([0.25]))
Q3_df = pd.DataFrame(bweight['bwght'].quantile([0.75]))
Q1 = Q1_df.iloc[0]['bwght']
Q3 = Q3_df.iloc[0]['bwght']
IQR = abs(Q3) - abs(Q1)
Min = Q1 - 1.5*IQR
Max = Q3 + 1.5*IQR
for flag in enumerate(bweight.loc[:, 'bwght']):
    if flag[1] > Max:
        bweight.loc[flag[0], 'out_bwght'] = 1
    if flag[1] < Min:
        bweight.loc[flag[0], 'out_bwght'] = -1
        


##########################
### Correlation Analysis
##########################

bw_corr = bweight.corr().round(2)

print(bw_corr)

bw_corr.loc['bwght'].sort_values(ascending = False)


# Heatmap
sns.palplot(sns.color_palette('coolwarm', 12))
fig, ax = plt.subplots(figsize=(13,13))
bw_corr2 = bw_corr.iloc[0:18, 0:18]

sns.heatmap(bw_corr2,
            cmap = 'coolwarm',
            square = True,
            annot = True,
            linecolor = 'black',
            linewidths = 0.5)


plt.savefig('Birthweight Correlation Heatmap.png')
plt.show()


# Saving after Exploratory Data Analysis
bweight.to_excel('Birthweight_EDA.xlsx')


#######################################################################

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
