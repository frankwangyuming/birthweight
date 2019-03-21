#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 20:31:53 2019

@author: bryancruz

Purpose: This script is for Exploratory Data Analysis for birthweight dataset

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



############################################################################