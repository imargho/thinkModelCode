""" ----------------------------------------------------------------------------------------
thinkModeCode_carData.py

This dataset was taken from the UCI Machine Learning Repository 
(http://archive.ics.uci.edu/ml/datasets.html)

1. Number of Instances: 1728
   (instances completely cover the attribute space)

2. Number of Attributes (features): 6

3. Data feature descriptions:
	0 - buying:   vhigh, high, med, low.
	1 - maint:    vhigh, high, med, low.
	2 - doors:    2,  3,	4, 5more.
	3 - persons:  2,  4, more.
	4 - lug_boot: small,  med, big.
	5 - safety:   low, 	  med, high.

4. Class Labels (to predict thru classification):
	car evaluation: unacc, acc, good, vgood

5. Missing Attribute Values: none

6. Class Distribution (number of instances per class)
	There is a sample imbalance (very common to real world data sets)

   class      N          N[%]
   -----------------------------
   unacc     1210     (70.023 %) 
   acc        384     (22.222 %) 
   good        69     ( 3.993 %) 
   v-good      65     ( 3.762 %) 
---------------------------------------------------------------------------------------- """ 
#import naivebayes
from sklearn.processing import LabelBinarizer
from sklearn import naive_bayes
import numpy as np
import csv
import random
import urllib
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
webpage = urllib.urlopen(url)
datareader = csv.reader(webpage)
ct = 0;
for row in datareader:
 ct = ct+1

webpage = urllib.urlopen(url) 
datareader = csv.reader(webpage)
data = np.array(-1*np.ones((ct,7),float),object);
k=0;
for row in datareader:
 data[k,:] = np.array(row)
 k = k+1;

featnames = np.array(['buyPrice','maintPrice','numDoors','numPersons','lugBoot','safety'],str)

keys = [[]]*np.size(data,1)
numdata = -1.0*np.ones_like(data);
# convert string objects to integer values for modeling:
for k in range(np.size(data,1)):
 keys[k],garbage,numdata[:,k] = np.unique(data[:,k],True,True)

numrows = np.size(numdata,0); # number of instances in car data set
numcols = np.size(numdata,1); # number of columns in car data set
numdata = np.array(numdata,int)
xdata = numdata[:,:-1]; # x-data is all data BUT the last column which are the class labels
ydata = numdata[:,-1]; # y-data is set to class labels in the final column, signified by -1

# ------------------ numdata multilabel -> binary conversion for NB-Model ---------------------
lbin = LabelBinarizer();

for k in range(np.size(xdata,1)): # loop thru number of columns in xdata
 if k==0:
  xdata_ml = lbin.fit_transform(xdata[:,k]);
 else:
  xdata_ml = np.hstack((xdata_ml,lbin.fit_transform(xdata[:,k])))
ydata_ml = lbin.fit_transform(ydata)


# -------------------------- Data Partitioning and Cross-Validation --------------------------
# As suggested by the UCI machine learning repository, do a 2/3 train, 1/3 test split
allIDX = np.arange(numrows);
random.shuffle(allIDX); # randomly shuffles allIDX order for creating 'holdout' sample
holdout_number = numrows/10; # holdout 10% of full sample set to perform validation
testIDX = allIDX[0:holdout_number];
trainIDX = allIDX[holdout_number:];

# create training and test data sets
xtest = xdata_ml[testIDX,:];
xtrain = xdata_ml[trainIDX,:];
ytest = ydata_ml[testIDX];
ytrain = ydata_ml[trainIDX];

# ------------------------------ Naive_Bayes Model Construction ------------------------------
# ------------------------------  MultinomialNB & ComplementNB  ------------------------------
mnb = naive_bayes.MultinomialNB();
mnb.fit(xtrain,ytrain);
mnb.score(xtest,ytest); # test data pred vs. true class labels (outputs classification accuracy)

cnb = naivebayes.ComplementNB();
cnb.fit(xtrain,ytrain);
cnb.score(xtest,ytest)

ova = naivebayes.OneVsAllNB();
ova.fit(xtrain,ytrain);
ova.score(xtest,ytest)



#=============================================================================================
#=============================================================================================
# ------------------------------ Naive_Bayes Model Construction ------------------------------
# ------------------------------  MultinomialNB & ComplementNB  ------------------------------
#=============================================================================================
#=============================================================================================
# Convert xtrain data into multilabeled data for MNB() & CNB() classification
xtest = xdata[testIDX,:];
xtrain = xdata[trainIDX,:];
ytest = ydata[testIDX];
ytrain = ydata[trainIDX];


lbin = naivebayes.NestedLabelBinarizer();
lbin.nested_fit_transform(xtrain,xtest);
mnb = naive_bayes.MultinomialNB();
mnb.fit(lbin.x_multi,ytrain);
mnb.score(lbin.xtest_multi,ytest); # predict test data against ytest true class vector

cnb = naivebayes.ComplementNB();
cnb.fit(lbin.x_multi,ytrain);
cnb.score(lbin.xtest_multi,ytest)

ova = naivebayes.OneVsAllNB();
ova.fit(lbin.x_multi,ytrain);
ova.score(lbin.xtest_multi,ytest)


#=============================================================================================
#=============================================================================================
#=============================================================================================
"""
- Feature Ranking using difference in feature_log_prob_ between class 0 and 1 (discriminating power!)
- Then after looking at 
"""
import pylab as pl

cnbrankIDX = abs(cnb.feature_log_prob_c_[0,:]-cnb.feature_log_prob_c_[1,:]).argsort()[::-1]
cnbrankScore = abs(cnb.feature_log_prob_c_[0,:]-cnb.feature_log_prob_c_[1,:])
pl.hist(cnbrankScore,20); # shows there's definitely ~ 20 feature likelihood with strong discriminating power


cnb_f = naivebayes.ComplementNB();
cnb_f.fit(lbin.x_multi[:,cnbrankIDX[:50]],ytrain);
cnb_f.score(lbin.xtest_multi[:,cnbrankIDX[:50]],ytest)

mnb = naive_bayes.MultinomialNB();
mnb.fit(lbin.x_multi[:,cnbrankIDX[:50]],ytrain);
mnb.score(lbin.xtest_multi[:,cnbrankIDX[:50]],ytest)


