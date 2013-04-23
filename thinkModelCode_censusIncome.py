# thinkModeCode_censusIncome.py

from numpy import array,flatnonzero,vstack,hstack,ones,zeros,unique,zeros_like,ones_like,shape
from numpy import size, nonzero, delete,mean,median,sort,argsort,corrcoef,diag,eye,exp,delete
from pylab import hist,histogram,plot,bar,figure,xlabel,ylabel,legend,title,subplot
import bMath
import csv
import urllib
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
webpage = urllib.urlopen(url)
datareader = csv.reader(webpage)
ct = 0;
for row in datareader:
 ct = ct+1

webpage = urllib.urlopen(url) 
datareader = csv.reader(webpage)
data = array(-1*ones((ct,15),float),object);
k=0;
for row in datareader:
 data[k,:] = array(row)
 k = k+1;

# delete the last row of data (the reader is reading a blank line at the end)
data = delete(data,-1,0);

""" ----------------------------------------------------------------------------------------
Data features to discretize
 0 - age
 2 - fnlwgt
 4 - education-num
 10 - capital-gain
 11 - capital-loss
 12 - hours-per-week

Categorical features to convert to numerical labels
 1 - workclass
 3 - education
 5 - marital-status
 6 - occupation
 7 - relationship
 8 - race
 9 - sex
 13 - native-country 
 14 - income class (target label)
---------------------------------------------------------------------------------------- """ 

featnames = array(['age','workclass','fnlwgt','education','education-num','marital-status',
	'occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week',
	'native-country','income'],str)

# Find ALL missing or irrelevant rows of data (these need to be removed)
missingIDX,garbage = nonzero(data==' ?')
data = delete(data,missingIDX,0); # delete rows with missing data

# numerical data array initialization
dd = -1*ones((shape(data)),int)

# Now plot the histograms w.r.t. >50k and <=50k income classes
overIDX = flatnonzero(data[:,-1]==' >50K')
underIDX = flatnonzero(data[:,-1]==' <=50K')

# convert income class labels to numerical values
feat = 15; # income: >50K == 1 or <=50K == 0
uniqIncome,garbage,income = unique(data[:-1,feat-1],return_index=True,return_inverse=True)

# Populate dd-array with continuous data feature values
contIDX = [0,2,4,10,11,12]; # continuous valued features indices
for k in contIDX:
 dd[:,k] = array(data[:,k],int);

# --------------------------- Convert categorical to nominal values ---------------------------
catIDX = [1,3,5,6,7,8,9,13,14];
catsPerFeat = [[]]*size(data,1); # number of uniqcat columns is number of features in data
for k in catIDX:
 temp0,temp1,dd[:,k] = unique(data[:,k],return_index=True,return_inverse=True)
 catsPerFeat[k] = data[temp1,k];

# ---------------------- Entropic Quantization of Continuous Features ------------------------
ddq,bin_edges = bMath.quantize(dd[:,contIDX],dd[:,-1])
ct = 0;
for k in contIDX:
 dd[:,k] = ddq[:,ct];
 ct+=1;

# for k in range(size(dd,1)):
#  print k,unique(dd[:,k])

# ------------------ Finally: Convert quantized features to nominal values ------------------
catsPerFeat2 = [[]]*size(data,1);
for k in contIDX:
 temp0,temp1,dd[:,k] = unique(dd[:,k],return_index=True,return_inverse=True)
 catsPerFeat2[k] = data[temp1,k];

# for k in range(size(dd,1)):
#  print k,unique(dd[:,k])


# -------------------------- Data Partitioning and Cross-Validation --------------------------
# As suggested by the UCI machine learning repository, do a 2/3 train, 1/3 test split
numrows = size(dd,0); # number of instances in census income data set
cvp = bMath.cvpartition(numrows,'holdout',0.333);
trainIDX = flatnonzero(cvp['training']);
testIDX = flatnonzero(cvp['test']);
# make sure the 75% (<=50K) vs. 25% (>50K) income class ratios are met!
fracTrainOver50 = len(flatnonzero(dd[trainIDX,-1]))*1.0/len(trainIDX); # 0.2483
fracTrainUnder50 = len(flatnonzero(dd[trainIDX,-1]==0))*1.0/len(trainIDX); # 7516
fracTestOver50 = len(flatnonzero(dd[testIDX,-1]))*1.0/len(testIDX); # 0.2501
fracTestUnder50 = len(flatnonzero(dd[testIDX,-1]==0))*1.0/len(testIDX); # 0.7499
xtrain = dd[trainIDX,:-1];
ytrain = dd[trainIDX,-1];
xtest = dd[testIDX,:-1];
ytest = dd[testIDX,-1];


# # ----------------------------- Feature Ranking and Selection --------------------------------
# featrank = {};
# stat = {};
# bins = {};
# featrank['chi'],stat['chi'],bins['chi'] = bMath.chifs(xtrain,ytrain)
# featrank['corr'],stat['corr'] = bMath.corrfs(xtrain,ytrain)
# featrank['ent'],stat['ent'] = bMath.chiEntropyfs(xtrain,ytrain)
# tempavg = [];
# for j in range(len(featrank['chi'])):
#  temp = [];
#  for k in featrank.keys():
#   temp.append(int(flatnonzero(featrank[k]==j)))
#  tempavg.append(mean(temp));
# featrank['avg'] = argsort(tempavg);

# import informationTheory
# featrank['nmi'],garbage = informationTheory.nmifs3(xtrain,ytrain)

# # Inspecting feature ranking stats, it seems 6 features is sufficient to capture MOST of the info
# sort(stat['chi'])[::-1]
# sort(stat['corr'])[::-1]
# sort(stat['ent'])[::-1]

# # Visualize and Plot stat-values for feature rankings
# subplot(1,3,1) 
# bar(range(len(stat['chi'])),sort(stat['chi'])[::-1]); title('chi')
# subplot(1,3,2)
# bar(range(len(stat['corr'])),sort(abs(stat['corr']))[::-1]); title('corr')
# subplot(1,3,3)
# bar(range(len(stat['ent'])),sort(stat['ent'])[::-1]); title('ent')
# """Good fall off after 6 or 8 features!"""


# goodfeat = featrank['avg'][:8];
# xtrain = xtrain[:,goodfeat]
# xtest = xtest[:,goodfeat]



# # ------------------------------ Naive-Bayes Model Construction ------------------------------
# import naivebayes
# nbClass = naivebayes.nb(xtrain,ytrain,xtest,ytest) 
# nbClass.nbtrain(xtrain,ytrain);
# nbClass.nbclassify(xtrain[:100,:],nbClass.model)
# nbClass.nbholdoutCVsets(xtrain,ytrain,xtest,ytest)

# traindata = vstack((xtrain.T,ytrain)).T
# traincorr = corrcoef(traindata.T);
# traincorr1 = traincorr - eye(len(traincorr)); # remove diagonal 1's from correlation matrix

# from pylab import imshow,figure,colorbar
# figure();
# imshow(traincorr1,interpolation='Nearest')
# colorbar();

# """Look at correlation matrix for entire 14 features!"""



# ------------------------------ Naive_Bayes Model Construction ------------------------------
# ------------------------------  MultinomialNB & ComplementNB  ------------------------------
import naivebayes
from sklearn import naive_bayes
a1 = naive_bayes.GaussianNB() # create class object 'a'
a1.fit(xtrain+1,ytrain); # create fit attributes of mnb class (add +1 to all xtrain values, no zero values)
# determine cross-validation accuracy on "test" set easily:
a1.score(xtest+1,ytest); # ~ 78% accuracy, (add +1 to all xtest values, no zero values)

# Practicing naive_bayes class from sklearn:
from sklearn import naive_bayes
a = naive_bayes.MultinomialNB() # create class object 'a'
a.fit(xtrain+1,ytrain); # create fit attributes of mnb class (add +1 to all xtrain values, no zero values)
# determine cross-validation accuracy on "test" set easily:
a.score(xtest+1,ytest); # ~ 78% accuracy, (add +1 to all xtest values, no zero values)

# created ComplementNB (CNB) version of MNB, based on complementary training data for each class model
b = naivebayes.ComplementNB(); # create class object 'b' for CNB classification model
b.fit(xtrain+1,ytrain);
b.score(xtest+1,ytest);
featnames[argsort(b.coef_)[::-1]]

c = naivebayes.OneVsAll(); # create class object 'b' for CNB classification model
c.fit(xtrain+1,ytrain);
c.score(xtest+1,ytest);
featnames[argsort(c.coef_)[::-1]]



# Convert xtrain data into multilabeled data for MNB() & CNB() classification
lbin2 = naivebayes.NestedLabelBinarizer();
lbin2.nested_fit_transform(xtrain,xtest);
a2 = naive_bayes.MultinomialNB();
a2.fit(lbin2.x_multi,ytrain);
a2.score(lbin2.xtest_multi,ytest); # predict test data against ytest true class vector

a3 = naivebayes.ComplementNB();
a3.fit(lbin2.x_multi,ytrain);
a3.score(lbin2.xtest_multi,ytest)

a4 = naivebayes.OneVsAllNB();
a4.fit(lbin2.x_multi,ytrain);
a4.score(lbin2.xtest_multi,ytest)


"""
- Feature Ranking using difference in feature_log_prob_ between class 0 and 1 (discriminating power!)
- Then after looking at 
"""

a3rankIDX = abs(a3.feature_log_prob_c_[0,:]-a3.feature_log_prob_c_[1,:]).argsort()[::-1]
a3rankScore = abs(a3.feature_log_prob_c_[0,:]-a3.feature_log_prob_c_[1,:])
hist(a3rankScore,20); # shows there's definitely ~ 20 feature likelihood with strong discriminating power


a5 = naivebayes.ComplementNB();
a5.fit(lbin2.x_multi[:,a3rankIDX[:50]],ytrain);
a5.score(lbin2.xtest_multi[:,a3rankIDX[:50]],ytest)

a6 = naive_bayes.MultinomialNB();
a6.fit(lbin2.x_multi[:,a3rankIDX[:50]],ytrain);
a6.score(lbin2.xtest_multi[:,a3rankIDX[:50]],ytest)







# -------------------------------- AGE Feature Manipulation ---------------------------------
# Plot histogram of Age for Census Income Data:
hist(array(data[:-1,0],int),10)
xlabel('Age')
ylabel('Number of Individuals')
title('Histogram of Age')

# Plot histogram of Age for Census Income Data:
subplot(1,2,1); 
hist(array(data[underIDX,0],int),10)
xlabel('Age (<=50K income)')
ylabel('Number of Individuals')
title('Histogram of Age')
subplot(1,2,2); 
hist(array(data[overIDX,0],int),10)
xlabel('Age (>50K income)')
ylabel('Number of Individuals')
title('Histogram of Age')


# -------------------------------- fnlwgt Feature Manipulation ---------------------------------
# Plot histogram of fnlwgt for Census Income Data:
feat = 2;
hist(array(data[:-1,feat],int),10)
xlabel('fnlgt')
ylabel('Number of Individuals')
title('Histogram of fnlgt')

bins = [percentile(array(data[:-1,2],int),0),percentile(array(data[:-1,2],int),20),percentile(array(data[:-1,2],int),40),percentile(array(data[:-1,2],int),60),percentile(array(data[:-1,2],int),80),percentile(array(data[:-1,2],int),100)]

# Plot histogram of Age for Census Income Data:
subplot(1,2,1); 
hist(array(data[underIDX,feat],int),10)
xlabel('Age (<=50K income)')
ylabel('Number of Individuals')
title('Histogram of Age')
subplot(1,2,2); 
hist(array(data[overIDX,feat],int),10)
xlabel('Age (>50K income)')
ylabel('Number of Individuals')
title('Histogram of Age')

