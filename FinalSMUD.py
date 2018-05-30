# -*- coding: utf-8 -*-
"""
This is a regime-dependent ANN method for solar irradiance prediction.  
This classify-then-predict technique first classifies regimes with K-means classification and then predicts solar irradiance (Kt) with regime-dependent ANNs.

Written by Tyler McCandless as part of his PhD Dissertation
"""
import matplotlib.pyplot as plt
import numpy as np
#import pymc as pm
import pandas as pd
import statsmodels.api as sm
import matplotlib.cm as cm
import scipy
#import theano
from pylab import *
from sklearn import preprocessing
from scipy.cluster.vq import kmeans,vq
from scipy.spatial.distance import *
import pybrain
import cPickle as pickle
from math import sqrt
from pybrain.datasets.supervised import SupervisedDataSet as SDS
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.validation import ModuleValidator, Validator
from pybrain.supervised.trainers import BackpropTrainer as trainer
from sklearn.metrics import mean_squared_error as MSE
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.structure import SigmoidLayer
import neurolab as nl
import csv

def NLabPredict(x,y,neurons,x_test,y_test,regime, netALL,z):
    y = y.T
    y_t = y_test
    y_test = y_test.T
    size = len(x)
    input_size = x.shape[1]
#    print("input size...")
#    print (input_size)
    inp = x
    tar = y.reshape(size,1)
    
    norm_inp = nl.tool.Norm(inp)
    inp = norm_inp(inp)

    size_test = len(x_test)
    x_test_size = x_test.shape[1]
    tar_test = y_test.reshape(size_test,1)
    
    ## Make KT_Persistence Prediction
    kt_pred = x_test['KT_prev15'].values
    MAE_kt = np.mean(abs(kt_pred-y_t))
    MAE_kt_std = np.std(abs(kt_pred-y_t))  
    norm_inp_test = nl.tool.Norm(x_test)
    x_test = norm_inp_test(x_test)
    if z == 1:
        out_test_all = netALL.sim(x_test)
        MAE_all = np.nanmean(abs(out_test_all-tar_test))
        MAE_all_std = np.std(abs(out_test_all-tar_test))
    else:
        MAE_all = 'NaN'
        MAE_all_std = 'NaN'
        
    net = nl.net.newff([[0, 1]]*input_size,[neurons, 1])
    # Change train function
    net.trainf = nl.train.train_rprop
    # new inicialized
    net.init()
    # Change error function
    net.errorf = nl.error.MSE()
    # Train network
    error = net.train(inp, tar, epochs=500, show=500, goal = 0.000001)
    # Simulate network
    out = net.sim(inp)
    out_test = net.sim(x_test)
    MAE = np.mean(abs(out_test-tar_test))
    MAEstd = np.std(abs(out_test-tar_test))
    bestMAE = MAE
    bestSTD = MAEstd
    bestNum = neurons
        
    #print(titlelist[z])
    print("The MAE and STD_MAE for Regime "+str(regime)+" with "+str(bestNum)+"neurons KT Persistence is...")
    print(MAE_kt)
    print(MAE_kt_std)
    print("The MAE and STD_MAE for Regime "+str(regime)+" with "+str(bestNum)+"neurons (ANN w/o regime ID) is...")
    print(MAE_all)
    print(MAE_all_std)
    print("The MAE and STD_MAE for Regime "+str(regime)+" with "+str(bestNum)+"neurons (regime-dependent ANNs) is...")
    print(bestMAE)
    print(bestSTD)
    print("The length of this regime is...")
    print(len(y_test))
    Length = len(y_test)
    return MAE, Length, net, out, tar, MAE_kt
    
def kMeansRegime(regimes,kt_var_train,kt_var_trainTest,kt_var_test):
    ############### Test K-Means Clustering Algorithm for Kt ########################
    X = kt_var_train.values
    X_normalized = preprocessing.normalize(X, norm='l2',axis=0)
    X_test = kt_var_test.values
    X_test_normalized = preprocessing.normalize(X_test, norm='l2',axis=0)
    X_trainTest = kt_var_trainTest.values
    X_trainTest_normalized = preprocessing.normalize(X_trainTest, norm='l2',axis=0)
    # Determine your k-range
    K = range(1,regimes+1)
    
# Scipy.cluster.vq.kmeans
    KM = [kmeans(X_normalized,k) for k in K ] # Apply kmeans 1 to 10
    centroids = [cent for (cent,var) in KM] # Cluster centroids
#    D_k = [cdist(X_normalized,cent,'euclidean') for cent in centroids]
#    cIdx = [np.argmin(D,axis=1) for D in D_k]
#    dist = [np.min(D,axis=1) for D in D_k]
#    avgWithinSS = [sum(d)/X_normalized.shape[0] for d in dist]
#    kIdx = 2    
    
    
    from sklearn.cluster import KMeans
    km = KMeans(regimes, init='k-means++') # initialize - Note that indexing is 0 based, thus k = 7 
    #km = KMeans()
    km.fit(X_normalized)
    c = km.predict(X_normalized) # classify into regimes
    c_trainTest = km.predict(X_trainTest_normalized) # classify trainTest dataset regimes...
    c_test = km.predict(X_test_normalized) # classify regimes on independent test dataset
    initial = [scipy.cluster.vq.kmeans(X_normalized,u) for u in range(1,regimes+1)]
    cent, var = initial[regimes-1]
    assignment,cdist = scipy.cluster.vq.vq(X_normalized,cent) # Use vq() to get as assignment for each obs.
    test_assignment,c_test_dist = scipy.cluster.vq.vq(X_test_normalized,cent)
    trainTest_assignment,c_trainTest_dist = scipy.cluster.vq.vq(X_trainTest_normalized,cent)
    smudTrainTrain['KMeans_Cluster'] = assignment
    smudTest['KMeans_Cluster']= test_assignment
    smudTrainTest['KMeans_Cluster']= trainTest_assignment
#    print "assignment"
#    print assignment
    dictOfRegimesTrain = {}
    dictOfRegimesTrainTest = {}
    dictOfRegimesTest = {}
    for q in range(0,regimes):
        dictOfRegimesTrain['Regime'+str(q+1)+'_train'] = smudTrainTrain[smudTrainTrain.KMeans_Cluster == q]
        dictOfRegimesTrainTest['Regime'+str(q+1)+'_traintest'] = smudTrainTest[smudTrainTest.KMeans_Cluster == q]
        dictOfRegimesTest['Regime'+str(q+1)+'_test'] = smudTest[smudTest.KMeans_Cluster == q]
    return (dictOfRegimesTrain, dictOfRegimesTrainTest, dictOfRegimesTest)
        
        
############################  Start of the main part of the code #######################################################################################
## Now that the data is read in, it's time to decide which predictor dataset to use!!
## Sensitivity tests (Chapter 2 of dissertation) showed which predictor sets to use...code has optimal sets of inputs for K-means and predictors for ANN...
###########################  Test on Satellite data... #######################################################################
#header_row = ['Year','Month','Day','DOY','Hour','Lat','Lon','SZA','Mask','Type','CF','Height','P','T','Tau','LWP','r_e','R_650','R_3.75','T_650','T_3.75','epochTime']
smudTrainTrain = pd.read_csv('SMUDTrainTrainNO999s.csv') # This is the TRAINING dataset with all derived variables that are missing (i.e. the clear cases) removed
smudTrainTest = pd.read_csv('SMUDTrainTestNO999s.csv')  # This is the TEST SET OF THE TRAINING dataset with all derived variables that are missing (i.e. the clear cases) removed
smudTest = pd.read_csv('SMUDTestNO999s.csv')  # This is the INDEPENDENT TEST dataset with all derived variables that are missing (i.e. the clear cases) removed
# This dataset will have 999s in the derived variables from GOES-East.  Will need to remove change predictors to include only measured GOES and do no regime classification...
smudTrainTrain999 = pd.read_csv('SMUDTrainTrain999s.csv') # This is the TRAINING dataset with all derived and measured GOES 
smudTrainTest999 = pd.read_csv('SMUDTrainTest999s.csv')  # This is the TEST SET OF THE TRAINING dataset with all derived and measured GOES 
smudTest999 = pd.read_csv('SMUDTest999s.csv')  # This is the INDEPENDENT TEST dataset with all derived and measured GOES 

## Create variable that is the temporal variability of the Kt...
smudTrainTrain['Kt_Variability'] = smudTrainTrain['std_prev1Hr']
smudTrainTest['Kt_Variability'] = smudTrainTest['std_prev1Hr']
smudTest['Kt_Variability'] = smudTest['std_prev1Hr']
    
############## Train ANN without regime identification....######################
## Note that this is only required to compare ANN vs RDANN...
inputAll = smudTrainTrain[["MV72483000_T", "MV72483009_T", "MV72483016_T", "MV72483000_dewpt", "MV72483009_dewpt", "MV72483016_dewpt", "MV72483000_cloud_cov", "MV72483009_cloud_cov", "MV72483016_cloud_cov", "MV72483000_prob_precip01", "MV72483009_prob_precip01", "MV72483016_prob_precip01", "MV72483000_qpf01", "MV72483009_qpf01", "MV72483016_qpf01", "MV72483000_wind_speed", "MV72483009_wind_speed", "MV72483016_wind_speed", "T_Td", "cloud_cov_STD", "cloud_cov_SQRD","NearbyKt_prev15", "NearbyKt_prev15std", "KT_prev60", "KT_prev45", "KT_prev30", "KT_prev15",  "KT_Slope",  "std_prev1Hr","SineJulianDay", "CosJulianDay", "KT15_KT30",'SZA','Type','T','Tau','r_e','R_650','R_3.75','T_650','T_3.75']]# training[["MV72483000_T", "MV72483009_T", "MV72483016_T", "MV72483000_dewpt", "MV72483009_dewpt", "MV72483016_dewpt", "MV72483000_cloud_cov", "MV72483009_cloud_cov", "MV72483016_cloud_cov", "MV72483000_prob_precip01", "MV72483009_prob_precip01", "MV72483016_prob_precip01", "MV72483000_qpf01", "MV72483009_qpf01", "MV72483016_qpf01", "MV72483000_wind_speed", "MV72483009_wind_speed", "MV72483016_wind_speed", "T_Td", "cloud_cov_STD", "cloud_cov_SQRD", "KT_prev60", "KT_prev45", "KT_prev30", "KT_prev15",  "KT_Slope",  "std_prev1Hr", "NearbyKt_prev15", "NearbyKt_prev15std", "LocalHour", "SineJulianDay",  "KT15_KT30"]]
targetAll = smudTrainTrain[["KT_post15"]].values # Predictions for 15-min ahead
x_trainTestAll = smudTrainTest[["MV72483000_T", "MV72483009_T", "MV72483016_T", "MV72483000_dewpt", "MV72483009_dewpt", "MV72483016_dewpt", "MV72483000_cloud_cov", "MV72483009_cloud_cov", "MV72483016_cloud_cov", "MV72483000_prob_precip01", "MV72483009_prob_precip01", "MV72483016_prob_precip01", "MV72483000_qpf01", "MV72483009_qpf01", "MV72483016_qpf01", "MV72483000_wind_speed", "MV72483009_wind_speed", "MV72483016_wind_speed", "T_Td", "cloud_cov_STD", "cloud_cov_SQRD", "NearbyKt_prev15", "NearbyKt_prev15std","KT_prev60", "KT_prev45", "KT_prev30", "KT_prev15",  "KT_Slope", "std_prev1Hr","SineJulianDay",  "CosJulianDay", "KT15_KT30",'SZA','Type','T','Tau','r_e','R_650','R_3.75','T_650','T_3.75']]# training[["MV72483000_T", "MV72483009_T", "MV72483016_T", "MV72483000_dewpt", "MV72483009_dewpt", "MV72483016_dewpt", "MV72483000_cloud_cov", "MV72483009_cloud_cov", "MV72483016_cloud_cov", "MV72483000_prob_precip01", "MV72483009_prob_precip01", "MV72483016_prob_precip01", "MV72483000_qpf01", "MV72483009_qpf01", "MV72483016_qpf01", "MV72483000_wind_speed", "MV72483009_wind_speed", "MV72483016_wind_speed", "T_Td", "cloud_cov_STD", "cloud_cov_SQRD", "KT_prev60", "KT_prev45", "KT_prev30", "KT_prev15",  "KT_Slope",  "std_prev1Hr", "NearbyKt_prev15", "NearbyKt_prev15std", "LocalHour", "SineJulianDay",  "KT15_KT30"]]
y_trainTestAll = smudTrainTest[["KT_post15"]].values # Predictions for 15-min ahead 
x_testAll = smudTest[["MV72483000_T", "MV72483009_T", "MV72483016_T", "MV72483000_dewpt", "MV72483009_dewpt", "MV72483016_dewpt", "MV72483000_cloud_cov", "MV72483009_cloud_cov", "MV72483016_cloud_cov", "MV72483000_prob_precip01", "MV72483009_prob_precip01", "MV72483016_prob_precip01", "MV72483000_qpf01", "MV72483009_qpf01", "MV72483016_qpf01", "MV72483000_wind_speed", "MV72483009_wind_speed", "MV72483016_wind_speed", "T_Td", "cloud_cov_STD", "cloud_cov_SQRD", "NearbyKt_prev15", "NearbyKt_prev15std", "KT_prev60", "KT_prev45", "KT_prev30", "KT_prev15",  "KT_Slope",  "std_prev1Hr","SineJulianDay",  "CosJulianDay", "KT15_KT30",'SZA','Type','T','Tau','r_e','R_650','R_3.75','T_650','T_3.75']]# training[["MV72483000_T", "MV72483009_T", "MV72483016_T", "MV72483000_dewpt", "MV72483009_dewpt", "MV72483016_dewpt", "MV72483000_cloud_cov", "MV72483009_cloud_cov", "MV72483016_cloud_cov", "MV72483000_prob_precip01", "MV72483009_prob_precip01", "MV72483016_prob_precip01", "MV72483000_qpf01", "MV72483009_qpf01", "MV72483016_qpf01", "MV72483000_wind_speed", "MV72483009_wind_speed", "MV72483016_wind_speed", "T_Td", "cloud_cov_STD", "cloud_cov_SQRD", "KT_prev60", "KT_prev45", "KT_prev30", "KT_prev15",  "KT_Slope",  "std_prev1Hr", "NearbyKt_prev15", "NearbyKt_prev15std", "LocalHour", "SineJulianDay",  "KT15_KT30"]]
y_testAll = smudTest[["KT_post15"]].values # Predictions for 15-min ahead    
x = inputAll
y = targetAll
## Will need to change next two lines to the independent verification to plot final results (not sensitivity study results)
x_test = x_testAll
y_test = y_testAll 
size = len(x)
input_size = x.shape[1]
inp = x
tar = y.reshape(size,1)

norm_inp = nl.tool.Norm(inp)
inp = norm_inp(inp)        
size_test = len(x_test)
x_test_size = x_test.shape[1]
tar_test = y_test.reshape(size_test,1)
 #   size_testAll = len(x_test)
 #   tar_testAll = y_testAll.reshape(size_testAll,1)
norm_inp_test = nl.tool.Norm(x_test)
x_test = norm_inp_test(x_test)
netALL = nl.net.newff([[0, 1]]*input_size,[20, 1])
netALL.trainf = nl.train.train_rprop
netALL.init()
netALL.errorf = nl.error.MSE()
error = netALL.train(inp, tar, epochs=500, show=500, goal = 0.00001)


################# WILL NEED TO ADD SECTION FOR PREDICTING THE 999s (I.e. clear cases) ##################################################
############## Train ANN without regime identification....######################
## Note that this is only required to compare ANN vs RDANN...
inputAll999 = smudTrainTrain999[["MV72483000_T", "MV72483009_T", "MV72483016_T", "MV72483000_dewpt", "MV72483009_dewpt", "MV72483016_dewpt", "MV72483000_cloud_cov", "MV72483009_cloud_cov", "MV72483016_cloud_cov", "MV72483000_prob_precip01", "MV72483009_prob_precip01", "MV72483016_prob_precip01", "MV72483000_qpf01", "MV72483009_qpf01", "MV72483016_qpf01", "MV72483000_wind_speed", "MV72483009_wind_speed", "MV72483016_wind_speed", "T_Td", "cloud_cov_STD", "cloud_cov_SQRD","NearbyKt_prev15", "NearbyKt_prev15std", "KT_prev60", "KT_prev45", "KT_prev30", "KT_prev15",  "KT_Slope",  "std_prev1Hr","SineJulianDay", "CosJulianDay", "KT15_KT30",'SZA','R_650','R_3.75','T_650','T_3.75']]# training[["MV72483000_T", "MV72483009_T", "MV72483016_T", "MV72483000_dewpt", "MV72483009_dewpt", "MV72483016_dewpt", "MV72483000_cloud_cov", "MV72483009_cloud_cov", "MV72483016_cloud_cov", "MV72483000_prob_precip01", "MV72483009_prob_precip01", "MV72483016_prob_precip01", "MV72483000_qpf01", "MV72483009_qpf01", "MV72483016_qpf01", "MV72483000_wind_speed", "MV72483009_wind_speed", "MV72483016_wind_speed", "T_Td", "cloud_cov_STD", "cloud_cov_SQRD", "KT_prev60", "KT_prev45", "KT_prev30", "KT_prev15",  "KT_Slope",  "std_prev1Hr", "NearbyKt_prev15", "NearbyKt_prev15std", "LocalHour", "SineJulianDay",  "KT15_KT30"]]
targetAll999 = smudTrainTrain999[["KT_post15"]].values # Predictions for 15-min ahead
x_trainTestAll = smudTrainTest999[["MV72483000_T", "MV72483009_T", "MV72483016_T", "MV72483000_dewpt", "MV72483009_dewpt", "MV72483016_dewpt", "MV72483000_cloud_cov", "MV72483009_cloud_cov", "MV72483016_cloud_cov", "MV72483000_prob_precip01", "MV72483009_prob_precip01", "MV72483016_prob_precip01", "MV72483000_qpf01", "MV72483009_qpf01", "MV72483016_qpf01", "MV72483000_wind_speed", "MV72483009_wind_speed", "MV72483016_wind_speed", "T_Td", "cloud_cov_STD", "cloud_cov_SQRD","NearbyKt_prev15", "NearbyKt_prev15std", "KT_prev60", "KT_prev45", "KT_prev30", "KT_prev15",  "KT_Slope", "std_prev1Hr","SineJulianDay",  "CosJulianDay", "KT15_KT30",'SZA','R_650','R_3.75','T_650','T_3.75']]# training[["MV72483000_T", "MV72483009_T", "MV72483016_T", "MV72483000_dewpt", "MV72483009_dewpt", "MV72483016_dewpt", "MV72483000_cloud_cov", "MV72483009_cloud_cov", "MV72483016_cloud_cov", "MV72483000_prob_precip01", "MV72483009_prob_precip01", "MV72483016_prob_precip01", "MV72483000_qpf01", "MV72483009_qpf01", "MV72483016_qpf01", "MV72483000_wind_speed", "MV72483009_wind_speed", "MV72483016_wind_speed", "T_Td", "cloud_cov_STD", "cloud_cov_SQRD", "KT_prev60", "KT_prev45", "KT_prev30", "KT_prev15",  "KT_Slope",  "std_prev1Hr", "NearbyKt_prev15", "NearbyKt_prev15std", "LocalHour", "SineJulianDay",  "KT15_KT30"]]
y_trainTestAll = smudTrainTest999[["KT_post15"]].values # Predictions for 15-min ahead 
x_testAll999 = smudTest999[["MV72483000_T", "MV72483009_T", "MV72483016_T", "MV72483000_dewpt", "MV72483009_dewpt", "MV72483016_dewpt", "MV72483000_cloud_cov", "MV72483009_cloud_cov", "MV72483016_cloud_cov", "MV72483000_prob_precip01", "MV72483009_prob_precip01", "MV72483016_prob_precip01", "MV72483000_qpf01", "MV72483009_qpf01", "MV72483016_qpf01", "MV72483000_wind_speed", "MV72483009_wind_speed", "MV72483016_wind_speed", "T_Td", "cloud_cov_STD", "cloud_cov_SQRD","NearbyKt_prev15", "NearbyKt_prev15std", "KT_prev60", "KT_prev45", "KT_prev30", "KT_prev15",  "KT_Slope",  "std_prev1Hr","SineJulianDay",  "CosJulianDay", "KT15_KT30",'SZA','R_650','R_3.75','T_650','T_3.75']]# training[["MV72483000_T", "MV72483009_T", "MV72483016_T", "MV72483000_dewpt", "MV72483009_dewpt", "MV72483016_dewpt", "MV72483000_cloud_cov", "MV72483009_cloud_cov", "MV72483016_cloud_cov", "MV72483000_prob_precip01", "MV72483009_prob_precip01", "MV72483016_prob_precip01", "MV72483000_qpf01", "MV72483009_qpf01", "MV72483016_qpf01", "MV72483000_wind_speed", "MV72483009_wind_speed", "MV72483016_wind_speed", "T_Td", "cloud_cov_STD", "cloud_cov_SQRD", "KT_prev60", "KT_prev45", "KT_prev30", "KT_prev15",  "KT_Slope",  "std_prev1Hr", "NearbyKt_prev15", "NearbyKt_prev15std", "LocalHour", "SineJulianDay",  "KT15_KT30"]]
y_testAll999 = smudTest999[["KT_post15"]].values # Predictions for 15-min ahead    
x = inputAll999
y = targetAll999
## Will need to change next two lines to the independent verification to plot final results (not sensitivity study results)
x_test = x_testAll999
y_test = y_testAll999 
kt_predNoRegimes = x_testAll999['KT_prev15'].values
MAE_ktNoRegimes = np.mean(abs(kt_predNoRegimes-y_testAll999))
MAE_kt_stdNoRegimes = np.std(abs(kt_predNoRegimes-y_test))
print("The MAE and STD_MAE for KT Persistence with 999s is...")
print(MAE_ktNoRegimes)
print(MAE_kt_stdNoRegimes)
size = len(x)
input_size = x.shape[1]
inp = x
tar = y.reshape(size,1)

norm_inp = nl.tool.Norm(inp)
inp = norm_inp(inp)        
size_test = len(x_test)
x_test_size = x_test.shape[1]
tar_test = y_test.reshape(size_test,1)
 #   size_testAll = len(x_test)
 #   tar_testAll = y_testAll.reshape(size_testAll,1)
norm_inp_test = nl.tool.Norm(x_test)
x_test = norm_inp_test(x_test)

net999 = nl.net.newff([[0, 1]]*input_size,[5, 1])
net999.trainf = nl.train.train_rprop
net999.init()
net999.errorf = nl.error.MSE()
error = net999.train(inp, tar, epochs=500, show=500, goal = 0.00001)
outNoRegimes = net999.sim(inp)
out_testNoRegimes = net999.sim(x_test)
MAENoRegimes = np.mean(abs(out_testNoRegimes-tar_test))
MAEstdNoRegimes = np.std(abs(out_testNoRegimes-tar_test))
print("The MAE and STD_MAE for ANN w/o regime ID and 999s is...")
print(MAENoRegimes)
print(MAEstdNoRegimes)

## Which variables to use for Regime Identification.... ##
titlelist = ["Original idea - from Chapter 2","Original plus only GOES MEASURED variables","Original, GOES measured and derived variables","Use Cloud Type from GOES-East"]
for z in xrange(0,3): # Start with sensitivity study test using all new data....
    if z == 0: ## Original idea - from Chapter 2... 
        kt_var_train = smudTrainTrain[["std_prev1Hr", 'KT_prev15',"NearbyKt_prev15", "NearbyKt_prev15std","KT15_KT30","KT_Slope","cloud_cov_STD", "cloud_cov_SQRD"]]
        kt_var_trainTest = smudTrainTest[["std_prev1Hr", 'KT_prev15',"NearbyKt_prev15", "NearbyKt_prev15std","KT15_KT30","KT_Slope","cloud_cov_STD", "cloud_cov_SQRD"]]  
        kt_var_test = smudTest[["std_prev1Hr", 'KT_prev15',"NearbyKt_prev15", "NearbyKt_prev15std","KT15_KT30","KT_Slope","cloud_cov_STD", "cloud_cov_SQRD"]] 
        ########################## Which Predictor set to use!!!! ##########################################################################
        # This is the original variable list from Chapter 2 of dissertation...
        original_train = smudTrainTrain[["MV72483000_T", "MV72483009_T", "MV72483016_T", "MV72483000_dewpt", "MV72483009_dewpt", "MV72483016_dewpt", "MV72483000_cloud_cov", "MV72483009_cloud_cov", "MV72483016_cloud_cov", "MV72483000_prob_precip01", "MV72483009_prob_precip01", "MV72483016_prob_precip01", "MV72483000_qpf01", "MV72483009_qpf01", "MV72483016_qpf01", "MV72483000_wind_speed", "MV72483009_wind_speed", "MV72483016_wind_speed", "T_Td", "cloud_cov_STD", "cloud_cov_SQRD", "KT_prev60", "KT_prev45", "KT_prev30", "KT_prev15", "KT_Slope",  "std_prev1Hr", "LocalHour", "SineJulianDay",  "KT15_KT30","NearbyKt_prev15", "NearbyKt_prev15std"]]
        original_trainTest = smudTrainTest[["MV72483000_T", "MV72483009_T", "MV72483016_T", "MV72483000_dewpt", "MV72483009_dewpt", "MV72483016_dewpt", "MV72483000_cloud_cov", "MV72483009_cloud_cov", "MV72483016_cloud_cov", "MV72483000_prob_precip01", "MV72483009_prob_precip01", "MV72483016_prob_precip01", "MV72483000_qpf01", "MV72483009_qpf01", "MV72483016_qpf01", "MV72483000_wind_speed", "MV72483009_wind_speed", "MV72483016_wind_speed", "T_Td", "cloud_cov_STD", "cloud_cov_SQRD", "KT_prev60", "KT_prev45", "KT_prev30", "KT_prev15", "KT_Slope",  "std_prev1Hr", "LocalHour", "SineJulianDay",  "KT15_KT30","NearbyKt_prev15", "NearbyKt_prev15std"]]
        original_test = smudTest[["MV72483000_T", "MV72483009_T", "MV72483016_T", "MV72483000_dewpt", "MV72483009_dewpt", "MV72483016_dewpt", "MV72483000_cloud_cov", "MV72483009_cloud_cov", "MV72483016_cloud_cov", "MV72483000_prob_precip01", "MV72483009_prob_precip01", "MV72483016_prob_precip01", "MV72483000_qpf01", "MV72483009_qpf01", "MV72483016_qpf01", "MV72483000_wind_speed", "MV72483009_wind_speed", "MV72483016_wind_speed", "T_Td", "cloud_cov_STD", "cloud_cov_SQRD", "KT_prev60", "KT_prev45", "KT_prev30", "KT_prev15", "KT_Slope",  "std_prev1Hr",  "LocalHour", "SineJulianDay",  "KT15_KT30","NearbyKt_prev15", "NearbyKt_prev15std"]]
        for regimes in range(7,8): # n_Cloud_Regimes_training+1:
            dictOfRegimesTrain, dictOfRegimesTrainTest, dictOfRegimesTest = kMeansRegime(regimes,kt_var_train,kt_var_trainTest,kt_var_test) 
            dictOfInputs = {}
            dictOfTargets = {}
            dictOfResults = {}
            dictOfOutput = {}
            dictOfTarget = {}
            dictOfNets = {}
            dictOfLength = {}
            for i in range(1,regimes+1):
                for j in range(10,15,5):
                    hidden_size = j
                    dictOfInputs['input'+str(i)] = dictOfRegimesTrain['Regime'+str(i)+'_train'][["MV72483000_T", "MV72483009_T", "MV72483016_T", "MV72483000_dewpt", "MV72483009_dewpt", "MV72483016_dewpt", "MV72483000_cloud_cov", "MV72483009_cloud_cov", "MV72483016_cloud_cov", "MV72483000_prob_precip01", "MV72483009_prob_precip01", "MV72483016_prob_precip01", "MV72483000_qpf01", "MV72483009_qpf01", "MV72483016_qpf01", "MV72483000_wind_speed", "MV72483009_wind_speed", "MV72483016_wind_speed", "T_Td", "cloud_cov_STD", "cloud_cov_SQRD","NearbyKt_prev15", "NearbyKt_prev15std","KT_prev60", "KT_prev45", "KT_prev30", "KT_prev15",  "KT_Slope",  "std_prev1Hr", "SineJulianDay", "CosJulianDay", "KT15_KT30"]]
                    dictOfInputs['x_trainTest'+str(i)] = dictOfRegimesTrainTest['Regime'+str(i)+'_traintest'][["MV72483000_T", "MV72483009_T", "MV72483016_T", "MV72483000_dewpt", "MV72483009_dewpt", "MV72483016_dewpt", "MV72483000_cloud_cov", "MV72483009_cloud_cov", "MV72483016_cloud_cov", "MV72483000_prob_precip01", "MV72483009_prob_precip01", "MV72483016_prob_precip01", "MV72483000_qpf01", "MV72483009_qpf01", "MV72483016_qpf01", "MV72483000_wind_speed", "MV72483009_wind_speed", "MV72483016_wind_speed", "T_Td", "cloud_cov_STD", "cloud_cov_SQRD","NearbyKt_prev15", "NearbyKt_prev15std", "KT_prev60", "KT_prev45", "KT_prev30", "KT_prev15",  "KT_Slope",  "std_prev1Hr", "SineJulianDay", "CosJulianDay", "KT15_KT30"]]
                    dictOfInputs['x_test'+str(i)] = dictOfRegimesTest['Regime'+str(i)+'_test'][["MV72483000_T", "MV72483009_T", "MV72483016_T", "MV72483000_dewpt", "MV72483009_dewpt", "MV72483016_dewpt", "MV72483000_cloud_cov", "MV72483009_cloud_cov", "MV72483016_cloud_cov", "MV72483000_prob_precip01", "MV72483009_prob_precip01", "MV72483016_prob_precip01", "MV72483000_qpf01", "MV72483009_qpf01", "MV72483016_qpf01", "MV72483000_wind_speed", "MV72483009_wind_speed", "MV72483016_wind_speed", "T_Td", "cloud_cov_STD", "cloud_cov_SQRD","NearbyKt_prev15", "NearbyKt_prev15std", "KT_prev60", "KT_prev45", "KT_prev30", "KT_prev15",  "KT_Slope",  "std_prev1Hr", "SineJulianDay", "CosJulianDay", "KT15_KT30"]]
                    dictOfTargets['target'+str(i)] = dictOfRegimesTrain['Regime'+str(i)+'_train']["KT_post15"].values   
                    dictOfTargets['y_trainTest'+str(i)] = dictOfRegimesTrainTest['Regime'+str(i)+'_traintest']["KT_post15"].values                
                    dictOfTargets['y_test'+str(i)] = dictOfRegimesTest['Regime'+str(i)+'_test']["KT_post15"].values        
                    dictOfResults['MAE_K'+str(regimes)+"R"+str(i)+"Nodes"+str(j)], dictOfResults['LengthT'+str(regimes)+"R"+str(i)+"Nodes"+str(j)], dictOfNets['NetReg'+str(i)], dictOfOutput['OutReg'+str(i)], dictOfTarget['TargetReg'+str(i)], dictOfResults['MAE_KTper_K'+str(regimes)+"R"+str(i)+"Nodes"+str(j)] = NLabPredict(dictOfInputs['input'+str(i)],dictOfTargets['target'+str(i)],j,dictOfInputs['x_trainTest'+str(i)],dictOfTargets['y_trainTest'+str(i)],i, netALL,z)
            filename = 'OriginalFinal15_minsmud_MAE_K'+str(regimes)+"R"+str(i)+"Nodes"+str(j)+'.csv'
            writer = csv.writer(open(filename, 'wb'))
            for key, val in dictOfResults.items():
                writer.writerow([key, val])

    if z == 1:## Original, GOES measured and derived variables ... 
        kt_var_train = smudTrainTrain[["std_prev1Hr", 'KT_prev15',"NearbyKt_prev15", "NearbyKt_prev15std","KT15_KT30","KT_Slope","cloud_cov_STD", "cloud_cov_SQRD",'CF','Height','Tau','r_e','R_650','R_3.75','T_650','T_3.75']]
        kt_var_trainTest = smudTrainTest[["std_prev1Hr", 'KT_prev15',"NearbyKt_prev15", "NearbyKt_prev15std",'KT15_KT30',"KT_Slope","cloud_cov_STD", "cloud_cov_SQRD",'CF','Height','Tau','r_e','R_650','R_3.75','T_650','T_3.75']]
        kt_var_test = smudTest[["std_prev1Hr", 'KT_prev15',"NearbyKt_prev15", "NearbyKt_prev15std","KT15_KT30","KT_Slope","cloud_cov_STD", "cloud_cov_SQRD",'CF','Height','Tau','r_e','R_650','R_3.75','T_650','T_3.75']]    
        ################ Make ANN test prediction for each regime ###################
        for regimes in range(2,3): # n_Cloud_Regimes_training+1:
            dictOfRegimesTrain, dictOfRegimesTrainTest, dictOfRegimesTest = kMeansRegime(regimes,kt_var_train,kt_var_trainTest,kt_var_test) 
            dictOfInputs = {}
            dictOfTargets = {}
            dictOfResults = {}
            dictOfOutput = {}
            dictOfTarget = {}
            dictOfNets = {}
            dictOfLength = {}
            for i in range(1,regimes+1):
                for j in range(5,10,5):
                    hidden_size = j
                    dictOfInputs['input'+str(i)] = dictOfRegimesTrain['Regime'+str(i)+'_train'][["MV72483000_T", "MV72483009_T", "MV72483016_T", "MV72483000_dewpt", "MV72483009_dewpt", "MV72483016_dewpt", "MV72483000_cloud_cov", "MV72483009_cloud_cov", "MV72483016_cloud_cov", "MV72483000_prob_precip01", "MV72483009_prob_precip01", "MV72483016_prob_precip01", "MV72483000_qpf01", "MV72483009_qpf01", "MV72483016_qpf01", "MV72483000_wind_speed", "MV72483009_wind_speed", "MV72483016_wind_speed", "T_Td", "cloud_cov_STD", "cloud_cov_SQRD","NearbyKt_prev15", "NearbyKt_prev15std", "KT_prev60", "KT_prev45", "KT_prev30", "KT_prev15",  "KT_Slope",  "std_prev1Hr", "SineJulianDay", "CosJulianDay", "KT15_KT30",'SZA','Type','T','Tau','r_e','R_650','R_3.75','T_650','T_3.75']]
                    dictOfInputs['x_trainTest'+str(i)] = dictOfRegimesTrainTest['Regime'+str(i)+'_traintest'][["MV72483000_T", "MV72483009_T", "MV72483016_T", "MV72483000_dewpt", "MV72483009_dewpt", "MV72483016_dewpt", "MV72483000_cloud_cov", "MV72483009_cloud_cov", "MV72483016_cloud_cov", "MV72483000_prob_precip01", "MV72483009_prob_precip01", "MV72483016_prob_precip01", "MV72483000_qpf01", "MV72483009_qpf01", "MV72483016_qpf01", "MV72483000_wind_speed", "MV72483009_wind_speed", "MV72483016_wind_speed", "T_Td", "cloud_cov_STD", "cloud_cov_SQRD","NearbyKt_prev15", "NearbyKt_prev15std", "KT_prev60", "KT_prev45", "KT_prev30", "KT_prev15",  "KT_Slope",  "std_prev1Hr", "SineJulianDay", "CosJulianDay", "KT15_KT30",'SZA','Type','T','Tau','r_e','R_650','R_3.75','T_650','T_3.75']]
                    dictOfInputs['x_test'+str(i)] = dictOfRegimesTest['Regime'+str(i)+'_test'][["MV72483000_T", "MV72483009_T", "MV72483016_T", "MV72483000_dewpt", "MV72483009_dewpt", "MV72483016_dewpt", "MV72483000_cloud_cov", "MV72483009_cloud_cov", "MV72483016_cloud_cov", "MV72483000_prob_precip01", "MV72483009_prob_precip01", "MV72483016_prob_precip01", "MV72483000_qpf01", "MV72483009_qpf01", "MV72483016_qpf01", "MV72483000_wind_speed", "MV72483009_wind_speed", "MV72483016_wind_speed", "T_Td", "cloud_cov_STD", "cloud_cov_SQRD","NearbyKt_prev15", "NearbyKt_prev15std", "KT_prev60", "KT_prev45", "KT_prev30", "KT_prev15",  "KT_Slope",  "std_prev1Hr", "SineJulianDay", "CosJulianDay", "KT15_KT30",'SZA','Type','T','Tau','r_e','R_650','R_3.75','T_650','T_3.75']]
                    dictOfTargets['target'+str(i)] = dictOfRegimesTrain['Regime'+str(i)+'_train']["KT_post15"].values   
                    dictOfTargets['y_trainTest'+str(i)] = dictOfRegimesTrainTest['Regime'+str(i)+'_traintest']["KT_post15"].values                
                    dictOfTargets['y_test'+str(i)] = dictOfRegimesTest['Regime'+str(i)+'_test']["KT_post15"].values        
                    dictOfResults['MAE_K'+str(regimes)+"R"+str(i)+"Nodes"+str(j)], dictOfResults['LengthT'+str(regimes)+"R"+str(i)+"Nodes"+str(j)], dictOfNets['NetReg'+str(i)], dictOfOutput['OutReg'+str(i)], dictOfTarget['TargetReg'+str(i)], dictOfResults['MAE_KTper_K'+str(regimes)+"R"+str(i)+"Nodes"+str(j)] = NLabPredict(dictOfInputs['input'+str(i)],dictOfTargets['target'+str(i)],j,dictOfInputs['x_trainTest'+str(i)],dictOfTargets['y_trainTest'+str(i)],i, netALL,z)
            filename = 'WithGOESFinal15_minsmud_MAE_K'+str(regimes)+"R"+str(i)+"Nodes"+str(j)+'.csv'
            writer = csv.writer(open(filename, 'wb'))
            for key, val in dictOfResults.items():
                writer.writerow([key, val])
 
    if z == 2:
        dictOfRegimesTrain = {}
        dictOfRegimesTrainTest = {}
        dictOfRegimesTest = {}
        regs =  list(smudTrainTrain['Type'].unique())
        print regs
        for reg in regs:
            dictOfRegimesTrain['Regime'+str(reg)+'_train'] = smudTrainTrain[smudTrainTrain.Type == reg]
            dictOfRegimesTrainTest['Regime'+str(reg)+'_traintest'] = smudTrainTest[smudTrainTest.Type == reg]
            dictOfRegimesTest['Regime'+str(reg)+'_test'] = smudTest[smudTest.Type == reg]
            dictOfInputs = {}
            dictOfTargets = {}
            dictOfResults = {}
            dictOfOutput = {}
            dictOfTarget = {}
            dictOfNets = {}
            dictOfLength = {}
            hidden_size = 5
            dictOfInputs['input'+str(reg)] = dictOfRegimesTrain['Regime'+str(reg)+'_train'][["MV72483000_T", "MV72483009_T", "MV72483016_T", "MV72483000_dewpt", "MV72483009_dewpt", "MV72483016_dewpt", "MV72483000_cloud_cov", "MV72483009_cloud_cov", "MV72483016_cloud_cov", "MV72483000_prob_precip01", "MV72483009_prob_precip01", "MV72483016_prob_precip01", "MV72483000_qpf01", "MV72483009_qpf01", "MV72483016_qpf01", "MV72483000_wind_speed", "MV72483009_wind_speed", "MV72483016_wind_speed", "T_Td", "cloud_cov_STD", "cloud_cov_SQRD","NearbyKt_prev15", "NearbyKt_prev15std", "KT_prev60", "KT_prev45", "KT_prev30", "KT_prev15",  "KT_Slope",  "std_prev1Hr", "SineJulianDay", "CosJulianDay", "KT15_KT30",'SZA','T','Tau','r_e','R_650','R_3.75','T_650','T_3.75']]
            dictOfInputs['x_trainTest'+str(reg)] = dictOfRegimesTrainTest['Regime'+str(reg)+'_traintest'][["MV72483000_T", "MV72483009_T", "MV72483016_T", "MV72483000_dewpt", "MV72483009_dewpt", "MV72483016_dewpt", "MV72483000_cloud_cov", "MV72483009_cloud_cov", "MV72483016_cloud_cov", "MV72483000_prob_precip01", "MV72483009_prob_precip01", "MV72483016_prob_precip01", "MV72483000_qpf01", "MV72483009_qpf01", "MV72483016_qpf01", "MV72483000_wind_speed", "MV72483009_wind_speed", "MV72483016_wind_speed", "T_Td", "cloud_cov_STD","NearbyKt_prev15", "NearbyKt_prev15std", "cloud_cov_SQRD", "KT_prev60", "KT_prev45", "KT_prev30", "KT_prev15",  "KT_Slope",  "std_prev1Hr", "SineJulianDay", "CosJulianDay", "KT15_KT30",'SZA','T','Tau','r_e','R_650','R_3.75','T_650','T_3.75']]
            dictOfInputs['x_test'+str(reg)] = dictOfRegimesTest['Regime'+str(reg)+'_test'][["MV72483000_T", "MV72483009_T", "MV72483016_T", "MV72483000_dewpt", "MV72483009_dewpt", "MV72483016_dewpt", "MV72483000_cloud_cov", "MV72483009_cloud_cov", "MV72483016_cloud_cov", "MV72483000_prob_precip01", "MV72483009_prob_precip01", "MV72483016_prob_precip01", "MV72483000_qpf01", "MV72483009_qpf01", "MV72483016_qpf01", "MV72483000_wind_speed", "MV72483009_wind_speed", "MV72483016_wind_speed", "T_Td", "cloud_cov_STD", "cloud_cov_SQRD","NearbyKt_prev15", "NearbyKt_prev15std", "KT_prev60", "KT_prev45", "KT_prev30", "KT_prev15",  "KT_Slope",  "std_prev1Hr", "SineJulianDay", "CosJulianDay", "KT15_KT30",'SZA','T','Tau','r_e','R_650','R_3.75','T_650','T_3.75']]
            dictOfTargets['target'+str(reg)] = dictOfRegimesTrain['Regime'+str(reg)+'_train']["KT_post15"].values   
            dictOfTargets['y_trainTest'+str(reg)] = dictOfRegimesTrainTest['Regime'+str(reg)+'_traintest']["KT_post15"].values                
            dictOfTargets['y_test'+str(reg)] = dictOfRegimesTest['Regime'+str(reg)+'_test']["KT_post15"].values        
            dictOfResults['MAE_CloudTypes_'+"R"+str(reg)], dictOfResults['LengthT'+str(regimes)+"R"+str(reg)], dictOfNets['NetReg'+str(reg)], dictOfOutput['OutReg'+str(reg)], dictOfTarget['TargetReg'+str(reg)], dictOfResults['MAE_KTper_K'+str(reg)+"R"+str(reg)] = NLabPredict(dictOfInputs['input'+str(reg)],dictOfTargets['target'+str(reg)],hidden_size,dictOfInputs['x_trainTest'+str(reg)],dictOfTargets['y_trainTest'+str(reg)],reg, netALL,z)
#                writer = csv.writer(open('LengthsOFKvsNodes.csv','wb'))                
#                for key,val in dictOfLength.items():
#                    writer.writerow([key,val])
            filename = 'FinalCloudType15_minsmud_MAE_K'+str(regimes)+"R"+str(i)+'.csv'
        writer = csv.writer(open(filename, 'wb'))
        for key, val in dictOfResults.items():
            writer.writerow([key, val])
               # print("The MAE for Regime "+str(i)+" with Number of ANN Nodes = "+str(j)+" is... "+ dictOfResults['MAE_R'+str(i)]")
       #     avgMAE_test =  (MAE_R1*LengthT1 + MAE_R2*LengthT2 + MAE_R3*LengthT3 + MAE_R4*LengthT4+ MAE_R5*LengthT5 + MAE_R6*LengthT6 + MAE_R7*LengthT7)/(LengthT1+LengthT2+LengthT3+LengthT4+LengthT5+LengthT6+LengthT7)
       #     print ("The Averaged MAE for all Regimes on the TrainTest data is...")
       #     print (avgMAE_test)
            
### Once again - this was the old way of doing this...
#            if i == 2:
#                hidden_size = j
#                ## Create input and target datasets for training and testing
#                input2 = Regime2_training[["MV72483000_T", "MV72483009_T", "MV72483016_T", "MV72483000_dewpt", "MV72483009_dewpt", "MV72483016_dewpt", "MV72483000_cloud_cov", "MV72483009_cloud_cov", "MV72483016_cloud_cov", "MV72483000_prob_precip01", "MV72483009_prob_precip01", "MV72483016_prob_precip01", "MV72483000_qpf01", "MV72483009_qpf01", "MV72483016_qpf01", "MV72483000_wind_speed", "MV72483009_wind_speed", "MV72483016_wind_speed", "T_Td", "cloud_cov_STD", "cloud_cov_SQRD", "KT_prev60", "KT_prev45", "KT_prev30", "KT_prev15",  "KT_Slope",  "std_prev1Hr", "NearbyKt_prev15", "NearbyKt_prev15std", "LocalHour", "SineJulianDay",  "KT15_KT30"]]
#                target2 = Regime2_training['KT_post15'].values
#                x_test2 = Regime2_test[["MV72483000_T", "MV72483009_T", "MV72483016_T", "MV72483000_dewpt", "MV72483009_dewpt", "MV72483016_dewpt", "MV72483000_cloud_cov", "MV72483009_cloud_cov", "MV72483016_cloud_cov", "MV72483000_prob_precip01", "MV72483009_prob_precip01", "MV72483016_prob_precip01", "MV72483000_qpf01", "MV72483009_qpf01", "MV72483016_qpf01", "MV72483000_wind_speed", "MV72483009_wind_speed", "MV72483016_wind_speed", "T_Td", "cloud_cov_STD", "cloud_cov_SQRD", "KT_prev60", "KT_prev45", "KT_prev30", "KT_prev15",  "KT_Slope",  "std_prev1Hr", "NearbyKt_prev15", "NearbyKt_prev15std", "LocalHour", "SineJulianDay",  "KT15_KT30"]]
#                y_test2 = Regime2_test['KT_post15'].values
#                MAE_R2, LengthT2, NetReg2, OutReg2, TargetReg2 = NLabPredict(input2,target2,x_test2,y_test2,i,netALL)

#    avgMAE_test =  (MAE_R1*LengthT1 + MAE_R2*LengthT2 + MAE_R3*LengthT3 + MAE_R4*LengthT4+ MAE_R5*LengthT5 + MAE_R6*LengthT6 + MAE_R7*LengthT7)/(LengthT1+LengthT2+LengthT3+LengthT4+LengthT5+LengthT6+LengthT7)
#    print ("The Averaged MAE for all Regimes on the test data is...")
#    print (avgMAE_test)
    
#       Cloud_Regimes_idx_TrainTrain = smudTrainTrain['Cloud_Reg_Code'].values
 #       Cloud_Regimes_idx_Test = smudTest['Cloud_Reg_Code'].values
 #       Cloud_Regimes_idx_TrainTest = smudTrainTest['Cloud_Reg_Code'].values
 #       n_Cloud_Regimes_TrainTrain = len(smudTrainTrain.KMeans_Cluster.unique())
 #       n_Cloud_Regimes_Test = len(smudTest.KMeans_Cluster.unique())
 #       n_Cloud_Regimes_TrainTest = len(smudTrainTest.KMeans_Cluster.unique())   

######### MISC PLOTS .... ############
#### These plots that are commented out are used to show predictor relationships amoung regime classifications...
#    ############## PLOT KT30 VS KT15 for CLUSTER ASSIGNMENTS #################
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    xmin = np.min(X[:,1])
#    xmax = np.max(X[:,1])
#    plt.xlabel('Predictor - Clearness Index Previous 30-Min Avg')
#    plt.ylabel('Last 15-Min Avg Clearness Index')
#    tt = plt.title('K-Means Clustering: Training Data Classification')
#    plt.axis([xmin, xmax, 0, 1])
#    clr = ['b','g','r','c','m','y','k']
#    for i in range(0,7):
#        ind = (assignment==i)
#        ii = i + 1
#        ax.scatter(X[ind,1],X[ind,2], s=30, c=clr[i], label='Cluster %d'%ii)
#        # Shrink current axis by 20%
#        #box = ax.get_position()
#        #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#        # Put a legend to the right of the current axis
#        #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#    plt.savefig('KMeans3_KT30_KtVars_normalized.png', bbox_inches='tight', dpi=300)
# #   pyplot.show()
#    ############## PLOT KT45 VS KT15 for CLUSTER ASSIGNMENTS #################
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    xmin = np.min(X[:,0])
#    xmax = np.max(X[:,0])
#    plt.xlabel('Predictor - Clearness Index Previous 45-Min Avg')
#    plt.ylabel('Last 15-Min Avg Clearness Index')
#    tt = plt.title('K-Means Clustering Predictor Comparison')
#    plt.axis([xmin, xmax, 0, 1])
#    for i in range(0,7):
#        ind = (assignment==i)
#        ii = i + 1
#        ax.scatter(X[ind,0],X[ind,2], s=30, c=clr[i], label='Cluster %d'%ii)
#        # Shrink current axis by 20%
#   #     box = ax.get_position()
#   #     ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#        # Put a legend to the right of the current axis
#   #     ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#        plt.savefig('KMeans3_KT45_KtVars_normalized.png', bbox_inches='tight', dpi=300)
##        pyplot.show()
#
############# FOR TESTING DATA.........####################################
############### PLOT KT30 VS KT15 for CLUSTER ASSIGNMENTS ################# 
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    xmin = np.min(X_test[:,1])
#    xmax = np.max(X_test[:,1])                                                                                                      
#    plt.xlabel('Predictor - Clearness Index Previous 15 to 30-Min Avg')
#    plt.ylabel('Predictor - Clearness Index Previous 0 to 15-Min Avg')
#    tt = plt.title('K-Means Clustering: Test Data Classification')
#    plt.axis([xmin, xmax, 0, 1])
#    for i in range(0,7):
#        ind = (test_assignment==i)
#        ii = i + 1
#        ax.scatter(X_test[ind,1],X_test[ind,2], s=30, c=clr[i], label='Cluster %d'%ii)
## Shrink current axis by 20%
##box = ax.get_position()
##ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
## Put a legend to the right of the current axis
##ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#    plt.savefig('KMeans3_KT30_KtVars.png', bbox_inches='tight', dpi=300)                                                                                                            
##    pyplot.show()
############### PLOT KT45 VS KT15 for CLUSTER ASSIGNMENTS #################
#    fig = plt.figure()
#    ax = fig.add_subplot(111)  
#    xmin = np.min(X_test[:,0])
#    xmax = np.max(X_test[:,0])                                                                                                          
#    plt.xlabel('Predictor - Clearness Index Previous 45-Min Avg')
#    plt.ylabel('Last 15-Min Avg Clearness Index')
#    tt = plt.title('K-Means Clustering Predictor Comparison: Test Data')
#    plt.axis([xmin, xmax, 0, 1])
#    for i in range(0,7):
#        ind = (test_assignment==i)
#        ii = i + 1
#        ax.scatter(X_test[ind,0],X_test[ind,2], s=30, c=clr[i], label='Cluster %d'%ii)
#    # Shrink current axis by 20%
# #   box = ax.get_position()
# #   ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# #   # Put a legend to the right of the current axis
# #   ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#    plt.savefig('KMeans3_KT45_KtVars.png', bbox_inches='tight', dpi=300)                                                                                                             
##    pyplot.show()





###################### PLOT THE VARIABILITY FOR EACH REGIME #################
### This is only necessary for analysis, not real-time implementation and is therefore commented out...
#    clr = ['b','g','r']#,'c','m','y','k']
#    # Set axis for plotting...
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    # K-Means Regime 3
#    num_bins = 20
#    ax3 = fig.add_subplot(333)
#    setp( ax3.get_xticklabels(), visible=False)
#    setp( ax3.get_yticklabels(), visible=False)
#    plt.hist(var_to_plot_Regime3,num_bins,normed=True,facecolor='r')
#    plt.grid(True)
#    # K-Means Regime 1
#    num_bins = 20
#    ax1 = fig.add_subplot(331,  sharex=ax3, sharey=ax3)
#    setp( ax1.get_xticklabels(), visible=False)
#    setp( ax1.get_yticklabels(), visible=False)
#    plt.hist(var_to_plot_Regime1,num_bins,normed=True,facecolor='b',alpha=0.5)
#    plt.grid(True)
#    # K-Means Regime 2
#    num_bins = 20
#    ax2 = fig.add_subplot(332,  sharex=ax3, sharey=ax3)
#    setp( ax2.get_xticklabels(), visible=False)
#    setp( ax2.get_yticklabels(), visible=False)
#    plt.hist(var_to_plot_Regime2,num_bins,normed=True,facecolor='g',alpha=0.5)
#    plt.grid(True)
#    # K-Means Regime 4
#    num_bins = 20
#    ax4 = fig.add_subplot(334,  sharex=ax3, sharey=ax3)
#    setp( ax4.get_xticklabels(), visible=False)
#    setp( ax4.get_yticklabels(), visible=False)
#    plt.hist(var_to_plot_Regime4,num_bins,normed=True,facecolor='c',alpha=0.5)
#    plt.grid(True)
#    # K-Means Regime 5
#    num_bins = 20
#    ax5 = fig.add_subplot(335,  sharex=ax3, sharey=ax3)
#    setp( ax5.get_xticklabels(), visible=False)
#    setp( ax5.get_yticklabels(), visible=False)
#    plt.hist(var_to_plot_Regime5,num_bins,normed=True,facecolor='m',alpha=0.5)
#    plt.grid(True)
#    # K-Means Regime 6
#    num_bins = 20
#    ax6 = fig.add_subplot(336,  sharex=ax3, sharey=ax3)
#    setp( ax6.get_xticklabels(), visible=False)
#    setp( ax6.get_yticklabels(), visible=False)
#    plt.hist(var_to_plot_Regime6,num_bins,normed=True,facecolor='y',alpha=0.5)
#    plt.grid(True)
#    # K-Means Regime 7
#    num_bins = 20
#    ax7 = fig.add_subplot(338,  sharex=ax3, sharey=ax3)
#    ax7.xaxis.set_ticks(np.arange(0.0, 0.4, 0.1))
#    setp( ax7.get_xticklabels(), fontsize=8)
#    setp( ax7.get_yticklabels(), fontsize=8)
#    plt.hist(var_to_plot_Regime7,num_bins,normed=True,facecolor='k',alpha=0.5)
#    plt.grid(True)
#    # Turn off axis lines and ticks of the big subplot
#    ax.spines['top'].set_color('none')
#    ax.spines['bottom'].set_color('none')
#    ax.spines['left'].set_color('none')
#    ax.spines['right'].set_color('none')
#    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
#    ax.set_xlabel('Clearness Index Variability (Standard Deviation of Kt)')
#    ax.set_ylabel('Percentage of Data')
#    plt.suptitle('Clearness Index Variability for each Regime')
#    plt.savefig('KMeans7_KT_STD_KtRegimeSubplot_colors_NoGOESinKMeans.png', bbox_inches='tight', dpi=300)
#    plt.show()  



## KMEANS PLOTTING>>>
    # Scipy.cluster.vq.kmeans
    #KM = [kmeans(X_normalized,k) for k in K ] # Apply kmeans 1 to 10
    #centroids = [cent for (cent,var) in KM] # Cluster centroids
    #D_k = [cdist(X_normalized,cent,'euclidean') for cent in centroids]
    #cIdx = [np.argmin(D,axis=1) for D in D_k]
    #dist = [np.min(D,axis=1) for D in D_k]
    #avgWithinSS = [sum(d)/X_normalized.shape[0] for d in dist]
#    kIdx = 6
#    # Plot Elbow Curve for Average Within-Cluster Sum-of-Squares
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    ax.plot(K,avgWithinSS,'b*-')
#    ax.plot(K[kIdx],avgWithinSS[kIdx],marker='o',markersize=12, markeredgewidth=2,markeredgecolor='r',markerfacecolor='None')
#    plt.grid(True)
#    plt.xlabel('Number of Clusters')
#    plt.ylabel('Average Within-Cluster Sum of Squares')
#    ax.set_title(titlelist[z])
#    fig.savefig('KMeans_Elbow_noGOES'+str(z)+'.png')



##############  Test for optimal epochs ....
#netALL1 = nl.net.newff([[0, 1]]*input_size,[10, 1])
#    netALL1.trainf = nl.train.train_rprop
#    netALL1.init()
#    netALL1.errorf = nl.error.MSE()
#    error = netALL1.train(inp, tar, epochs=100, show=100, goal = 0.0001)
#    netALL2 = nl.net.newff([[0, 1]]*input_size,[10, 1])
#    netALL2.trainf = nl.train.train_rprop
#    netALL2.init()
#    netALL2.errorf = nl.error.MSE()    
#    error = netALL2.train(inp, tar, epochs=250, show=100, goal = 0.0001)
#    netALL3 = nl.net.newff([[0, 1]]*input_size,[10, 1])
#    netALL3.trainf = nl.train.train_rprop
#    netALL3.init()
#    netALL3.errorf = nl.error.MSE()    
#    error = netALL3.train(inp, tar, epochs=500, show=100, goal = 0.0001)
#    netALL4 = nl.net.newff([[0, 1]]*input_size,[10, 1])
#    netALL4.trainf = nl.train.train_rprop
#    netALL4.init()
#    netALL4.errorf = nl.error.MSE() 
#    error = netALL4.train(inp, tar, epochs=1000, show=100, goal = 0.0001)
#    netALL5 = nl.net.newff([[0, 1]]*input_size,[10, 1])
#    netALL5.trainf = nl.train.train_rprop
#    netALL5.init()
#    netALL5.errorf = nl.error.MSE()     
#    error = netALL5.train(inp, tar, epochs=2000, show=100, goal = 0.0001)
#
#    ### TEST OF OPTIMAL CONFIG FOR ANN>.....
#    out_test1 = netALL1.sim(x_testAll)
#    MAE100 = np.mean(abs(out_test1-tar_testAll))
#    print ("MAE for 100 epochs...")
#    print (MAE100)
#     ### TEST OF OPTIMAL CONFIG FOR ANN>.....
#    out_test2 = netALL2.sim(x_testAll)
#    MAE250 = np.mean(abs(out_test2-tar_testAll))
#    print ("MAE for 250 epochs...")
#    print (MAE250)
#     ### TEST OF OPTIMAL CONFIG FOR ANN>.....
#    out_test3 = netALL3.sim(x_testAll)
#    MAE500 = np.mean(abs(out_test3-tar_testAll))
#    print ("MAE for 500 epochs...")
#    print (MAE500)
#     ### TEST OF OPTIMAL CONFIG FOR ANN>.....
#    out_test4 = netALL4.sim(x_testAll)
#    MAE1000 = np.mean(abs(out_test4-tar_testAll))
#    print ("MAE for 1000 epochs...")
#    print (MAE1000)
#     ### TEST OF OPTIMAL CONFIG FOR ANN>.....
#    out_test5 = netALL5.sim(x_testAll)
#    MAE2000 = np.mean(abs(out_test5-tar_testAll))
#    print ("MAE for 2000 epochs...")
#    print (MAE2000)
    