# RDANN
Regime-Dependent Artificial Neural Network
**Note: the Python scripts were run using Python 2.7.10 | Anaconda 2.1.0 (x86_64)  Data Pre-Processing: There are several derived variables needed for the RDANN methods. 

StatCast – Regime-Dependent ANN (RDANN) Documentation
This document describes the data, the model configuration process, and the steps for real-time implementation.

DATA
The first step in implementing StatCast-RDANN at a new location is to build the dataset for training the model. The model requires two sets of inputs: irradiance observations and weather observations from the nearest METAR sites.

Irradiance observations need to be averaged into 15-minute intervals leading into the forecast initialization time.  There should be four 15-minute intervals, which are then named in the file (DV SMUD000** prev45-60avg_GHI, DV SMUD000** prev30-45avg_GHI, DV SMUD000** prev15-30avg_GHI, DV SMUD000** prev0-15avg_GHI). Note that the files are named based on their Site IDs. In addition to the past hour of 15-minute averages, the following three hours must be included in the training data: SMUD Obs Data Location on Minivet - /d2/ldm/data/solar_obs/smud_67_74/15min_avg/netcdf/

Note that if possible, adding in a regional average of the irradiance and a regional irradiance variability (standard deviation) would be predictors to add that would improve the forecast skill of the StatCast-RDANN model. This was not done in the initial StatCast-RDANN implementation at SMUD.

METAR observations for the last available observations (i.e. the hour preceding forecast initialization) are required.  The weather variables needed are Temperature, Dewpoint Temperature, Cloud Cover, Probability of Precipitation, QPF, and Wind Speed: METAR Obs Data Location on Minivet - /var/autofs/mnt/final_fcst

The irradiance observations and weather observations are next joined in a dataset with additional Time Variables (labeled with prefix TV in the .names files) UnixUtcTime, LocalTime, UtcTime, UtcYear, UtcMonth, UtcHour, UtcDayOfWeek, and JulianDay. Note that only the UtcHour and Julian day are used in the prediction, the other time variables are still included in the dataset to match the configuration of the StatCast-Cubist operational implementations.

Quality control the data by removing any cases with missing data or irradiance values above 1362 W/M2. Note that removing the entire row when any of the columns are missing is best so that the training data will have only complete cases.

If the data are all for the same region (SMUD, BNL, SCE) I would recommend aggregating the sites together so that one RDANN model is built for each lead time, rather than one for each lead time AND one for each specific site.  The balance is the amount of training data you gain by aggregating the data together versus the minor differences among the different irradiance measuring sites.  The differences will be minimal if the sites are in close proximity. 

II.	MODEL CONFIGURATION PROCESS – Preprocess.py
Two functions – split data and compute TOA irradiance
First step reads in data that is saved per station (SMUD00067.data…SMUD00074.data)
This data has METAR observations, time data, and GHI past and future (predictand) data.
Next steps removes all rows with NaNs, missing data, and removes LocalTime, UtcTime, and UtcDayOfWeek.
Next step is more quality control: removed all rows where the clearness index is above 1.0 or below 0.0

Derived Variables: Averages, Temperature, Dewpoint Temperature, Wind Speed, Prob Precip, QPF, Cloud Cover
Next step is more quality control: remove bad METAR data values

More Derived Variables: Dewpoint Depression, Cloud Cover Variability, Cloud Cover Squared

Compute TOA irradiance for all time-steps and all seven SMUD sites

Even More Derived Variables: Slope and correlation coeff, Compute temporal variability (Stdev)

Predictor: Stdev prev hour
Predictands: Stdev next hour, 60-120min, and 120-180min
Delete all rows with NaNs (i.e. when TOA is predicted to be 0)
Delete all rows with Kt above 1.0 or below 0.0
Compute spatial variability

Predictor: Compute spatial stdev of other 7 sites last observations (T-15min)
Predictands: Compute spatial stdev of other 7 sites at 15-min interval ending at T+15min, T+60min, T+120min, and T+180min
Compute spatial 15-min average Kt not including forecast site.
Convert UtcHour to LocalHour
Compute both the Sine of the Julian Day and Cosine of the Julian Day
Compute slope multiplied by R^2
Compute last change in Kt (Kt_15min-Kt_30min)
Last quality control to remove rows with NaNs
Split each site’s data into three (2/3 for training, 1/3 for testing)
Write each site to .csv file independently. 

Imports Required for StatCast-RDANN
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.cm as cm
import scipy
from pylab import *
from sklearn import preprocessing
from scipy.cluster.vq import kmeans,vq
from scipy.spatial.distance import *
import cPickle as pickle
from math import sqrt
import neurolab as nl


 Python Scripts (Prediction Methods): 1.) FinalSMUD.py …Use to make predictions for SMUD 2.) FInalBNL.py
…Use to make predictions for BNL 3.) StdSMUD.py …Use to make variability predictions 4.)SMUDatBNL.py …Use to make predictions at BNL after training at SMUD
4.) CaseStudy.py …Still in development.  Attempt to make predictions for specific days.  Data Files: 1.) SMUDTrain / SMUDTrainTest / SMUDTest  …Files used to train and test the RDANN methods. …Files listed with “NO999s” means that all missing data has been removed.
2.) CaseStudyTest/SMUDTestCaseStudies/SMUDTrainCaseStudies/SMUDTrainTestCaseStudies
…Case study days set up for CaseStudy.py - still in development.
