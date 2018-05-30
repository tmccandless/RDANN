#!/usr/bin/env python

# Create datasets
import os
import fileinput
import string
import numpy as np
import csv
import sys
import time
import math
import matplotlib.pyplot as plt
import datetime
from math import sqrt
import itertools
import math, calendar
from math import pi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels as sm
import scipy
import pybrain
import cPickle as pickle
from math import sqrt
from pybrain.datasets.supervised import SupervisedDataSet as SDS
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pylab import *
#from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans,vq
from scipy.spatial.distance import *
from scipy import stats


def splitData(df, trainPerc=0.34, cvPerc=0.33, testPerc=0.33):
    """
    return: training, cv, test
            (as pandas dataframes)
    params:
              df: pandas dataframe
       trainPerc: float | percentage of data for trainin set (default=0.6
          cvPerc: float | percentage of data for cross validation set (default=0.2)
        testPerc: float | percentage of data for test set (default=0.2)
                  (trainPerc + cvPerc + testPerc must equal 1.0)
    """
    assert trainPerc + cvPerc + testPerc == 1.0

    # create random list of indices
    from random import shuffle
    N = len(df)
    l = range(N)
    shuffle(l)

    # get splitting indicies
    trainLen = int(N*trainPerc)
    cvLen    = int(N*cvPerc)
    testLen  = int(N*testPerc)

    # get training, cv, and test sets
    training = df.ix[l[:trainLen]]
    cv       = df.ix[l[trainLen:trainLen+cvLen]]
    test     = df.ix[l[trainLen+cvLen:]]

    #print len(cl), len(training), len(cv), len(test)

    return training, cv, test

################# Extra-terrestrial Irradiance ########################################
def TOA(lat,lon,alt,day,month,year,ToD):
    

    # Solar constant for the mean distance between the Earth and sun #######
    sol_const = 1367.8
    ########################################################################
    # GEO Parameters 
    #lat = 23
    #lon = 11
    #alt = 0
    ########################################################################
    # Time and Offsets
    #day             =               22
    #month           =               06
    #year            =               2011
    #ToD             =               12
    tz_off_deg      =               lon #originally was written as (0 + lon)
    if (month > 3 and month < 11):
        dst_off = 8
    if (month < 3 or month > 11):
        dst_off = 7
    if month == 3:
        if day > 8:
            dst_off = 8
        else:
            dst_off = 7
    if month == 11:
        if day == 1:
            dst_off = 8
        else:
            dst_off = 7
            
    #dst_off         =               8
    ########################################################################
    # Atmospheric Parameters
    # air temperature
    atm_temp        =               25.0    
    # relative humidity 
    atm_hum         =               20.0    # Default
    # turbidity coefficient - 0 < tc < 1.0 - where tc = 1.0 for clean air
    # and tc < 0.5 for extremely turbid, dusty or polluted air 
    atm_tc          =               0.8     # Default
    ########################################################################
    ##  MAIN
    ########################################################################
    # get Julian Day (Day of Year)
    if calendar.isleap(year):
        # Leap year, 366 days
        lMonth = [0,31,60,91,121,152,182,213,244,274,305,335,366]
    else:
        # Normal year, 365 days
        lMonth = [0,31,59,90,120,151,181,212,243,273,304,334,365]
    DoY = lMonth[month-1] + day
    ## print "--------------------------------------------------------------"
    ## print "%d.%d.%d | %d | %f | " % (day, month, year, DoY, ToD, ) 
    ## print "--------------------------------------------------------------" 
    ## print "Solar Constant                               : %s" % sol_const
    ## print "Atmospheric turbidity coefficient            : %s" % atm_tc 
    ## print "--------------------------------------------------------------" 
    # inverse relative distance factor for distance between Earth and Sun ##
    sun_rel_dist_f  = 1.0/(1.0-9.464e-4*math.sin(DoY)-                      \
                    + 0.01671*math.cos(DoY)-                                \
                    + 1.489e-4*math.cos(2.0*DoY)-2.917e-5*math.sin(3.0*DoY)-\
                    + 3.438e-4*math.cos(4.0*DoY))**2 
    ## print "Inverse relative distance factor             : %s" % sun_rel_dist_f
    # solar declination ####################################################
    sun_decl        = (math.asin(0.39785*(math.sin(((278.97+(0.9856*DoY))   \
                    + (1.9165*(math.sin((356.6+(0.9856*DoY))                \
                    * (math.pi/180)))))*(math.pi/180))))*180)               \
                    / math.pi

    # equation of time #####################################################
    # (More info on http://www.srrb.noaa.gov/highlights/sunrise/azel.html)
    eqt             = (((5.0323-(430.847*math.cos((((2*math.pi)*DoY)/366)+4.8718)))\
                    + (12.5024*(math.cos(2*((((2*math.pi)*DoY)/366)+4.8718))))\
                    + (18.25*(math.cos(3*((((2*math.pi)*DoY)/366)+4.8718))))\
                    - (100.976*(math.sin((((2*math.pi)*DoY)/366)+4.8718))))\
                    + (595.275*(math.sin(2*((((2*math.pi)*DoY)/366)+4.8718))))\
                    + (3.6858*(math.sin(3*((((2*math.pi)*DoY)/366)+4.871))))\
                    - (12.47*(math.sin(4*((((2*math.pi)*DoY)/366)+4.8718)))))\
                    / 60
    ## print "Equation of time                             : %s min" % eqt
    # time of solar noon ###################################################
    sol_noon        = ((12+dst_off)-(eqt/60))-((tz_off_deg-lon)/15)
    ## print "Solar Noon                                   : %s " % sol_noon
    # solar zenith angle in DEG ############################################
    sol_zen         = math.acos(((math.sin(lat*(math.pi/180)))              \
                    * (math.sin(sun_decl*(math.pi/180))))                   \
                    + (((math.cos(lat*((math.pi/180))))                     \
                    * (math.cos(sun_decl*(math.pi/180))))                   \
                    * (math.cos((ToD-sol_noon)*(math.pi/12)))))             \
                    * (180/math.pi)
    # in extreme latitude, values over 90 may occurs.
    #if sol_zen > 90: 
    # barometric pressure of the measurement site
    # (this should be replaced by the real measured value) in kPa
    atm_press       = 101.325                                               \
                    * math.pow(((288-(0.0065*(alt-0)))/288)                 \
                    , (9.80665/(0.0065*287)))
    atm_press=100.5 
    ## print "Estimated Barometric Pressure at site        : %s kPa" % atm_press
    # Estimated air vapor pressure in kPa ###################################
    atm_vapor_press = (0.61121*math.exp((17.502*atm_temp)                   \
                    / (240.97+atm_temp)))                                   \
                    * (atm_hum/100) 
    ## print "Estimated Vapor Pressure at site             : %s kPa" % atm_vapor_press
    # extraterrestrial radiation in W/m2 ###################################
    toa  = (sol_const*sun_rel_dist_f)                            \
                    * (math.cos(sol_zen*(math.pi/180))) 

    if toa < 0:
        toa = 0
        
    return toa    


# Note that this is the first code that is actually run.  Everything previous was definitions until this section of script begins.
## Read Data... 
header_row=['UnixUtcTime','LocalTime','UtcTime','UtcYear','UtcMonth','UtcDay','UtcHour','UtcDayOfWeek','JulianDay','MV72483000_T','MV72483009_T','MV72483016_T','MV72483000_dewpt','MV72483009_dewpt','MV72483016_dewpt','MV72483000_cloud_cov','MV72483009_cloud_cov','MV72483016_cloud_cov','MV72483000_prob_precip01','MV72483009_prob_precip01','MV72483016_prob_precip01','MV72483000_qpf01','MV72483009_qpf01','MV72483016_qpf01','MV72483000_wind_speed','MV72483009_wind_speed','MV72483016_wind_speed','GHI_prev75','GHI_prev60','GHI_prev45','GHI_prev30','GHI_prev15','GHI_post15','GHI_post30','GHI_post45','GHI_post60','GHI_post75','GHI_post90','GHI_post105','GHI_post120','GHI_post135','GHI_post150','GHI_post165','GHI_post180']
data67 = pd.read_csv('SMUD00067.data',names=header_row)
## Load data for site 68
header_row=['UnixUtcTime','LocalTime','UtcTime','UtcYear','UtcMonth','UtcDay','UtcHour','UtcDayOfWeek','JulianDay','MV72483000_T','MV72483009_T','MV72483016_T','MV72483000_dewpt','MV72483009_dewpt','MV72483016_dewpt','MV72483000_cloud_cov','MV72483009_cloud_cov','MV72483016_cloud_cov','MV72483000_prob_precip01','MV72483009_prob_precip01','MV72483016_prob_precip01','MV72483000_qpf01','MV72483009_qpf01','MV72483016_qpf01','MV72483000_wind_speed','MV72483009_wind_speed','MV72483016_wind_speed','GHI_prev75','GHI_prev60','GHI_prev45','GHI_prev30','GHI_prev15','GHI_post15','GHI_post30','GHI_post45','GHI_post60','GHI_post75','GHI_post90','GHI_post105','GHI_post120','GHI_post135','GHI_post150','GHI_post165','GHI_post180']
data68 = pd.read_csv('SMUD00068.data',names=header_row)
## Load data for site 69
header_row=['UnixUtcTime','LocalTime','UtcTime','UtcYear','UtcMonth','UtcDay','UtcHour','UtcDayOfWeek','JulianDay','MV72483000_T','MV72483009_T','MV72483016_T','MV72483000_dewpt','MV72483009_dewpt','MV72483016_dewpt','MV72483000_cloud_cov','MV72483009_cloud_cov','MV72483016_cloud_cov','MV72483000_prob_precip01','MV72483009_prob_precip01','MV72483016_prob_precip01','MV72483000_qpf01','MV72483009_qpf01','MV72483016_qpf01','MV72483000_wind_speed','MV72483009_wind_speed','MV72483016_wind_speed','GHI_prev75','GHI_prev60','GHI_prev45','GHI_prev30','GHI_prev15','GHI_post15','GHI_post30','GHI_post45','GHI_post60','GHI_post75','GHI_post90','GHI_post105','GHI_post120','GHI_post135','GHI_post150','GHI_post165','GHI_post180']
data69 = pd.read_csv('SMUD00069.data',names=header_row)
## Load data for site 70
header_row=['UnixUtcTime','LocalTime','UtcTime','UtcYear','UtcMonth','UtcDay','UtcHour','UtcDayOfWeek','JulianDay','MV72483000_T','MV72483009_T','MV72483016_T','MV72483000_dewpt','MV72483009_dewpt','MV72483016_dewpt','MV72483000_cloud_cov','MV72483009_cloud_cov','MV72483016_cloud_cov','MV72483000_prob_precip01','MV72483009_prob_precip01','MV72483016_prob_precip01','MV72483000_qpf01','MV72483009_qpf01','MV72483016_qpf01','MV72483000_wind_speed','MV72483009_wind_speed','MV72483016_wind_speed','GHI_prev75','GHI_prev60','GHI_prev45','GHI_prev30','GHI_prev15','GHI_post15','GHI_post30','GHI_post45','GHI_post60','GHI_post75','GHI_post90','GHI_post105','GHI_post120','GHI_post135','GHI_post150','GHI_post165','GHI_post180']
data70 = pd.read_csv('SMUD00070.data',names=header_row)
## Load data for site 71
header_row=['UnixUtcTime','LocalTime','UtcTime','UtcYear','UtcMonth','UtcDay','UtcHour','UtcDayOfWeek','JulianDay','MV72483000_T','MV72483009_T','MV72483016_T','MV72483000_dewpt','MV72483009_dewpt','MV72483016_dewpt','MV72483000_cloud_cov','MV72483009_cloud_cov','MV72483016_cloud_cov','MV72483000_prob_precip01','MV72483009_prob_precip01','MV72483016_prob_precip01','MV72483000_qpf01','MV72483009_qpf01','MV72483016_qpf01','MV72483000_wind_speed','MV72483009_wind_speed','MV72483016_wind_speed','GHI_prev75','GHI_prev60','GHI_prev45','GHI_prev30','GHI_prev15','GHI_post15','GHI_post30','GHI_post45','GHI_post60','GHI_post75','GHI_post90','GHI_post105','GHI_post120','GHI_post135','GHI_post150','GHI_post165','GHI_post180']
data71 = pd.read_csv('SMUD00071.data',names=header_row)
## Load data for site 72
header_row=['UnixUtcTime','LocalTime','UtcTime','UtcYear','UtcMonth','UtcDay','UtcHour','UtcDayOfWeek','JulianDay','MV72483000_T','MV72483009_T','MV72483016_T','MV72483000_dewpt','MV72483009_dewpt','MV72483016_dewpt','MV72483000_cloud_cov','MV72483009_cloud_cov','MV72483016_cloud_cov','MV72483000_prob_precip01','MV72483009_prob_precip01','MV72483016_prob_precip01','MV72483000_qpf01','MV72483009_qpf01','MV72483016_qpf01','MV72483000_wind_speed','MV72483009_wind_speed','MV72483016_wind_speed','GHI_prev75','GHI_prev60','GHI_prev45','GHI_prev30','GHI_prev15','GHI_post15','GHI_post30','GHI_post45','GHI_post60','GHI_post75','GHI_post90','GHI_post105','GHI_post120','GHI_post135','GHI_post150','GHI_post165','GHI_post180']
data72 = pd.read_csv('SMUD00072.data',names=header_row)
## Load data for site 73
header_row=['UnixUtcTime','LocalTime','UtcTime','UtcYear','UtcMonth','UtcDay','UtcHour','UtcDayOfWeek','JulianDay','MV72483000_T','MV72483009_T','MV72483016_T','MV72483000_dewpt','MV72483009_dewpt','MV72483016_dewpt','MV72483000_cloud_cov','MV72483009_cloud_cov','MV72483016_cloud_cov','MV72483000_prob_precip01','MV72483009_prob_precip01','MV72483016_prob_precip01','MV72483000_qpf01','MV72483009_qpf01','MV72483016_qpf01','MV72483000_wind_speed','MV72483009_wind_speed','MV72483016_wind_speed','GHI_prev75','GHI_prev60','GHI_prev45','GHI_prev30','GHI_prev15','GHI_post15','GHI_post30','GHI_post45','GHI_post60','GHI_post75','GHI_post90','GHI_post105','GHI_post120','GHI_post135','GHI_post150','GHI_post165','GHI_post180']
data73 = pd.read_csv('SMUD00073.data',names=header_row)
## Load data for site 74
header_row=['UnixUtcTime','LocalTime','UtcTime','UtcYear','UtcMonth','UtcDay','UtcHour','UtcDayOfWeek','JulianDay','MV72483000_T','MV72483009_T','MV72483016_T','MV72483000_dewpt','MV72483009_dewpt','MV72483016_dewpt','MV72483000_cloud_cov','MV72483009_cloud_cov','MV72483016_cloud_cov','MV72483000_prob_precip01','MV72483009_prob_precip01','MV72483016_prob_precip01','MV72483000_qpf01','MV72483009_qpf01','MV72483016_qpf01','MV72483000_wind_speed','MV72483009_wind_speed','MV72483016_wind_speed','GHI_prev75','GHI_prev60','GHI_prev45','GHI_prev30','GHI_prev15','GHI_post15','GHI_post30','GHI_post45','GHI_post60','GHI_post75','GHI_post90','GHI_post105','GHI_post120','GHI_post135','GHI_post150','GHI_post165','GHI_post180']
data74 = pd.read_csv('SMUD00074.data',names=header_row)

###### Remove all rows with NaNs in any of the columns, after removing LocalTime, UtcTime, and UtcDayOfWeek ####
data67 = data67.drop(['LocalTime','UtcTime','UtcDayOfWeek'],axis=1)
data68 = data68.drop(['LocalTime','UtcTime','UtcDayOfWeek'],axis=1)
data69 = data69.drop(['LocalTime','UtcTime','UtcDayOfWeek'],axis=1)
data70 = data70.drop(['LocalTime','UtcTime','UtcDayOfWeek'],axis=1)
data71 = data71.drop(['LocalTime','UtcTime','UtcDayOfWeek'],axis=1)
data72 = data72.drop(['LocalTime','UtcTime','UtcDayOfWeek'],axis=1)
data73 = data73.drop(['LocalTime','UtcTime','UtcDayOfWeek'],axis=1)
data74 = data74.drop(['LocalTime','UtcTime','UtcDayOfWeek'],axis=1)

## ## Delete all rows with missing data...
data67 = data67.dropna(subset=['UnixUtcTime','UtcYear','UtcMonth','UtcDay','UtcHour','JulianDay','MV72483000_T','MV72483009_T','MV72483016_T','MV72483000_dewpt','MV72483009_dewpt','MV72483016_dewpt','MV72483000_cloud_cov','MV72483009_cloud_cov','MV72483016_cloud_cov','MV72483000_prob_precip01','MV72483009_prob_precip01','MV72483016_prob_precip01','MV72483000_qpf01','MV72483009_qpf01','MV72483016_qpf01','MV72483000_wind_speed','MV72483009_wind_speed','MV72483016_wind_speed','GHI_prev75','GHI_prev60','GHI_prev45','GHI_prev30','GHI_prev15','GHI_post15','GHI_post30','GHI_post45','GHI_post60','GHI_post75','GHI_post90','GHI_post105','GHI_post120','GHI_post135','GHI_post150','GHI_post165','GHI_post180']).reset_index(drop=True)
data68 = data68.dropna(subset=['UnixUtcTime','UtcYear','UtcMonth','UtcDay','UtcHour','JulianDay','MV72483000_T','MV72483009_T','MV72483016_T','MV72483000_dewpt','MV72483009_dewpt','MV72483016_dewpt','MV72483000_cloud_cov','MV72483009_cloud_cov','MV72483016_cloud_cov','MV72483000_prob_precip01','MV72483009_prob_precip01','MV72483016_prob_precip01','MV72483000_qpf01','MV72483009_qpf01','MV72483016_qpf01','MV72483000_wind_speed','MV72483009_wind_speed','MV72483016_wind_speed','GHI_prev75','GHI_prev60','GHI_prev45','GHI_prev30','GHI_prev15','GHI_post15','GHI_post30','GHI_post45','GHI_post60','GHI_post75','GHI_post90','GHI_post105','GHI_post120','GHI_post135','GHI_post150','GHI_post165','GHI_post180']).reset_index(drop=True)
data69 = data69.dropna(subset=['UnixUtcTime','UtcYear','UtcMonth','UtcDay','UtcHour','JulianDay','MV72483000_T','MV72483009_T','MV72483016_T','MV72483000_dewpt','MV72483009_dewpt','MV72483016_dewpt','MV72483000_cloud_cov','MV72483009_cloud_cov','MV72483016_cloud_cov','MV72483000_prob_precip01','MV72483009_prob_precip01','MV72483016_prob_precip01','MV72483000_qpf01','MV72483009_qpf01','MV72483016_qpf01','MV72483000_wind_speed','MV72483009_wind_speed','MV72483016_wind_speed','GHI_prev75','GHI_prev60','GHI_prev45','GHI_prev30','GHI_prev15','GHI_post15','GHI_post30','GHI_post45','GHI_post60','GHI_post75','GHI_post90','GHI_post105','GHI_post120','GHI_post135','GHI_post150','GHI_post165','GHI_post180']).reset_index(drop=True)
data70 = data70.dropna(subset=['UnixUtcTime','UtcYear','UtcMonth','UtcDay','UtcHour','JulianDay','MV72483000_T','MV72483009_T','MV72483016_T','MV72483000_dewpt','MV72483009_dewpt','MV72483016_dewpt','MV72483000_cloud_cov','MV72483009_cloud_cov','MV72483016_cloud_cov','MV72483000_prob_precip01','MV72483009_prob_precip01','MV72483016_prob_precip01','MV72483000_qpf01','MV72483009_qpf01','MV72483016_qpf01','MV72483000_wind_speed','MV72483009_wind_speed','MV72483016_wind_speed','GHI_prev75','GHI_prev60','GHI_prev45','GHI_prev30','GHI_prev15','GHI_post15','GHI_post30','GHI_post45','GHI_post60','GHI_post75','GHI_post90','GHI_post105','GHI_post120','GHI_post135','GHI_post150','GHI_post165','GHI_post180']).reset_index(drop=True)
data71 = data71.dropna(subset=['UnixUtcTime','UtcYear','UtcMonth','UtcDay','UtcHour','JulianDay','MV72483000_T','MV72483009_T','MV72483016_T','MV72483000_dewpt','MV72483009_dewpt','MV72483016_dewpt','MV72483000_cloud_cov','MV72483009_cloud_cov','MV72483016_cloud_cov','MV72483000_prob_precip01','MV72483009_prob_precip01','MV72483016_prob_precip01','MV72483000_qpf01','MV72483009_qpf01','MV72483016_qpf01','MV72483000_wind_speed','MV72483009_wind_speed','MV72483016_wind_speed','GHI_prev75','GHI_prev60','GHI_prev45','GHI_prev30','GHI_prev15','GHI_post15','GHI_post30','GHI_post45','GHI_post60','GHI_post75','GHI_post90','GHI_post105','GHI_post120','GHI_post135','GHI_post150','GHI_post165','GHI_post180']).reset_index(drop=True)
data72 = data72.dropna(subset=['UnixUtcTime','UtcYear','UtcMonth','UtcDay','UtcHour','JulianDay','MV72483000_T','MV72483009_T','MV72483016_T','MV72483000_dewpt','MV72483009_dewpt','MV72483016_dewpt','MV72483000_cloud_cov','MV72483009_cloud_cov','MV72483016_cloud_cov','MV72483000_prob_precip01','MV72483009_prob_precip01','MV72483016_prob_precip01','MV72483000_qpf01','MV72483009_qpf01','MV72483016_qpf01','MV72483000_wind_speed','MV72483009_wind_speed','MV72483016_wind_speed','GHI_prev75','GHI_prev60','GHI_prev45','GHI_prev30','GHI_prev15','GHI_post15','GHI_post30','GHI_post45','GHI_post60','GHI_post75','GHI_post90','GHI_post105','GHI_post120','GHI_post135','GHI_post150','GHI_post165','GHI_post180']).reset_index(drop=True)
data73 = data73.dropna(subset=['UnixUtcTime','UtcYear','UtcMonth','UtcDay','UtcHour','JulianDay','MV72483000_T','MV72483009_T','MV72483016_T','MV72483000_dewpt','MV72483009_dewpt','MV72483016_dewpt','MV72483000_cloud_cov','MV72483009_cloud_cov','MV72483016_cloud_cov','MV72483000_prob_precip01','MV72483009_prob_precip01','MV72483016_prob_precip01','MV72483000_qpf01','MV72483009_qpf01','MV72483016_qpf01','MV72483000_wind_speed','MV72483009_wind_speed','MV72483016_wind_speed','GHI_prev75','GHI_prev60','GHI_prev45','GHI_prev30','GHI_prev15','GHI_post15','GHI_post30','GHI_post45','GHI_post60','GHI_post75','GHI_post90','GHI_post105','GHI_post120','GHI_post135','GHI_post150','GHI_post165','GHI_post180']).reset_index(drop=True)
data74 = data74.dropna(subset=['UnixUtcTime','UtcYear','UtcMonth','UtcDay','UtcHour','JulianDay','MV72483000_T','MV72483009_T','MV72483016_T','MV72483000_dewpt','MV72483009_dewpt','MV72483016_dewpt','MV72483000_cloud_cov','MV72483009_cloud_cov','MV72483016_cloud_cov','MV72483000_prob_precip01','MV72483009_prob_precip01','MV72483016_prob_precip01','MV72483000_qpf01','MV72483009_qpf01','MV72483016_qpf01','MV72483000_wind_speed','MV72483009_wind_speed','MV72483016_wind_speed','GHI_prev75','GHI_prev60','GHI_prev45','GHI_prev30','GHI_prev15','GHI_post15','GHI_post30','GHI_post45','GHI_post60','GHI_post75','GHI_post90','GHI_post105','GHI_post120','GHI_post135','GHI_post150','GHI_post165','GHI_post180']).reset_index(drop=True)

## Delete all rows where the clearness index is above 1.0 or below 0.0...
data67 = data67[data67.GHI_prev15 < 1362.0].reset_index(drop=True)
data67 = data67[data67.GHI_prev30 < 1362.0].reset_index(drop=True)
data67 = data67[data67.GHI_prev45 < 1362.0].reset_index(drop=True)
data67 = data67[data67.GHI_prev60 < 1362.0].reset_index(drop=True)
data67 = data67[data67.GHI_post15 < 1362.0].reset_index(drop=True)
data67 = data67[data67.GHI_post30 < 1362.0].reset_index(drop=True)
data67 = data67[data67.GHI_post45 < 1362.0].reset_index(drop=True)
data67 = data67[data67.GHI_post60 < 1362.0].reset_index(drop=True)
data67 = data67[data67.GHI_post75 < 1362.0].reset_index(drop=True)
data67 = data67[data67.GHI_post90 < 1362.0].reset_index(drop=True)
data67 = data67[data67.GHI_post105 < 1362.0].reset_index(drop=True)
data67 = data67[data67.GHI_post120 < 1362.0].reset_index(drop=True)
data67 = data67[data67.GHI_post135 < 1362.0].reset_index(drop=True)
data67 = data67[data67.GHI_post150 < 1362.0].reset_index(drop=True)
data67 = data67[data67.GHI_post165 < 1362.0].reset_index(drop=True)
data67 = data67[data67.GHI_post180 < 1362.0].reset_index(drop=True)
data68 = data68[data68.GHI_prev15 < 1362.0].reset_index(drop=True)
data68 = data68[data68.GHI_prev30 < 1362.0].reset_index(drop=True)
data68 = data68[data68.GHI_prev45 < 1362.0].reset_index(drop=True)
data68 = data68[data68.GHI_prev60 < 1362.0].reset_index(drop=True)
data68 = data68[data68.GHI_post15 < 1362.0].reset_index(drop=True)
data68 = data68[data68.GHI_post30 < 1362.0].reset_index(drop=True)
data68 = data68[data68.GHI_post45 < 1362.0].reset_index(drop=True)
data68 = data68[data68.GHI_post60 < 1362.0].reset_index(drop=True)
data68 = data68[data68.GHI_post75 < 1362.0].reset_index(drop=True)
data68 = data68[data68.GHI_post90 < 1362.0].reset_index(drop=True)
data68 = data68[data68.GHI_post105 < 1362.0].reset_index(drop=True)
data68 = data68[data68.GHI_post120 < 1362.0].reset_index(drop=True)
data68 = data68[data68.GHI_post135 < 1362.0].reset_index(drop=True)
data68 = data68[data68.GHI_post150 < 1362.0].reset_index(drop=True)
data68 = data68[data68.GHI_post165 < 1362.0].reset_index(drop=True)
data68 = data68[data68.GHI_post180 < 1362.0].reset_index(drop=True)
data69 = data69[data69.GHI_prev15 < 1362.0].reset_index(drop=True)
data69 = data69[data69.GHI_prev30 < 1362.0].reset_index(drop=True)
data69 = data69[data69.GHI_prev45 < 1362.0].reset_index(drop=True)
data69 = data69[data69.GHI_prev60 < 1362.0].reset_index(drop=True)
data69 = data69[data69.GHI_post15 < 1362.0].reset_index(drop=True)
data69 = data69[data69.GHI_post30 < 1362.0].reset_index(drop=True)
data69 = data69[data69.GHI_post45 < 1362.0].reset_index(drop=True)
data69 = data69[data69.GHI_post60 < 1362.0].reset_index(drop=True)
data69 = data69[data69.GHI_post75 < 1362.0].reset_index(drop=True)
data69 = data69[data69.GHI_post90 < 1362.0].reset_index(drop=True)
data69 = data69[data69.GHI_post105 < 1362.0].reset_index(drop=True)
data69 = data69[data69.GHI_post120 < 1362.0].reset_index(drop=True)
data69 = data69[data69.GHI_post135 < 1362.0].reset_index(drop=True)
data69 = data69[data69.GHI_post150 < 1362.0].reset_index(drop=True)
data69 = data69[data69.GHI_post165 < 1362.0].reset_index(drop=True)
data69 = data69[data69.GHI_post180 < 1362.0].reset_index(drop=True)
data70 = data70[data70.GHI_prev15 < 1362.0].reset_index(drop=True)
data70 = data70[data70.GHI_prev30 < 1362.0].reset_index(drop=True)
data70 = data70[data70.GHI_prev45 < 1362.0].reset_index(drop=True)
data70 = data70[data70.GHI_prev60 < 1362.0].reset_index(drop=True)
data70 = data70[data70.GHI_post15 < 1362.0].reset_index(drop=True)
data70 = data70[data70.GHI_post30 < 1362.0].reset_index(drop=True)
data70 = data70[data70.GHI_post45 < 1362.0].reset_index(drop=True)
data70 = data70[data70.GHI_post60 < 1362.0].reset_index(drop=True)
data70 = data70[data70.GHI_post75 < 1362.0].reset_index(drop=True)
data70 = data70[data70.GHI_post90 < 1362.0].reset_index(drop=True)
data70 = data70[data70.GHI_post105 < 1362.0].reset_index(drop=True)
data70 = data70[data70.GHI_post120 < 1362.0].reset_index(drop=True)
data70 = data70[data70.GHI_post135 < 1362.0].reset_index(drop=True)
data70 = data70[data70.GHI_post150 < 1362.0].reset_index(drop=True)
data70 = data70[data70.GHI_post165 < 1362.0].reset_index(drop=True)
data70 = data70[data70.GHI_post180 < 1362.0].reset_index(drop=True)
data71 = data71[data71.GHI_prev15 < 1362.0].reset_index(drop=True)
data71 = data71[data71.GHI_prev30 < 1362.0].reset_index(drop=True)
data71 = data71[data71.GHI_prev45 < 1362.0].reset_index(drop=True)
data71 = data71[data71.GHI_prev60 < 1362.0].reset_index(drop=True)
data71 = data71[data71.GHI_post15 < 1362.0].reset_index(drop=True)
data71 = data71[data71.GHI_post30 < 1362.0].reset_index(drop=True)
data71 = data71[data71.GHI_post45 < 1362.0].reset_index(drop=True)
data71 = data71[data71.GHI_post60 < 1362.0].reset_index(drop=True)
data71 = data71[data71.GHI_post75 < 1362.0].reset_index(drop=True)
data71 = data71[data71.GHI_post90 < 1362.0].reset_index(drop=True)
data71 = data71[data71.GHI_post105 < 1362.0].reset_index(drop=True)
data71 = data71[data71.GHI_post120 < 1362.0].reset_index(drop=True)
data71 = data71[data71.GHI_post135 < 1362.0].reset_index(drop=True)
data71 = data71[data71.GHI_post150 < 1362.0].reset_index(drop=True)
data71 = data71[data71.GHI_post165 < 1362.0].reset_index(drop=True)
data71 = data71[data71.GHI_post180 < 1362.0].reset_index(drop=True)
data72 = data72[data72.GHI_prev15 < 1362.0].reset_index(drop=True)
data72 = data72[data72.GHI_prev30 < 1362.0].reset_index(drop=True)
data72 = data72[data72.GHI_prev45 < 1362.0].reset_index(drop=True)
data72 = data72[data72.GHI_prev60 < 1362.0].reset_index(drop=True)
data72 = data72[data72.GHI_post15 < 1362.0].reset_index(drop=True)
data72 = data72[data72.GHI_post30 < 1362.0].reset_index(drop=True)
data72 = data72[data72.GHI_post45 < 1362.0].reset_index(drop=True)
data72 = data72[data72.GHI_post60 < 1362.0].reset_index(drop=True)
data72 = data72[data72.GHI_post75 < 1362.0].reset_index(drop=True)
data72 = data72[data72.GHI_post90 < 1362.0].reset_index(drop=True)
data72 = data72[data72.GHI_post105 < 1362.0].reset_index(drop=True)
data72 = data72[data72.GHI_post120 < 1362.0].reset_index(drop=True)
data72 = data72[data72.GHI_post135 < 1362.0].reset_index(drop=True)
data72 = data72[data72.GHI_post150 < 1362.0].reset_index(drop=True)
data72 = data72[data72.GHI_post165 < 1362.0].reset_index(drop=True)
data72 = data72[data72.GHI_post180 < 1362.0].reset_index(drop=True)
data73 = data73[data73.GHI_prev15 < 1362.0].reset_index(drop=True)
data73 = data73[data73.GHI_prev30 < 1362.0].reset_index(drop=True)
data73 = data73[data73.GHI_prev45 < 1362.0].reset_index(drop=True)
data73 = data73[data73.GHI_prev60 < 1362.0].reset_index(drop=True)
data73 = data73[data73.GHI_post15 < 1362.0].reset_index(drop=True)
data73 = data73[data73.GHI_post30 < 1362.0].reset_index(drop=True)
data73 = data73[data73.GHI_post45 < 1362.0].reset_index(drop=True)
data73 = data73[data73.GHI_post60 < 1362.0].reset_index(drop=True)
data73 = data73[data73.GHI_post75 < 1362.0].reset_index(drop=True)
data73 = data73[data73.GHI_post90 < 1362.0].reset_index(drop=True)
data73 = data73[data73.GHI_post105 < 1362.0].reset_index(drop=True)
data73 = data73[data73.GHI_post120 < 1362.0].reset_index(drop=True)
data73 = data73[data73.GHI_post135 < 1362.0].reset_index(drop=True)
data73 = data73[data73.GHI_post150 < 1362.0].reset_index(drop=True)
data73 = data73[data73.GHI_post165 < 1362.0].reset_index(drop=True)
data73 = data73[data73.GHI_post180 < 1362.0].reset_index(drop=True)
data74 = data74[data74.GHI_prev15 < 1362.0].reset_index(drop=True)
data74 = data74[data74.GHI_prev30 < 1362.0].reset_index(drop=True)
data74 = data74[data74.GHI_prev45 < 1362.0].reset_index(drop=True)
data74 = data74[data74.GHI_prev60 < 1362.0].reset_index(drop=True)
data74 = data74[data74.GHI_post15 < 1362.0].reset_index(drop=True)
data74 = data74[data74.GHI_post30 < 1362.0].reset_index(drop=True)
data74 = data74[data74.GHI_post45 < 1362.0].reset_index(drop=True)
data74 = data74[data74.GHI_post60 < 1362.0].reset_index(drop=True)
data74 = data74[data74.GHI_post75 < 1362.0].reset_index(drop=True)
data74 = data74[data74.GHI_post90 < 1362.0].reset_index(drop=True)
data74 = data74[data74.GHI_post105 < 1362.0].reset_index(drop=True)
data74 = data74[data74.GHI_post120 < 1362.0].reset_index(drop=True)
data74 = data74[data74.GHI_post135 < 1362.0].reset_index(drop=True)
data74 = data74[data74.GHI_post150 < 1362.0].reset_index(drop=True)
data74 = data74[data74.GHI_post165 < 1362.0].reset_index(drop=True)
data74 = data74[data74.GHI_post180 < 1362.0].reset_index(drop=True)


####### Create averages and add columns to each of the datasets #######
## Temperature ##
data67['T_AVG'] = data67[['MV72483000_T','MV72483009_T','MV72483016_T']].mean(axis=1)
data68['T_AVG'] = data68[['MV72483000_T','MV72483009_T','MV72483016_T']].mean(axis=1)
data69['T_AVG'] = data69[['MV72483000_T','MV72483009_T','MV72483016_T']].mean(axis=1)
data70['T_AVG'] = data70[['MV72483000_T','MV72483009_T','MV72483016_T']].mean(axis=1)
data71['T_AVG'] = data71[['MV72483000_T','MV72483009_T','MV72483016_T']].mean(axis=1)
data72['T_AVG'] = data72[['MV72483000_T','MV72483009_T','MV72483016_T']].mean(axis=1)
data73['T_AVG'] = data73[['MV72483000_T','MV72483009_T','MV72483016_T']].mean(axis=1)
data74['T_AVG'] = data74[['MV72483000_T','MV72483009_T','MV72483016_T']].mean(axis=1)
## Dewpoint Temperature ##
data67['dewpt_AVG'] = data67[['MV72483000_dewpt','MV72483009_dewpt','MV72483016_dewpt']].mean(axis=1)
data68['dewpt_AVG'] = data68[['MV72483000_dewpt','MV72483009_dewpt','MV72483016_dewpt']].mean(axis=1)
data69['dewpt_AVG'] = data69[['MV72483000_dewpt','MV72483009_dewpt','MV72483016_dewpt']].mean(axis=1)
data70['dewpt_AVG'] = data70[['MV72483000_dewpt','MV72483009_dewpt','MV72483016_dewpt']].mean(axis=1)
data71['dewpt_AVG'] = data71[['MV72483000_dewpt','MV72483009_dewpt','MV72483016_dewpt']].mean(axis=1)
data72['dewpt_AVG'] = data72[['MV72483000_dewpt','MV72483009_dewpt','MV72483016_dewpt']].mean(axis=1)
data73['dewpt_AVG'] = data73[['MV72483000_dewpt','MV72483009_dewpt','MV72483016_dewpt']].mean(axis=1)
data74['dewpt_AVG'] = data74[['MV72483000_dewpt','MV72483009_dewpt','MV72483016_dewpt']].mean(axis=1)
## Wind Speed ##
data67['wind_speed_AVG'] = data67[['MV72483000_wind_speed','MV72483009_wind_speed','MV72483016_wind_speed']].mean(axis=1)
data68['wind_speed_AVG'] = data68[['MV72483000_wind_speed','MV72483009_wind_speed','MV72483016_wind_speed']].mean(axis=1)
data69['wind_speed_AVG'] = data69[['MV72483000_wind_speed','MV72483009_wind_speed','MV72483016_wind_speed']].mean(axis=1)
data70['wind_speed_AVG'] = data70[['MV72483000_wind_speed','MV72483009_wind_speed','MV72483016_wind_speed']].mean(axis=1)
data71['wind_speed_AVG'] = data71[['MV72483000_wind_speed','MV72483009_wind_speed','MV72483016_wind_speed']].mean(axis=1)
data72['wind_speed_AVG'] = data72[['MV72483000_wind_speed','MV72483009_wind_speed','MV72483016_wind_speed']].mean(axis=1)
data73['wind_speed_AVG'] = data73[['MV72483000_wind_speed','MV72483009_wind_speed','MV72483016_wind_speed']].mean(axis=1)
data74['wind_speed_AVG'] = data74[['MV72483000_wind_speed','MV72483009_wind_speed','MV72483016_wind_speed']].mean(axis=1)
## Prob Precip ##
data67['prob_precip01_AVG'] = data67[['MV72483000_prob_precip01','MV72483009_prob_precip01','MV72483016_prob_precip01']].mean(axis=1)
data68['prob_precip01_AVG'] = data68[['MV72483000_prob_precip01','MV72483009_prob_precip01','MV72483016_prob_precip01']].mean(axis=1)
data69['prob_precip01_AVG'] = data69[['MV72483000_prob_precip01','MV72483009_prob_precip01','MV72483016_prob_precip01']].mean(axis=1)
data70['prob_precip01_AVG'] = data70[['MV72483000_prob_precip01','MV72483009_prob_precip01','MV72483016_prob_precip01']].mean(axis=1)
data71['prob_precip01_AVG'] = data71[['MV72483000_prob_precip01','MV72483009_prob_precip01','MV72483016_prob_precip01']].mean(axis=1)
data72['prob_precip01_AVG'] = data72[['MV72483000_prob_precip01','MV72483009_prob_precip01','MV72483016_prob_precip01']].mean(axis=1)
data73['prob_precip01_AVG'] = data73[['MV72483000_prob_precip01','MV72483009_prob_precip01','MV72483016_prob_precip01']].mean(axis=1)
data74['prob_precip01_AVG'] = data74[['MV72483000_prob_precip01','MV72483009_prob_precip01','MV72483016_prob_precip01']].mean(axis=1)
## QPF ##
data67['qpf01_AVG'] = data67[['MV72483000_qpf01','MV72483009_qpf01','MV72483016_qpf01']].mean(axis=1)
data68['qpf01_AVG'] = data68[['MV72483000_qpf01','MV72483009_qpf01','MV72483016_qpf01']].mean(axis=1)
data69['qpf01_AVG'] = data69[['MV72483000_qpf01','MV72483009_qpf01','MV72483016_qpf01']].mean(axis=1)
data70['qpf01_AVG'] = data70[['MV72483000_qpf01','MV72483009_qpf01','MV72483016_qpf01']].mean(axis=1)
data71['qpf01_AVG'] = data71[['MV72483000_qpf01','MV72483009_qpf01','MV72483016_qpf01']].mean(axis=1)
data72['qpf01_AVG'] = data72[['MV72483000_qpf01','MV72483009_qpf01','MV72483016_qpf01']].mean(axis=1)
data73['qpf01_AVG'] = data73[['MV72483000_qpf01','MV72483009_qpf01','MV72483016_qpf01']].mean(axis=1)
data74['qpf01_AVG'] = data74[['MV72483000_qpf01','MV72483009_qpf01','MV72483016_qpf01']].mean(axis=1)
## Cloud Cover ##
data67['cloud_cov_AVG'] = data67[['MV72483000_cloud_cov','MV72483009_cloud_cov','MV72483016_cloud_cov']].mean(axis=1)
data68['cloud_cov_AVG'] = data68[['MV72483000_cloud_cov','MV72483009_cloud_cov','MV72483016_cloud_cov']].mean(axis=1)
data69['cloud_cov_AVG'] = data69[['MV72483000_cloud_cov','MV72483009_cloud_cov','MV72483016_cloud_cov']].mean(axis=1)
data70['cloud_cov_AVG'] = data70[['MV72483000_cloud_cov','MV72483009_cloud_cov','MV72483016_cloud_cov']].mean(axis=1)
data71['cloud_cov_AVG'] = data71[['MV72483000_cloud_cov','MV72483009_cloud_cov','MV72483016_cloud_cov']].mean(axis=1)
data72['cloud_cov_AVG'] = data72[['MV72483000_cloud_cov','MV72483009_cloud_cov','MV72483016_cloud_cov']].mean(axis=1)
data73['cloud_cov_AVG'] = data73[['MV72483000_cloud_cov','MV72483009_cloud_cov','MV72483016_cloud_cov']].mean(axis=1)
data74['cloud_cov_AVG'] = data74[['MV72483000_cloud_cov','MV72483009_cloud_cov','MV72483016_cloud_cov']].mean(axis=1)

########################### Delete rows with bad data ############################################################
## Delete all rows where the clearness index is above 1.0 or below 0.0...
data67 = data67[data67.T_AVG > 2.0].reset_index(drop=True)
data67 = data67[data67.T_AVG < 120.0].reset_index(drop=True)
data67 = data67[data67.dewpt_AVG > -2.0].reset_index(drop=True)
data67 = data67[data67.dewpt_AVG < 120.0].reset_index(drop=True)
data67 = data67[data67.wind_speed_AVG > -30.0].reset_index(drop=True)
data67 = data67[data67.wind_speed_AVG < 60.0].reset_index(drop=True)
data67 = data67[data67.prob_precip01_AVG > -0.01].reset_index(drop=True)
data67 = data67[data67.prob_precip01_AVG < 100.1].reset_index(drop=True)
data67 = data67[data67.qpf01_AVG > -0.1].reset_index(drop=True)
data67 = data67[data67.qpf01_AVG < 2.0].reset_index(drop=True)
data67 = data67[data67.cloud_cov_AVG > -0.1].reset_index(drop=True)
data67 = data67[data67.cloud_cov_AVG < 1.01].reset_index(drop=True)
data68 = data68[data68.T_AVG > 2.0].reset_index(drop=True)
data68 = data68[data68.T_AVG < 120.0].reset_index(drop=True)
data68 = data68[data68.dewpt_AVG > -2.0].reset_index(drop=True)
data68 = data68[data68.dewpt_AVG < 120.0].reset_index(drop=True)
data68 = data68[data68.wind_speed_AVG > -30.0].reset_index(drop=True)
data68 = data68[data68.wind_speed_AVG < 60.0].reset_index(drop=True)
data68 = data68[data68.prob_precip01_AVG > -0.1].reset_index(drop=True)
data68 = data68[data68.prob_precip01_AVG < 100.1].reset_index(drop=True)
data68 = data68[data68.qpf01_AVG > -0.1].reset_index(drop=True)
data68 = data68[data68.qpf01_AVG < 2.0].reset_index(drop=True)
data68 = data68[data68.cloud_cov_AVG > -0.1].reset_index(drop=True)
data68 = data68[data68.cloud_cov_AVG < 1.01].reset_index(drop=True)
data69 = data69[data69.T_AVG > 2.0].reset_index(drop=True)
data69 = data69[data69.T_AVG < 120.0].reset_index(drop=True)
data69 = data69[data69.dewpt_AVG > -2.0].reset_index(drop=True)
data69 = data69[data69.dewpt_AVG < 120.0].reset_index(drop=True)
data69 = data69[data69.wind_speed_AVG > -30.0].reset_index(drop=True)
data69 = data69[data69.wind_speed_AVG < 60.0].reset_index(drop=True)
data69 = data69[data69.prob_precip01_AVG > -0.01].reset_index(drop=True)
data69 = data69[data69.prob_precip01_AVG < 100.1].reset_index(drop=True)
data69 = data69[data69.qpf01_AVG > -0.01].reset_index(drop=True)
data69 = data69[data69.qpf01_AVG < 2.0].reset_index(drop=True)
data69 = data69[data69.cloud_cov_AVG > -0.1].reset_index(drop=True)
data69 = data69[data69.cloud_cov_AVG < 1.01].reset_index(drop=True)
data70 = data70[data70.T_AVG > 2.0].reset_index(drop=True)
data70 = data70[data70.T_AVG < 120.0].reset_index(drop=True)
data70 = data70[data70.dewpt_AVG > -2.0].reset_index(drop=True)
data70 = data70[data70.dewpt_AVG < 120.0].reset_index(drop=True)
data70 = data70[data70.wind_speed_AVG > -30.0].reset_index(drop=True)
data70 = data70[data70.wind_speed_AVG < 60.0].reset_index(drop=True)
data70 = data70[data70.prob_precip01_AVG > -0.01].reset_index(drop=True)
data70 = data70[data70.prob_precip01_AVG < 100.1].reset_index(drop=True)
data70 = data70[data70.qpf01_AVG > -0.01].reset_index(drop=True)
data70 = data70[data70.qpf01_AVG < 2.0].reset_index(drop=True)
data70 = data70[data70.cloud_cov_AVG > -0.1].reset_index(drop=True)
data70 = data70[data70.cloud_cov_AVG < 1.1].reset_index(drop=True)
data71 = data71[data71.T_AVG > 2.0].reset_index(drop=True)
data71 = data71[data71.T_AVG < 120.0].reset_index(drop=True)
data71 = data71[data71.dewpt_AVG > -2.0].reset_index(drop=True)
data71 = data71[data71.dewpt_AVG < 120.0].reset_index(drop=True)
data71 = data71[data71.wind_speed_AVG > -30.0].reset_index(drop=True)
data71 = data71[data71.wind_speed_AVG < 60.0].reset_index(drop=True)
data71 = data71[data71.prob_precip01_AVG > -0.01].reset_index(drop=True)
data71 = data71[data71.prob_precip01_AVG < 100.1].reset_index(drop=True)
data71 = data71[data71.qpf01_AVG > -0.01].reset_index(drop=True)
data71 = data71[data71.qpf01_AVG < 2.0].reset_index(drop=True)
data71 = data71[data71.cloud_cov_AVG > -0.1].reset_index(drop=True)
data71 = data71[data71.cloud_cov_AVG < 1.01].reset_index(drop=True)
data72 = data72[data72.T_AVG > 2.0].reset_index(drop=True)
data72 = data72[data72.T_AVG < 120.0].reset_index(drop=True)
data72 = data72[data72.dewpt_AVG > -2.0].reset_index(drop=True)
data72 = data72[data72.dewpt_AVG < 120.0].reset_index(drop=True)
data72 = data72[data72.wind_speed_AVG > -30.0].reset_index(drop=True)
data72 = data72[data72.wind_speed_AVG < 60.0].reset_index(drop=True)
data72 = data72[data72.prob_precip01_AVG > -0.1].reset_index(drop=True)
data72 = data72[data72.prob_precip01_AVG < 100.1].reset_index(drop=True)
data72 = data72[data72.qpf01_AVG > -0.01].reset_index(drop=True)
data72 = data72[data72.qpf01_AVG < 2.0].reset_index(drop=True)
data72 = data72[data72.cloud_cov_AVG > -0.1].reset_index(drop=True)
data72 = data72[data72.cloud_cov_AVG < 1.01].reset_index(drop=True)
data73 = data73[data73.T_AVG > 2.0].reset_index(drop=True)
data73 = data73[data73.T_AVG < 120.0].reset_index(drop=True)
data73 = data73[data73.dewpt_AVG > -2.0].reset_index(drop=True)
data73 = data73[data73.dewpt_AVG < 120.0].reset_index(drop=True)
data73 = data73[data73.wind_speed_AVG > -30.0].reset_index(drop=True)
data73 = data73[data73.wind_speed_AVG < 60.0].reset_index(drop=True)
data73 = data73[data73.prob_precip01_AVG > -0.01].reset_index(drop=True)
data73 = data73[data73.prob_precip01_AVG < 100.1].reset_index(drop=True)
data73 = data73[data73.qpf01_AVG > -0.1].reset_index(drop=True)
data73 = data73[data73.qpf01_AVG < 2.0].reset_index(drop=True)
data73 = data73[data73.cloud_cov_AVG > -0.1].reset_index(drop=True)
data73 = data73[data73.cloud_cov_AVG < 1.1].reset_index(drop=True)
data74 = data74[data74.T_AVG > 2.0].reset_index(drop=True)
data74 = data74[data74.T_AVG < 120.0].reset_index(drop=True)
data74 = data74[data74.dewpt_AVG > -2.0].reset_index(drop=True)
data74 = data74[data74.dewpt_AVG < 120.0].reset_index(drop=True)
data74 = data74[data74.wind_speed_AVG > -30.0].reset_index(drop=True)
data74 = data74[data74.wind_speed_AVG < 60.0].reset_index(drop=True)
data74 = data74[data74.prob_precip01_AVG > -0.1].reset_index(drop=True)
data74 = data74[data74.prob_precip01_AVG < 100.1].reset_index(drop=True)
data74 = data74[data74.qpf01_AVG > -0.01].reset_index(drop=True)
data74 = data74[data74.qpf01_AVG < 2.0].reset_index(drop=True)
data74 = data74[data74.cloud_cov_AVG > -0.01].reset_index(drop=True)
data74 = data74[data74.cloud_cov_AVG < 1.01].reset_index(drop=True)

## Compute Dewpoint Depression ##
data67['T_Td'] = data67['T_AVG']-data67['dewpt_AVG']
data68['T_Td'] = data68['T_AVG']-data68['dewpt_AVG']
data69['T_Td'] = data69['T_AVG']-data69['dewpt_AVG']
data70['T_Td'] = data70['T_AVG']-data70['dewpt_AVG']
data71['T_Td'] = data71['T_AVG']-data71['dewpt_AVG']
data72['T_Td'] = data72['T_AVG']-data72['dewpt_AVG']
data73['T_Td'] = data73['T_AVG']-data73['dewpt_AVG']
data74['T_Td'] = data74['T_AVG']-data74['dewpt_AVG']

## Cloud Cover Variability (Standard Deviation) ##
data67['cloud_cov_STD'] = data67[['MV72483000_cloud_cov','MV72483009_cloud_cov','MV72483016_cloud_cov']].std(axis=1)
data68['cloud_cov_STD'] = data68[['MV72483000_cloud_cov','MV72483009_cloud_cov','MV72483016_cloud_cov']].std(axis=1)
data69['cloud_cov_STD'] = data69[['MV72483000_cloud_cov','MV72483009_cloud_cov','MV72483016_cloud_cov']].std(axis=1)
data70['cloud_cov_STD'] = data70[['MV72483000_cloud_cov','MV72483009_cloud_cov','MV72483016_cloud_cov']].std(axis=1)
data71['cloud_cov_STD'] = data71[['MV72483000_cloud_cov','MV72483009_cloud_cov','MV72483016_cloud_cov']].std(axis=1)
data72['cloud_cov_STD'] = data72[['MV72483000_cloud_cov','MV72483009_cloud_cov','MV72483016_cloud_cov']].std(axis=1)
data73['cloud_cov_STD'] = data73[['MV72483000_cloud_cov','MV72483009_cloud_cov','MV72483016_cloud_cov']].std(axis=1)
data74['cloud_cov_STD'] = data74[['MV72483000_cloud_cov','MV72483009_cloud_cov','MV72483016_cloud_cov']].std(axis=1)

## Cloud Cover Squared  ##
data67['cloud_cov_SQRD'] = data67['cloud_cov_AVG']**2
data68['cloud_cov_SQRD'] = data68['cloud_cov_AVG']**2
data69['cloud_cov_SQRD'] = data69['cloud_cov_AVG']**2
data70['cloud_cov_SQRD'] = data70['cloud_cov_AVG']**2
data71['cloud_cov_SQRD'] = data71['cloud_cov_AVG']**2
data72['cloud_cov_SQRD'] = data72['cloud_cov_AVG']**2
data73['cloud_cov_SQRD'] = data73['cloud_cov_AVG']**2
data74['cloud_cov_SQRD'] = data74['cloud_cov_AVG']**2

## Compute TOA irradiance...add to data...

for t in xrange(0,8):
    data = []
    if t == 0:
        data = data67
    if t == 1:
        data = data68
    if t == 2:
        data = data69
    if t == 3:
        data = data70
    if t == 4:
        data = data71
    if t == 5:
        data = data72
    if t == 6:
        data = data73
    if t == 7:
        data = data74
    sLength = len(data)
    index = -1
    lat = 38.40 #Sacramento
    lon = -121.35
    alt = 10
    kt = 0.0
    toa_prev75 = []
    toa_prev60 = []
    toa_prev45 = []
    toa_prev30 = []
    toa_prev15 = []
    toa_post15 = []
    toa_post30 = []
    toa_post45 = []
    toa_post60 = []
    toa_post75 = []
    toa_post90 = []
    toa_post105 = []
    toa_post120 = []
    toa_post135 = []
    toa_post150 = []
    toa_post165 = []
    toa_post180 = []
    tme_prev75 = []
    tme_prev60 = []
    tme_prev45 = []
    tme_prev30 = []
    tme_prev15 = []
    tme_post15 = []
    tme_post30 = []
    tme_post45 = []
    tme_post60 = []
    tme_post75 = []
    tme_post90 = []
    tme_post105 = []
    tme_post120 = []
    tme_post135 = []
    tme_post150 = []
    tme_post165 = []
    tme_post180 = []
    print "t is..."
    print t
    for x in xrange(0,len(data)):
        tmp = []
        tme_prev75 = data['UnixUtcTime'].values-4050
        UnixSMUD = datetime.datetime(1970,1,1) + datetime.timedelta(seconds=float(tme_prev75[x]))
        tme = UnixSMUD.strftime("%Y-%m-%d %H:%M:%S")
        day = int(tme[8:10])
        month = int(tme[5:7])
        year = int(tme[0:4])
        hr = int(tme[11:13])
        mm = int(tme[14:16])
        ToD = (hr + mm/60.0)
        toa_prev75.append(TOA(lat,lon,alt,day,month,year,ToD))
       
        tme_prev60 = data['UnixUtcTime'].values-3150
        UnixSMUD = datetime.datetime(1970,1,1) + datetime.timedelta(seconds=float(tme_prev60[x]))
        tme = UnixSMUD.strftime("%Y-%m-%d %H:%M:%S")
        day = int(tme[8:10])
        month = int(tme[5:7])
        year = int(tme[0:4])
        hr = int(tme[11:13])
        mm = int(tme[14:16])
        ToD = (hr + mm/60.0)
        toa_prev60.append(TOA(lat,lon,alt,day,month,year,ToD))
        
        tme_prev45 = data['UnixUtcTime'].values-2250
        UnixSMUD = datetime.datetime(1970,1,1) + datetime.timedelta(seconds=float(tme_prev45[x]))
        tme = UnixSMUD.strftime("%Y-%m-%d %H:%M:%S")
        day = int(tme[8:10])
        month = int(tme[5:7])
        year = int(tme[0:4])
        hr = int(tme[11:13])
        mm = int(tme[14:16])
        ToD = (hr + mm/60.0)
        toa_prev45.append(TOA(lat,lon,alt,day,month,year,ToD))
        
        tme_prev30 = data['UnixUtcTime'].values-1350
        UnixSMUD = datetime.datetime(1970,1,1) + datetime.timedelta(seconds=float(tme_prev30[x]))
        tme = UnixSMUD.strftime("%Y-%m-%d %H:%M:%S")
        day = int(tme[8:10])
        month = int(tme[5:7])
        year = int(tme[0:4])
        hr = int(tme[11:13])
        mm = int(tme[14:16])
        ToD = (hr + mm/60.0)
        toa_prev30.append(TOA(lat,lon,alt,day,month,year,ToD))
        
        tme_prev15 = data['UnixUtcTime'].values-450
        UnixSMUD = datetime.datetime(1970,1,1) + datetime.timedelta(seconds=float(tme_prev15[x]))
        tme = UnixSMUD.strftime("%Y-%m-%d %H:%M:%S")
        day = int(tme[8:10])
        month = int(tme[5:7])
        year = int(tme[0:4])
        hr = int(tme[11:13])
        mm = int(tme[14:16])
        ToD = (hr + mm/60.0)
        toa_prev15.append(TOA(lat,lon,alt,day,month,year,ToD))
        
        tme_post15 = data['UnixUtcTime'].values+450
        UnixSMUD = datetime.datetime(1970,1,1) + datetime.timedelta(seconds=float(tme_post15[x]))
        tme = UnixSMUD.strftime("%Y-%m-%d %H:%M:%S")
        day = int(tme[8:10])
        month = int(tme[5:7])
        year = int(tme[0:4])
        hr = int(tme[11:13])
        mm = int(tme[14:16])
        ToD = (hr + mm/60.0)
        toa_post15.append(TOA(lat,lon,alt,day,month,year,ToD))
        
        tme_post30 = data['UnixUtcTime'].values+1350
        UnixSMUD = datetime.datetime(1970,1,1) + datetime.timedelta(seconds=float(tme_post30[x]))
        tme = UnixSMUD.strftime("%Y-%m-%d %H:%M:%S")
        day = int(tme[8:10])
        month = int(tme[5:7])
        year = int(tme[0:4])
        hr = int(tme[11:13])
        mm = int(tme[14:16])
        ToD = (hr + mm/60.0)
        toa_post30.append(TOA(lat,lon,alt,day,month,year,ToD))
        
        tme_post45 = data['UnixUtcTime'].values+2250
        UnixSMUD = datetime.datetime(1970,1,1) + datetime.timedelta(seconds=float(tme_post45[x]))
        tme = UnixSMUD.strftime("%Y-%m-%d %H:%M:%S")
        day = int(tme[8:10])
        month = int(tme[5:7])
        year = int(tme[0:4])
        hr = int(tme[11:13])
        mm = int(tme[14:16])
        TOD = (hr + mm/60.0)
        toa_post45.append(TOA(lat,lon,alt,day,month,year,ToD))
        
        tme_post60 = data['UnixUtcTime'].values+3150
        UnixSMUD = datetime.datetime(1970,1,1) + datetime.timedelta(seconds=float(tme_post60[x]))
        tme = UnixSMUD.strftime("%Y-%m-%d %H:%M:%S")
        day = int(tme[8:10])
        month = int(tme[5:7])
        year = int(tme[0:4])
        hr = int(tme[11:13])
        mm = int(tme[14:16])
        ToD = (hr + mm/60.0)
        toa_post60.append(TOA(lat,lon,alt,day,month,year,ToD))
        
        tme_post75 = data['UnixUtcTime'].values+4050
        UnixSMUD = datetime.datetime(1970,1,1) + datetime.timedelta(seconds=float(tme_post75[x]))
        tme = UnixSMUD.strftime("%Y-%m-%d %H:%M:%S")
        day = int(tme[8:10])
        month = int(tme[5:7])
        year = int(tme[0:4])
        hr = int(tme[11:13])
        mm = int(tme[14:16])
        ToD = (hr + mm/60.0)
        toa_post75.append(TOA(lat,lon,alt,day,month,year,ToD))
        
        tme_post90 = data['UnixUtcTime'].values+4950
        UnixSMUD = datetime.datetime(1970,1,1) + datetime.timedelta(seconds=float(tme_post90[x]))
        tme = UnixSMUD.strftime("%Y-%m-%d %H:%M:%S")
        day = int(tme[8:10])
        month = int(tme[5:7])
        year = int(tme[0:4])
        hr = int(tme[11:13])
        mm = int(tme[14:16])
        ToD = (hr + mm/60.0)
        toa_post90.append(TOA(lat,lon,alt,day,month,year,ToD))
        
        tme_post105 = data['UnixUtcTime'].values+5850
        UnixSMUD = datetime.datetime(1970,1,1) + datetime.timedelta(seconds=float(tme_post105[x]))
        tme = UnixSMUD.strftime("%Y-%m-%d %H:%M:%S")
        day = int(tme[8:10])
        month = int(tme[5:7])
        year = int(tme[0:4])
        hr = int(tme[11:13])
        mm = int(tme[14:16])
        ToD = (hr + mm/60.0)
        toa_post105.append(TOA(lat,lon,alt,day,month,year,ToD))
        
        tme_post120 = data['UnixUtcTime'].values+6750
        UnixSMUD = datetime.datetime(1970,1,1) + datetime.timedelta(seconds=float(tme_post120[x]))
        tme = UnixSMUD.strftime("%Y-%m-%d %H:%M:%S")
        day = int(tme[8:10])
        month = int(tme[5:7])
        year = int(tme[0:4])
        hr = int(tme[11:13])
        mm = int(tme[14:16])
        ToD = (hr + mm/60.0)
        toa_post120.append(TOA(lat,lon,alt,day,month,year,ToD))
        
        tme_post135 = data['UnixUtcTime'].values+7650
        UnixSMUD = datetime.datetime(1970,1,1) + datetime.timedelta(seconds=float(tme_post135[x]))
        tme = UnixSMUD.strftime("%Y-%m-%d %H:%M:%S")
        day = int(tme[8:10])
        month = int(tme[5:7])
        year = int(tme[0:4])
        hr = int(tme[11:13])
        mm = int(tme[14:16])
        ToD = (hr + mm/60.0)
        toa_post135.append(TOA(lat,lon,alt,day,month,year,ToD))
        
        tme_post150 = data['UnixUtcTime'].values+8550
        UnixSMUD = datetime.datetime(1970,1,1) + datetime.timedelta(seconds=float(tme_post150[x]))
        tme = UnixSMUD.strftime("%Y-%m-%d %H:%M:%S")
        day = int(tme[8:10])
        month = int(tme[5:7])
        year = int(tme[0:4])
        hr = int(tme[11:13])
        mm = int(tme[14:16])
        ToD = (hr + mm/60.0)
        toa_post150.append(TOA(lat,lon,alt,day,month,year,ToD))
        
        tme_post165 = data['UnixUtcTime'].values+9450
        UnixSMUD = datetime.datetime(1970,1,1) + datetime.timedelta(seconds=float(tme_post165[x]))
        tme = UnixSMUD.strftime("%Y-%m-%d %H:%M:%S")
        day = int(tme[8:10])
        month = int(tme[5:7])
        year = int(tme[0:4])
        hr = int(tme[11:13])
        mm = int(tme[14:16])
        ToD = (hr + mm/60.0)
        toa_post165.append(TOA(lat,lon,alt,day,month,year,ToD))
        
        tme_post180 = data['UnixUtcTime'].values+10350
        UnixSMUD = datetime.datetime(1970,1,1) + datetime.timedelta(seconds=float(tme_post180[x]))
        tme = UnixSMUD.strftime("%Y-%m-%d %H:%M:%S")
        day = int(tme[8:10])
        month = int(tme[5:7])
        year = int(tme[0:4])
        hr = int(tme[11:13])
        mm = int(tme[14:16])
        ToD = (hr + mm/60.0)
        TOAA = TOA(lat,lon,alt,day,month,year,ToD)
        toa_post180.append(TOA(lat,lon,alt,day,month,year,ToD))

    GHI_prev75 = data['GHI_prev75']
    GHI_prev60 = data['GHI_prev60']
    GHI_prev45 = data['GHI_prev45']
    GHI_prev30 = data['GHI_prev30']
    GHI_prev15 = data['GHI_prev15']
    GHI_post15 = data['GHI_post15']
    GHI_post30 = data['GHI_post30']
    GHI_post45 = data['GHI_post45']
    GHI_post60 = data['GHI_post60']
    GHI_post75 = data['GHI_post75']
    GHI_post90 = data['GHI_post90']
    GHI_post105 = data['GHI_post105']
    GHI_post120 = data['GHI_post120']
    GHI_post135 = data['GHI_post135']
    GHI_post150 = data['GHI_post150']
    GHI_post165 = data['GHI_post165']
    GHI_post180 = data['GHI_post180']
        
    if t == 0:
        data67['TOA_prev75'] = toa_prev75
        data67['KT_prev75'] = GHI_prev75/ toa_prev75
        data67['TOA_prev60'] = toa_prev60
        data67['KT_prev60'] = GHI_prev60/ toa_prev60
        data67['TOA_prev45'] = toa_prev45
        data67['KT_prev45'] = GHI_prev45/ toa_prev45
        data67['TOA_prev30'] = toa_prev30
        data67['KT_prev30'] = GHI_prev30/ toa_prev30
        data67['TOA_prev15'] = toa_prev15
        data67['KT_prev15'] = GHI_prev15/ toa_prev15
        data67['TOA_post15'] = toa_post15
        data67['KT_post15'] = GHI_post15/ toa_post15
        data67['TOA_post30'] = toa_post30
        data67['KT_post30'] = GHI_post30/ toa_post30
        data67['TOA_post45'] = toa_post45
        data67['KT_post45'] = GHI_post45/ toa_post45
        data67['TOA_post60'] = toa_post60
        data67['KT_post60'] = GHI_post60/ toa_post60
        data67['TOA_post75'] = toa_post75
        data67['KT_post75'] = GHI_post75/ toa_post75
        data67['TOA_post90'] = toa_post90
        data67['KT_post90'] = GHI_post90/ toa_post90
        data67['TOA_post105'] = toa_post105
        data67['KT_post105'] = GHI_post105/ toa_post105
        data67['TOA_post120'] = toa_post120
        data67['KT_post120'] = GHI_post120/ toa_post120
        data67['TOA_post135'] = toa_post135
        data67['KT_post135'] = GHI_post135/ toa_post135
        data67['TOA_post150'] = toa_post150
        data67['KT_post150'] = GHI_post150/ toa_post150
        data67['TOA_post165'] = toa_post165
        data67['KT_post165'] = GHI_post165/ toa_post165
        data67['TOA_post180'] = toa_post180
        data67['KT_post180'] = GHI_post180/ toa_post180
        
    if t == 1:
        data68['TOA_prev75'] = toa_prev75
        data68['KT_prev75'] = GHI_prev75/ toa_prev75
        data68['TOA_prev60'] = toa_prev60
        data68['KT_prev60'] = GHI_prev60/ toa_prev60
        data68['TOA_prev45'] = toa_prev45
        data68['KT_prev45'] = GHI_prev45/ toa_prev45
        data68['TOA_prev30'] = toa_prev30
        data68['KT_prev30'] = GHI_prev30/ toa_prev30
        data68['TOA_prev15'] = toa_prev15
        data68['KT_prev15'] = GHI_prev15/ toa_prev15
        data68['TOA_post15'] = toa_post15
        data68['KT_post15'] = GHI_post15/ toa_post15
        data68['TOA_post30'] = toa_post30
        data68['KT_post30'] = GHI_post30/ toa_post30
        data68['TOA_post45'] = toa_post45
        data68['KT_post45'] = GHI_post45/ toa_post45
        data68['TOA_post60'] = toa_post60
        data68['KT_post60'] = GHI_post60/ toa_post60
        data68['TOA_post75'] = toa_post75
        data68['KT_post75'] = GHI_post75/ toa_post75
        data68['TOA_post90'] = toa_post90
        data68['KT_post90'] = GHI_post90/ toa_post90
        data68['TOA_post105'] = toa_post105
        data68['KT_post105'] = GHI_post105/ toa_post105
        data68['TOA_post120'] = toa_post120
        data68['KT_post120'] = GHI_post120/ toa_post120
        data68['TOA_post135'] = toa_post135
        data68['KT_post135'] = GHI_post135/ toa_post135
        data68['TOA_post150'] = toa_post150
        data68['KT_post150'] = GHI_post150/ toa_post150
        data68['TOA_post165'] = toa_post165
        data68['KT_post165'] = GHI_post165/ toa_post165
        data68['TOA_post180'] = toa_post180
        data68['KT_post180'] = GHI_post180/ toa_post180

    if t == 2:
        data69['TOA_prev75'] = toa_prev75
        data69['KT_prev75'] = GHI_prev75/ toa_prev75
        data69['TOA_prev60'] = toa_prev60
        data69['KT_prev60'] = GHI_prev60/ toa_prev60
        data69['TOA_prev45'] = toa_prev45
        data69['KT_prev45'] = GHI_prev45/ toa_prev45
        data69['TOA_prev30'] = toa_prev30
        data69['KT_prev30'] = GHI_prev30/ toa_prev30
        data69['TOA_prev15'] = toa_prev15
        data69['KT_prev15'] = GHI_prev15/ toa_prev15
        data69['TOA_post15'] = toa_post15
        data69['KT_post15'] = GHI_post15/ toa_post15
        data69['TOA_post30'] = toa_post30
        data69['KT_post30'] = GHI_post30/ toa_post30
        data69['TOA_post45'] = toa_post45
        data69['KT_post45'] = GHI_post45/ toa_post45
        data69['TOA_post60'] = toa_post60
        data69['KT_post60'] = GHI_post60/ toa_post60
        data69['TOA_post75'] = toa_post75
        data69['KT_post75'] = GHI_post75/ toa_post75
        data69['TOA_post90'] = toa_post90
        data69['KT_post90'] = GHI_post90/ toa_post90
        data69['TOA_post105'] = toa_post105
        data69['KT_post105'] = GHI_post105/ toa_post105
        data69['TOA_post120'] = toa_post120
        data69['KT_post120'] = GHI_post120/ toa_post120
        data69['TOA_post135'] = toa_post135
        data69['KT_post135'] = GHI_post135/ toa_post135
        data69['TOA_post150'] = toa_post150
        data69['KT_post150'] = GHI_post150/ toa_post150
        data69['TOA_post165'] = toa_post165
        data69['KT_post165'] = GHI_post165/ toa_post165
        data69['TOA_post180'] = toa_post180
        data69['KT_post180'] = GHI_post180/ toa_post180

    if t == 3:
        data70['TOA_prev75'] = toa_prev75
        data70['KT_prev75'] = GHI_prev75/ toa_prev75
        data70['TOA_prev60'] = toa_prev60
        data70['KT_prev60'] = GHI_prev60/ toa_prev60
        data70['TOA_prev45'] = toa_prev45
        data70['KT_prev45'] = GHI_prev45/ toa_prev45
        data70['TOA_prev30'] = toa_prev30
        data70['KT_prev30'] = GHI_prev30/ toa_prev30
        data70['TOA_prev15'] = toa_prev15
        data70['KT_prev15'] = GHI_prev15/ toa_prev15
        data70['TOA_post15'] = toa_post15
        data70['KT_post15'] = GHI_post15/ toa_post15
        data70['TOA_post30'] = toa_post30
        data70['KT_post30'] = GHI_post30/ toa_post30
        data70['TOA_post45'] = toa_post45
        data70['KT_post45'] = GHI_post45/ toa_post45
        data70['TOA_post60'] = toa_post60
        data70['KT_post60'] = GHI_post60/ toa_post60
        data70['TOA_post75'] = toa_post75
        data70['KT_post75'] = GHI_post75/ toa_post75
        data70['TOA_post90'] = toa_post90
        data70['KT_post90'] = GHI_post90/ toa_post90
        data70['TOA_post105'] = toa_post105
        data70['KT_post105'] = GHI_post105/ toa_post105
        data70['TOA_post120'] = toa_post120
        data70['KT_post120'] = GHI_post120/ toa_post120
        data70['TOA_post135'] = toa_post135
        data70['KT_post135'] = GHI_post135/ toa_post135
        data70['TOA_post150'] = toa_post150
        data70['KT_post150'] = GHI_post150/ toa_post150
        data70['TOA_post165'] = toa_post165
        data70['KT_post165'] = GHI_post165/ toa_post165
        data70['TOA_post180'] = toa_post180
        data70['KT_post180'] = GHI_post180/ toa_post180

    if t == 4:
        data71['TOA_prev75'] = toa_prev75
        data71['KT_prev75'] = GHI_prev75/ toa_prev75
        data71['TOA_prev60'] = toa_prev60
        data71['KT_prev60'] = GHI_prev60/ toa_prev60
        data71['TOA_prev45'] = toa_prev45
        data71['KT_prev45'] = GHI_prev45/ toa_prev45
        data71['TOA_prev30'] = toa_prev30
        data71['KT_prev30'] = GHI_prev30/ toa_prev30
        data71['TOA_prev15'] = toa_prev15
        data71['KT_prev15'] = GHI_prev15/ toa_prev15
        data71['TOA_post15'] = toa_post15
        data71['KT_post15'] = GHI_post15/ toa_post15
        data71['TOA_post30'] = toa_post30
        data71['KT_post30'] = GHI_post30/ toa_post30
        data71['TOA_post45'] = toa_post45
        data71['KT_post45'] = GHI_post45/ toa_post45
        data71['TOA_post60'] = toa_post60
        data71['KT_post60'] = GHI_post60/ toa_post60
        data71['TOA_post75'] = toa_post75
        data71['KT_post75'] = GHI_post75/ toa_post75
        data71['TOA_post90'] = toa_post90
        data71['KT_post90'] = GHI_post90/ toa_post90
        data71['TOA_post105'] = toa_post105
        data71['KT_post105'] = GHI_post105/ toa_post105
        data71['TOA_post120'] = toa_post120
        data71['KT_post120'] = GHI_post120/ toa_post120
        data71['TOA_post135'] = toa_post135
        data71['KT_post135'] = GHI_post135/ toa_post135
        data71['TOA_post150'] = toa_post150
        data71['KT_post150'] = GHI_post150/ toa_post150
        data71['TOA_post165'] = toa_post165
        data71['KT_post165'] = GHI_post165/ toa_post165
        data71['TOA_post180'] = toa_post180
        data71['KT_post180'] = GHI_post180/ toa_post180
        
    if t == 5:
        data72['TOA_prev75'] = toa_prev75
        data72['KT_prev75'] = GHI_prev75/ toa_prev75
        data72['TOA_prev60'] = toa_prev60
        data72['KT_prev60'] = GHI_prev60/ toa_prev60
        data72['TOA_prev45'] = toa_prev45
        data72['KT_prev45'] = GHI_prev45/ toa_prev45
        data72['TOA_prev30'] = toa_prev30
        data72['KT_prev30'] = GHI_prev30/ toa_prev30
        data72['TOA_prev15'] = toa_prev15
        data72['KT_prev15'] = GHI_prev15/ toa_prev15
        data72['TOA_post15'] = toa_post15
        data72['KT_post15'] = GHI_post15/ toa_post15
        data72['TOA_post30'] = toa_post30
        data72['KT_post30'] = GHI_post30/ toa_post30
        data72['TOA_post45'] = toa_post45
        data72['KT_post45'] = GHI_post45/ toa_post45
        data72['TOA_post60'] = toa_post60
        data72['KT_post60'] = GHI_post60/ toa_post60
        data72['TOA_post75'] = toa_post75
        data72['KT_post75'] = GHI_post75/ toa_post75
        data72['TOA_post90'] = toa_post90
        data72['KT_post90'] = GHI_post90/ toa_post90
        data72['TOA_post105'] = toa_post105
        data72['KT_post105'] = GHI_post105/ toa_post105
        data72['TOA_post120'] = toa_post120
        data72['KT_post120'] = GHI_post120/ toa_post120
        data72['TOA_post135'] = toa_post135
        data72['KT_post135'] = GHI_post135/ toa_post135
        data72['TOA_post150'] = toa_post150
        data72['KT_post150'] = GHI_post150/ toa_post150
        data72['TOA_post165'] = toa_post165
        data72['KT_post165'] = GHI_post165/ toa_post165
        data72['TOA_post180'] = toa_post180
        data72['KT_post180'] = GHI_post180/ toa_post180
        
    if t == 6:
        data73['TOA_prev75'] = toa_prev75
        data73['KT_prev75'] = GHI_prev75/ toa_prev75
        data73['TOA_prev60'] = toa_prev60
        data73['KT_prev60'] = GHI_prev60/ toa_prev60
        data73['TOA_prev45'] = toa_prev45
        data73['KT_prev45'] = GHI_prev45/ toa_prev45
        data73['TOA_prev30'] = toa_prev30
        data73['KT_prev30'] = GHI_prev30/ toa_prev30
        data73['TOA_prev15'] = toa_prev15
        data73['KT_prev15'] = GHI_prev15/ toa_prev15
        data73['TOA_post15'] = toa_post15
        data73['KT_post15'] = GHI_post15/ toa_post15
        data73['TOA_post30'] = toa_post30
        data73['KT_post30'] = GHI_post30/ toa_post30
        data73['TOA_post45'] = toa_post45
        data73['KT_post45'] = GHI_post45/ toa_post45
        data73['TOA_post60'] = toa_post60
        data73['KT_post60'] = GHI_post60/ toa_post60
        data73['TOA_post75'] = toa_post75
        data73['KT_post75'] = GHI_post75/ toa_post75
        data73['TOA_post90'] = toa_post90
        data73['KT_post90'] = GHI_post90/ toa_post90
        data73['TOA_post105'] = toa_post105
        data73['KT_post105'] = GHI_post105/ toa_post105
        data73['TOA_post120'] = toa_post120
        data73['KT_post120'] = GHI_post120/ toa_post120
        data73['TOA_post135'] = toa_post135
        data73['KT_post135'] = GHI_post135/ toa_post135
        data73['TOA_post150'] = toa_post150
        data73['KT_post150'] = GHI_post150/ toa_post150
        data73['TOA_post165'] = toa_post165
        data73['KT_post165'] = GHI_post165/ toa_post165
        data73['TOA_post180'] = toa_post180
        data73['KT_post180'] = GHI_post180/ toa_post180
                        
    if t == 7:
        data74['TOA_prev75'] = toa_prev75
        data74['KT_prev75'] = GHI_prev75/ toa_prev75
        data74['TOA_prev60'] = toa_prev60
        data74['KT_prev60'] = GHI_prev60/ toa_prev60
        data74['TOA_prev45'] = toa_prev45
        data74['KT_prev45'] = GHI_prev45/ toa_prev45
        data74['TOA_prev30'] = toa_prev30
        data74['KT_prev30'] = GHI_prev30/ toa_prev30
        data74['TOA_prev15'] = toa_prev15
        data74['KT_prev15'] = GHI_prev15/ toa_prev15
        data74['TOA_post15'] = toa_post15
        data74['KT_post15'] = GHI_post15/ toa_post15
        data74['TOA_post30'] = toa_post30
        data74['KT_post30'] = GHI_post30/ toa_post30
        data74['TOA_post45'] = toa_post45
        data74['KT_post45'] = GHI_post45/ toa_post45
        data74['TOA_post60'] = toa_post60
        data74['KT_post60'] = GHI_post60/ toa_post60
        data74['TOA_post75'] = toa_post75
        data74['KT_post75'] = GHI_post75/ toa_post75
        data74['TOA_post90'] = toa_post90
        data74['KT_post90'] = GHI_post90/ toa_post90
        data74['TOA_post105'] = toa_post105
        data74['KT_post105'] = GHI_post105/ toa_post105
        data74['TOA_post120'] = toa_post120
        data74['KT_post120'] = GHI_post120/ toa_post120
        data74['TOA_post135'] = toa_post135
        data74['KT_post135'] = GHI_post135/ toa_post135
        data74['TOA_post150'] = toa_post150
        data74['KT_post150'] = GHI_post150/ toa_post150
        data74['TOA_post165'] = toa_post165
        data74['KT_post165'] = GHI_post165/ toa_post165
        data74['TOA_post180'] = toa_post180
        data74['KT_post180'] = GHI_post180/ toa_post180
        

for t in xrange(0,8):
    data = []
    if t == 0:
        data = data67
    if t == 1:
        data = data68
    if t == 2:
        data = data69
    if t == 3:
        data = data70
    if t == 4:
        data = data71
    if t == 5:
        data = data72
    if t == 6:
        data = data73
    if t == 7:
        data = data74
    sLength = len(data)
    SLOPE = []
    cor = []
    first_derivative = []
    second_derivative = []
    temp_prev1Hr = []
    temp_post1Hr = []
    temp_post2Hr = []
    temp_post3Hr = []
    for x in xrange(0,sLength):
        tmp = [[data.KT_prev60[x],data.KT_prev45[x],data.KT_prev30[x],data.KT_prev15[x]]]
        y = [1,2,3,4]
        slope, intercept, r_value, p_value, std_err = stats.linregress(y,tmp)
        SLOPE.append(slope)
        cor.append(r_value**2)
        ##X = np.arange(4)
        ##signal = intercept + slope*X
        ##first_derivative[x] = np.gradient(signal)
        ##second_derivative[x] = np.gradient(first_derivative)
        temp_prev1Hr.append(np.std([[data.KT_prev60[x],data.KT_prev45[x],data.KT_prev30[x],data.KT_prev15[x]]]))
        temp_post1Hr.append(np.std([[data.KT_post60[x],data.KT_post45[x],data.KT_post30[x],data.KT_post15[x]]]))
        temp_post2Hr.append(np.std([[data.KT_post120[x],data.KT_post105[x],data.KT_post90[x],data.KT_post75[x]]]))
        temp_post3Hr.append(np.std([[data.KT_post135[x],data.KT_post150[x],data.KT_post165[x],data.KT_post180[x]]]))
        
    if t == 0:
        data67['KT_Slope'] = SLOPE
        data67['R2'] = cor
        ##data67['Derive1'] = first_derivative
        ##data67['Derive2'] = second_derivative
        data67['std_prev1Hr'] = temp_prev1Hr
        data67['std_post1Hr'] = temp_post1Hr
        data67['std_post2Hr'] = temp_post2Hr
        data67['std_post3Hr'] = temp_post3Hr
    if t == 1:
        data68['KT_Slope'] = SLOPE
        data68['R2'] = cor
        ##data68['Derive1'] = first_derivative
        ##data68['Derive2'] = second_derivative
        data68['std_prev1Hr'] = temp_prev1Hr
        data68['std_post1Hr'] = temp_post1Hr
        data68['std_post2Hr'] = temp_post2Hr
        data68['std_post3Hr'] = temp_post3Hr
    if t == 2:
        data69['KT_Slope'] = SLOPE
        data69['R2'] = cor
       ## data69['Derive1'] = first_derivative
       ## data69['Derive2'] = second_derivative
        data69['std_prev1Hr'] = temp_prev1Hr
        data69['std_post1Hr'] = temp_post1Hr
        data69['std_post2Hr'] = temp_post2Hr
        data69['std_post3Hr'] = temp_post3Hr
    if t == 3:
        data70['KT_Slope'] = SLOPE
        data70['R2'] = cor
       ## data70['Derive1'] = first_derivative
       ## data70['Derive2'] = second_derivative
        data70['std_prev1Hr'] = temp_prev1Hr
        data70['std_post1Hr'] = temp_post1Hr
        data70['std_post2Hr'] = temp_post2Hr
        data70['std_post3Hr'] = temp_post3Hr
    if t == 4:
        data71['KT_Slope'] = SLOPE
        data71['R2'] = cor
       ## data71['Derive1'] = first_derivative
       ## data71['Derive2'] = second_derivative
        data71['std_prev1Hr'] = temp_prev1Hr
        data71['std_post1Hr'] = temp_post1Hr
        data71['std_post2Hr'] = temp_post2Hr
        data71['std_post3Hr'] = temp_post3Hr
    if t == 5:
        data72['KT_Slope'] = SLOPE
        data72['R2'] = cor
       ## data72['Derive1'] = first_derivative
       ## data72['Derive2'] = second_derivative
        data72['std_prev1Hr'] = temp_prev1Hr
        data72['std_post1Hr'] = temp_post1Hr
        data72['std_post2Hr'] = temp_post2Hr
        data72['std_post3Hr'] = temp_post3Hr
    if t == 6:
        data73['KT_Slope'] = SLOPE
        data73['R2'] = cor
       ## data73['Derive1'] = first_derivative
       ## data73['Derive2'] = second_derivative
        data73['std_prev1Hr'] = temp_prev1Hr
        data73['std_post1Hr'] = temp_post1Hr
        data73['std_post2Hr'] = temp_post2Hr
        data73['std_post3Hr'] = temp_post3Hr
    if t == 7:
        data74['KT_Slope'] = SLOPE
        data74['R2'] = cor
       ## data74['Derive1'] = first_derivative
       ## data74['Derive2'] = second_derivative
        data74['std_prev1Hr'] = temp_prev1Hr
        data74['std_post1Hr'] = temp_post1Hr
        data74['std_post2Hr'] = temp_post2Hr
        data74['std_post3Hr'] = temp_post3Hr
## Delete all rows with NaNs (which would occur when TOA is predicted to be 0...
data67 = data67.dropna().reset_index(drop=True)
data68 = data68.dropna().reset_index(drop=True)
data69 = data69.dropna().reset_index(drop=True)
data70 = data70.dropna().reset_index(drop=True)
data71 = data71.dropna().reset_index(drop=True)
data72 = data72.dropna().reset_index(drop=True)
data73 = data73.dropna().reset_index(drop=True)
data74 = data74.dropna().reset_index(drop=True)

## Delete all rows where the clearness index is above 1.0 or below 0.0...
data67 = data67[data67.KT_prev60 >0.01].reset_index(drop=True)
data67 = data67[data67.KT_prev60 < 1.00].reset_index(drop=True)
data67 = data67[data67.KT_prev45 >0.01].reset_index(drop=True)
data67 = data67[data67.KT_prev45 < 1.00].reset_index(drop=True)
data67 = data67[data67.KT_prev30 >0.01].reset_index(drop=True)
data67 = data67[data67.KT_prev30 < 1.00].reset_index(drop=True)
data67 = data67[data67.KT_prev15 >0.01].reset_index(drop=True)
data67 = data67[data67.KT_prev15 < 1.00].reset_index(drop=True)
data67 = data67[data67.KT_post15 >0.01].reset_index(drop=True)
data67 = data67[data67.KT_post15 < 1.00].reset_index(drop=True)
data67 = data67[data67.KT_post30 >0.01].reset_index(drop=True)
data67 = data67[data67.KT_post30 < 1.00].reset_index(drop=True)
data67 = data67[data67.KT_post45 >0.01].reset_index(drop=True)
data67 = data67[data67.KT_post45 < 1.00].reset_index(drop=True)
data67 = data67[data67.KT_post60 >0.01].reset_index(drop=True)
data67 = data67[data67.KT_post60 < 1.00].reset_index(drop=True)
data67 = data67[data67.KT_post75 >0.01].reset_index(drop=True)
data67 = data67[data67.KT_post75 < 1.00].reset_index(drop=True)
data67 = data67[data67.KT_post90 >0.01].reset_index(drop=True)
data67 = data67[data67.KT_post90 < 1.00].reset_index(drop=True)
data67 = data67[data67.KT_post105 >0.01].reset_index(drop=True)
data67 = data67[data67.KT_post105 < 1.00].reset_index(drop=True)
data67 = data67[data67.KT_post120 >0.01].reset_index(drop=True)
data67 = data67[data67.KT_post120 < 1.00].reset_index(drop=True)
data67 = data67[data67.KT_post135 >0.01].reset_index(drop=True)
data67 = data67[data67.KT_post135 < 1.00].reset_index(drop=True)
data67 = data67[data67.KT_post150 >0.01].reset_index(drop=True)
data67 = data67[data67.KT_post150 < 1.00].reset_index(drop=True)
data67 = data67[data67.KT_post165 >0.01].reset_index(drop=True)
data67 = data67[data67.KT_post165 < 1.00].reset_index(drop=True)
data67 = data67[data67.KT_post180 >0.01].reset_index(drop=True)
data67 = data67[data67.KT_post180 < 1.00].reset_index(drop=True)

## Delete all rows where the clearness index is above 1.0 or below 0.0...
data68 = data68[data68.KT_prev60 >0.01].reset_index(drop=True)
data68 = data68[data68.KT_prev60 < 1.00].reset_index(drop=True)
data68 = data68[data68.KT_prev45 >0.01].reset_index(drop=True)
data68 = data68[data68.KT_prev45 < 1.00].reset_index(drop=True)
data68 = data68[data68.KT_prev30 >0.01].reset_index(drop=True)
data68 = data68[data68.KT_prev30 < 1.00].reset_index(drop=True)
data68 = data68[data68.KT_prev15 >0.01].reset_index(drop=True)
data68 = data68[data68.KT_prev15 < 1.00].reset_index(drop=True)
data68 = data68[data68.KT_post15 >0.01].reset_index(drop=True)
data68 = data68[data68.KT_post15 < 1.00].reset_index(drop=True)
data68 = data68[data68.KT_post30 >0.01].reset_index(drop=True)
data68 = data68[data68.KT_post30 < 1.00].reset_index(drop=True)
data68 = data68[data68.KT_post45 >0.01].reset_index(drop=True)
data68 = data68[data68.KT_post45 < 1.00].reset_index(drop=True)
data68 = data68[data68.KT_post60 >0.01].reset_index(drop=True)
data68 = data68[data68.KT_post60 < 1.00].reset_index(drop=True)
data68 = data68[data68.KT_post75 >0.01].reset_index(drop=True)
data68 = data68[data68.KT_post75 < 1.00].reset_index(drop=True)
data68 = data68[data68.KT_post90 >0.01].reset_index(drop=True)
data68 = data68[data68.KT_post90 < 1.00].reset_index(drop=True)
data68 = data68[data68.KT_post105 >0.01].reset_index(drop=True)
data68 = data68[data68.KT_post105 < 1.00].reset_index(drop=True)
data68 = data68[data68.KT_post120 >0.01].reset_index(drop=True)
data68 = data68[data68.KT_post120 < 1.00].reset_index(drop=True)
data68 = data68[data68.KT_post135 >0.01].reset_index(drop=True)
data68 = data68[data68.KT_post135 < 1.00].reset_index(drop=True)
data68 = data68[data68.KT_post150 >0.01].reset_index(drop=True)
data68 = data68[data68.KT_post150 < 1.00].reset_index(drop=True)
data68 = data68[data68.KT_post165 >0.01].reset_index(drop=True)
data68 = data68[data68.KT_post165 < 1.00].reset_index(drop=True)
data68 = data68[data68.KT_post180 >0.01].reset_index(drop=True)
data68 = data68[data68.KT_post180 < 1.00].reset_index(drop=True)

## Delete all rows where the clearness index is above 1.0 or below 0.0...
data69 = data69[data69.KT_prev60 >0.01].reset_index(drop=True)
data69 = data69[data69.KT_prev60 < 1.00].reset_index(drop=True)
data69 = data69[data69.KT_prev45 >0.01].reset_index(drop=True)
data69 = data69[data69.KT_prev45 < 1.00].reset_index(drop=True)
data69 = data69[data69.KT_prev30 >0.01].reset_index(drop=True)
data69 = data69[data69.KT_prev30 < 1.00].reset_index(drop=True)
data69 = data69[data69.KT_prev15 >0.01].reset_index(drop=True)
data69 = data69[data69.KT_prev15 < 1.00].reset_index(drop=True)
data69 = data69[data69.KT_post15 >0.01].reset_index(drop=True)
data69 = data69[data69.KT_post15 < 1.00].reset_index(drop=True)
data69 = data69[data69.KT_post30 >0.01].reset_index(drop=True)
data69 = data69[data69.KT_post30 < 1.00].reset_index(drop=True)
data69 = data69[data69.KT_post45 >0.01].reset_index(drop=True)
data69 = data69[data69.KT_post45 < 1.00].reset_index(drop=True)
data69 = data69[data69.KT_post60 >0.01].reset_index(drop=True)
data69 = data69[data69.KT_post60 < 1.00].reset_index(drop=True)
data69 = data69[data69.KT_post75 >0.01].reset_index(drop=True)
data69 = data69[data69.KT_post75 < 1.00].reset_index(drop=True)
data69 = data69[data69.KT_post90 >0.01].reset_index(drop=True)
data69 = data69[data69.KT_post90 < 1.00].reset_index(drop=True)
data69 = data69[data69.KT_post105 >0.01].reset_index(drop=True)
data69 = data69[data69.KT_post105 < 1.00].reset_index(drop=True)
data69 = data69[data69.KT_post120 >0.01].reset_index(drop=True)
data69 = data69[data69.KT_post120 < 1.00].reset_index(drop=True)
data69 = data69[data69.KT_post135 >0.01].reset_index(drop=True)
data69 = data69[data69.KT_post135 < 1.00].reset_index(drop=True)
data69 = data69[data69.KT_post150 >0.01].reset_index(drop=True)
data69 = data69[data69.KT_post150 < 1.00].reset_index(drop=True)
data69 = data69[data69.KT_post165 >0.01].reset_index(drop=True)
data69 = data69[data69.KT_post165 < 1.00].reset_index(drop=True)
data69 = data69[data69.KT_post180 >0.01].reset_index(drop=True)
data69 = data69[data69.KT_post180 < 1.00].reset_index(drop=True)

## Delete all rows where the clearness index is above 1.0 or below 0.0...
data70 = data70[data70.KT_prev60 >0.01].reset_index(drop=True)
data70 = data70[data70.KT_prev60 < 1.00].reset_index(drop=True)
data70 = data70[data70.KT_prev45 >0.01].reset_index(drop=True)
data70 = data70[data70.KT_prev45 < 1.00].reset_index(drop=True)
data70 = data70[data70.KT_prev30 >0.01].reset_index(drop=True)
data70 = data70[data70.KT_prev30 < 1.00].reset_index(drop=True)
data70 = data70[data70.KT_prev15 >0.01].reset_index(drop=True)
data70 = data70[data70.KT_prev15 < 1.00].reset_index(drop=True)
data70 = data70[data70.KT_post15 >0.01].reset_index(drop=True)
data70 = data70[data70.KT_post15 < 1.00].reset_index(drop=True)
data70 = data70[data70.KT_post30 >0.01].reset_index(drop=True)
data70 = data70[data70.KT_post30 < 1.00].reset_index(drop=True)
data70 = data70[data70.KT_post45 >0.01].reset_index(drop=True)
data70 = data70[data70.KT_post45 < 1.00].reset_index(drop=True)
data70 = data70[data70.KT_post60 >0.01].reset_index(drop=True)
data70 = data70[data70.KT_post60 < 1.00].reset_index(drop=True)
data70 = data70[data70.KT_post75 >0.01].reset_index(drop=True)
data70 = data70[data70.KT_post75 < 1.00].reset_index(drop=True)
data70 = data70[data70.KT_post90 >0.01].reset_index(drop=True)
data70 = data70[data70.KT_post90 < 1.00].reset_index(drop=True)
data70 = data70[data70.KT_post105 >0.01].reset_index(drop=True)
data70 = data70[data70.KT_post105 < 1.00].reset_index(drop=True)
data70 = data70[data70.KT_post120 >0.01].reset_index(drop=True)
data70 = data70[data70.KT_post120 < 1.00].reset_index(drop=True)
data70 = data70[data70.KT_post135 >0.01].reset_index(drop=True)
data70 = data70[data70.KT_post135 < 1.00].reset_index(drop=True)
data70 = data70[data70.KT_post150 >0.01].reset_index(drop=True)
data70 = data70[data70.KT_post150 < 1.00].reset_index(drop=True)
data70 = data70[data70.KT_post165 >0.01].reset_index(drop=True)
data70 = data70[data70.KT_post165 < 1.00].reset_index(drop=True)
data70 = data70[data70.KT_post180 >0.01].reset_index(drop=True)
data70 = data70[data70.KT_post180 < 1.00].reset_index(drop=True)

## Delete all rows where the clearness index is above 1.0 or below 0.0...
data71 = data71[data71.KT_prev60 >0.01].reset_index(drop=True)
data71 = data71[data71.KT_prev60 < 1.00].reset_index(drop=True)
data71 = data71[data71.KT_prev45 >0.01].reset_index(drop=True)
data71 = data71[data71.KT_prev45 < 1.00].reset_index(drop=True)
data71 = data71[data71.KT_prev30 >0.01].reset_index(drop=True)
data71 = data71[data71.KT_prev30 < 1.00].reset_index(drop=True)
data71 = data71[data71.KT_prev15 >0.01].reset_index(drop=True)
data71 = data71[data71.KT_prev15 < 1.00].reset_index(drop=True)
data71 = data71[data71.KT_post15 >0.01].reset_index(drop=True)
data71 = data71[data71.KT_post15 < 1.00].reset_index(drop=True)
data71 = data71[data71.KT_post30 >0.01].reset_index(drop=True)
data71 = data71[data71.KT_post30 < 1.00].reset_index(drop=True)
data71 = data71[data71.KT_post45 >0.01].reset_index(drop=True)
data71 = data71[data71.KT_post45 < 1.00].reset_index(drop=True)
data71 = data71[data71.KT_post60 >0.01].reset_index(drop=True)
data71 = data71[data71.KT_post60 < 1.00].reset_index(drop=True)
data71 = data71[data71.KT_post75 >0.01].reset_index(drop=True)
data71 = data71[data71.KT_post75 < 1.00].reset_index(drop=True)
data71 = data71[data71.KT_post90 >0.01].reset_index(drop=True)
data71 = data71[data71.KT_post90 < 1.00].reset_index(drop=True)
data71 = data71[data71.KT_post105 >0.01].reset_index(drop=True)
data71 = data71[data71.KT_post105 < 1.00].reset_index(drop=True)
data71 = data71[data71.KT_post120 >0.01].reset_index(drop=True)
data71 = data71[data71.KT_post120 < 1.00].reset_index(drop=True)
data71 = data71[data71.KT_post135 >0.01].reset_index(drop=True)
data71 = data71[data71.KT_post135 < 1.00].reset_index(drop=True)
data71 = data71[data71.KT_post150 >0.01].reset_index(drop=True)
data71 = data71[data71.KT_post150 < 1.00].reset_index(drop=True)
data71 = data71[data71.KT_post165 >0.01].reset_index(drop=True)
data71 = data71[data71.KT_post165 < 1.00].reset_index(drop=True)
data71 = data71[data71.KT_post180 >0.01].reset_index(drop=True)
data71 = data71[data71.KT_post180 < 1.00].reset_index(drop=True)

## Delete all rows where the clearness index is above 1.0 or below 0.0...
data72 = data72[data72.KT_prev60 >0.01].reset_index(drop=True)
data72 = data72[data72.KT_prev60 < 1.00].reset_index(drop=True)
data72 = data72[data72.KT_prev45 >0.01].reset_index(drop=True)
data72 = data72[data72.KT_prev45 < 1.00].reset_index(drop=True)
data72 = data72[data72.KT_prev30 >0.01].reset_index(drop=True)
data72 = data72[data72.KT_prev30 < 1.00].reset_index(drop=True)
data72 = data72[data72.KT_prev15 >0.01].reset_index(drop=True)
data72 = data72[data72.KT_prev15 < 1.00].reset_index(drop=True)
data72 = data72[data72.KT_post15 >0.01].reset_index(drop=True)
data72 = data72[data72.KT_post15 < 1.00].reset_index(drop=True)
data72 = data72[data72.KT_post30 >0.01].reset_index(drop=True)
data72 = data72[data72.KT_post30 < 1.00].reset_index(drop=True)
data72 = data72[data72.KT_post45 >0.01].reset_index(drop=True)
data72 = data72[data72.KT_post45 < 1.00].reset_index(drop=True)
data72 = data72[data72.KT_post60 >0.01].reset_index(drop=True)
data72 = data72[data72.KT_post60 < 1.00].reset_index(drop=True)
data72 = data72[data72.KT_post75 >0.01].reset_index(drop=True)
data72 = data72[data72.KT_post75 < 1.00].reset_index(drop=True)
data72 = data72[data72.KT_post90 >0.01].reset_index(drop=True)
data72 = data72[data72.KT_post90 < 1.00].reset_index(drop=True)
data72 = data72[data72.KT_post105 >0.01].reset_index(drop=True)
data72 = data72[data72.KT_post105 < 1.00].reset_index(drop=True)
data72 = data72[data72.KT_post120 >0.01].reset_index(drop=True)
data72 = data72[data72.KT_post120 < 1.00].reset_index(drop=True)
data72 = data72[data72.KT_post135 >0.01].reset_index(drop=True)
data72 = data72[data72.KT_post135 < 1.00].reset_index(drop=True)
data72 = data72[data72.KT_post150 >0.01].reset_index(drop=True)
data72 = data72[data72.KT_post150 < 1.00].reset_index(drop=True)
data72 = data72[data72.KT_post165 >0.01].reset_index(drop=True)
data72 = data72[data72.KT_post165 < 1.00].reset_index(drop=True)
data72 = data72[data72.KT_post180 >0.01].reset_index(drop=True)
data72 = data72[data72.KT_post180 < 1.00].reset_index(drop=True)


## Delete all rows where the clearness index is above 1.0 or below 0.0...
data73 = data73[data73.KT_prev60 >0.01].reset_index(drop=True)
data73 = data73[data73.KT_prev60 < 1.00].reset_index(drop=True)
data73 = data73[data73.KT_prev45 >0.01].reset_index(drop=True)
data73 = data73[data73.KT_prev45 < 1.00].reset_index(drop=True)
data73 = data73[data73.KT_prev30 >0.01].reset_index(drop=True)
data73 = data73[data73.KT_prev30 < 1.00].reset_index(drop=True)
data73 = data73[data73.KT_prev15 >0.01].reset_index(drop=True)
data73 = data73[data73.KT_prev15 < 1.00].reset_index(drop=True)
data73 = data73[data73.KT_post15 >0.01].reset_index(drop=True)
data73 = data73[data73.KT_post15 < 1.00].reset_index(drop=True)
data73 = data73[data73.KT_post30 >0.01].reset_index(drop=True)
data73 = data73[data73.KT_post30 < 1.00].reset_index(drop=True)
data73 = data73[data73.KT_post45 >0.01].reset_index(drop=True)
data73 = data73[data73.KT_post45 < 1.00].reset_index(drop=True)
data73 = data73[data73.KT_post60 >0.01].reset_index(drop=True)
data73 = data73[data73.KT_post60 < 1.00].reset_index(drop=True)
data73 = data73[data73.KT_post75 >0.01].reset_index(drop=True)
data73 = data73[data73.KT_post75 < 1.00].reset_index(drop=True)
data73 = data73[data73.KT_post90 >0.01].reset_index(drop=True)
data73 = data73[data73.KT_post90 < 1.00].reset_index(drop=True)
data73 = data73[data73.KT_post105 >0.01].reset_index(drop=True)
data73 = data73[data73.KT_post105 < 1.00].reset_index(drop=True)
data73 = data73[data73.KT_post120 >0.01].reset_index(drop=True)
data73 = data73[data73.KT_post120 < 1.00].reset_index(drop=True)
data73 = data73[data73.KT_post135 >0.01].reset_index(drop=True)
data73 = data73[data73.KT_post135 < 1.00].reset_index(drop=True)
data73 = data73[data73.KT_post150 >0.01].reset_index(drop=True)
data73 = data73[data73.KT_post150 < 1.00].reset_index(drop=True)
data73 = data73[data73.KT_post165 >0.01].reset_index(drop=True)
data73 = data73[data73.KT_post165 < 1.00].reset_index(drop=True)
data73 = data73[data73.KT_post180 >0.01].reset_index(drop=True)
data73 = data73[data73.KT_post180 < 1.00].reset_index(drop=True)

## Delete all rows where the clearness index is above 1.0 or below 0.0...
data74 = data74[data74.KT_prev60 >0.01].reset_index(drop=True)
data74 = data74[data74.KT_prev60 < 1.00].reset_index(drop=True)
data74 = data74[data74.KT_prev45 >0.01].reset_index(drop=True)
data74 = data74[data74.KT_prev45 < 1.00].reset_index(drop=True)
data74 = data74[data74.KT_prev30 >0.01].reset_index(drop=True)
data74 = data74[data74.KT_prev30 < 1.00].reset_index(drop=True)
data74 = data74[data74.KT_prev15 >0.01].reset_index(drop=True)
data74 = data74[data74.KT_prev15 < 1.00].reset_index(drop=True)
data74 = data74[data74.KT_post15 >0.01].reset_index(drop=True)
data74 = data74[data74.KT_post15 < 1.00].reset_index(drop=True)
data74 = data74[data74.KT_post30 >0.01].reset_index(drop=True)
data74 = data74[data74.KT_post30 < 1.00].reset_index(drop=True)
data74 = data74[data74.KT_post45 >0.01].reset_index(drop=True)
data74 = data74[data74.KT_post45 < 1.00].reset_index(drop=True)
data74 = data74[data74.KT_post60 >0.01].reset_index(drop=True)
data74 = data74[data74.KT_post60 < 1.00].reset_index(drop=True)
data74 = data74[data74.KT_post75 >0.01].reset_index(drop=True)
data74 = data74[data74.KT_post75 < 1.00].reset_index(drop=True)
data74 = data74[data74.KT_post90 >0.01].reset_index(drop=True)
data74 = data74[data74.KT_post90 < 1.00].reset_index(drop=True)
data74 = data74[data74.KT_post105 >0.01].reset_index(drop=True)
data74 = data74[data74.KT_post105 < 1.00].reset_index(drop=True)
data74 = data74[data74.KT_post120 >0.01].reset_index(drop=True)
data74 = data74[data74.KT_post120 < 1.00].reset_index(drop=True)
data74 = data74[data74.KT_post135 >0.01].reset_index(drop=True)
data74 = data74[data74.KT_post135 < 1.00].reset_index(drop=True)
data74 = data74[data74.KT_post150 >0.01].reset_index(drop=True)
data74 = data74[data74.KT_post150 < 1.00].reset_index(drop=True)
data74 = data74[data74.KT_post165 >0.01].reset_index(drop=True)
data74 = data74[data74.KT_post165 < 1.00].reset_index(drop=True)
data74 = data74[data74.KT_post180 >0.01].reset_index(drop=True)
data74 = data74[data74.KT_post180 < 1.00].reset_index(drop=True)

## header = ["UnixUtcTime", "UtcYear", "UtcMonth", "UtcDay", "UtcHour", "JulianDay", "MV72483000_T", "MV72483009_T", "MV72483016_T", "MV72483000_dewpt", "MV72483009_dewpt", "MV72483016_dewpt", "MV72483000_cloud_cov", "MV72483009_cloud_cov", "MV72483016_cloud_cov", "MV72483000_prob_precip01", "MV72483009_prob_precip01", "MV72483016_prob_precip01", "MV72483000_qpf01", "MV72483009_qpf01", "MV72483016_qpf01", "MV72483000_wind_speed", "MV72483009_wind_speed", "MV72483016_wind_speed", "GHI_prev75", "GHI_prev60", "GHI_prev45", "GHI_prev30", "GHI_prev15", "GHI_post15", "GHI_post30", "GHI_post45", "GHI_post60", "GHI_post75", "GHI_post90", "GHI_post105", "GHI_post120", "GHI_post135", "GHI_post150", "GHI_post165", "GHI_post180", "T_AVG", "dewpt_AVG", "wind_speed_AVG", "prob_precip01_AVG", "qpf01_AVG", "cloud_cov_AVG", "T_Td", "cloud_cov_STD", "cloud_cov_SQRD", "TOA_prev75", "KT_prev75", "TOA_prev60", "KT_prev60", "TOA_prev45", "KT_prev45", "TOA_prev30", "KT_prev30", "TOA_prev15", "KT_prev15", "TOA_post15", "KT_post15", "TOA_post30", "KT_post30", "TOA_post45", "KT_post45", "TOA_post60", "KT_post60", "TOA_post75", "KT_post75", "TOA_post90", "KT_post90", "TOA_post105", "KT_post105", "TOA_post120", "KT_post120", "TOA_post135", "KT_post135", "TOA_post150", "KT_post150", "TOA_post165", "KT_post165", "TOA_post180", "KT_post180", "KT_Slope", "R2" ]
## data67.to_csv('data67out.csv', cols = header)
## data68.to_csv('data68out.csv', cols = header)
## data69.to_csv('data69out.csv', cols = header)
## data70.to_csv('data70out.csv', cols = header)
## data71.to_csv('data71out.csv', cols = header)
## data72.to_csv('data72out.csv', cols = header)
## data73.to_csv('data73out.csv', cols = header)
## data74.to_csv('data74out.csv', cols = header)

for t in xrange(0,8):
    Nearby = []
    SpatialPre15Std = []
    SpatialPost15Std = []
    SpatialPost60Std = []
    SpatialPost120Std = []
    SpatialPost180Std = []
    if t == 0:
        data = data67
        sLength = len(data)
        for x in xrange(0,sLength):
            Tim = data['UnixUtcTime'][x]
            Nearby.append(np.mean(np.concatenate([data68['KT_prev15'][data68['UnixUtcTime']==Tim].values,
                                   data69['KT_prev15'][data69['UnixUtcTime']==Tim].values,
                                   data70['KT_prev15'][data70['UnixUtcTime']==Tim].values,
                                   data71['KT_prev15'][data71['UnixUtcTime']==Tim].values,
                                   data72['KT_prev15'][data72['UnixUtcTime']==Tim].values,
                                   data73['KT_prev15'][data73['UnixUtcTime']==Tim].values,
                                   data74['KT_prev15'][data74['UnixUtcTime']==Tim].values])))
            SpatialPre15Std.append(np.std(np.concatenate([data68['KT_prev15'][data68['UnixUtcTime']==Tim].values,
                                           data69['KT_prev15'][data69['UnixUtcTime']==Tim].values,
                                           data70['KT_prev15'][data70['UnixUtcTime']==Tim].values,
                                           data71['KT_prev15'][data71['UnixUtcTime']==Tim].values,
                                           data72['KT_prev15'][data72['UnixUtcTime']==Tim].values,
                                           data73['KT_prev15'][data73['UnixUtcTime']==Tim].values,
                                           data74['KT_prev15'][data74['UnixUtcTime']==Tim].values])))
            SpatialPost15Std.append(np.std(np.concatenate([data68['KT_post15'][data68['UnixUtcTime']==Tim].values,
                                            data69['KT_post15'][data69['UnixUtcTime']==Tim].values,
                                            data70['KT_post15'][data70['UnixUtcTime']==Tim].values,
                                            data71['KT_post15'][data71['UnixUtcTime']==Tim].values,
                                            data72['KT_post15'][data72['UnixUtcTime']==Tim].values,
                                            data73['KT_post15'][data73['UnixUtcTime']==Tim].values,
                                            data74['KT_post15'][data74['UnixUtcTime']==Tim].values])))
            SpatialPost60Std.append(np.std(np.concatenate([data68['KT_post60'][data68['UnixUtcTime']==Tim].values,
                                            data69['KT_post60'][data69['UnixUtcTime']==Tim].values,
                                            data70['KT_post60'][data70['UnixUtcTime']==Tim].values,
                                            data71['KT_post60'][data71['UnixUtcTime']==Tim].values,
                                            data72['KT_post60'][data72['UnixUtcTime']==Tim].values,
                                            data73['KT_post60'][data73['UnixUtcTime']==Tim].values,
                                            data74['KT_post60'][data74['UnixUtcTime']==Tim].values])))
            SpatialPost120Std.append(np.std(np.concatenate([data68['KT_post120'][data68['UnixUtcTime']==Tim].values,
                                             data69['KT_post120'][data69['UnixUtcTime']==Tim].values,
                                             data70['KT_post120'][data70['UnixUtcTime']==Tim].values,
                                             data71['KT_post120'][data71['UnixUtcTime']==Tim].values,
                                             data72['KT_post120'][data72['UnixUtcTime']==Tim].values,
                                             data73['KT_post120'][data73['UnixUtcTime']==Tim].values,
                                             data74['KT_post120'][data74['UnixUtcTime']==Tim].values])))
            SpatialPost180Std.append(np.std(np.concatenate([data68['KT_post180'][data68['UnixUtcTime']==Tim].values,
                                             data69['KT_post180'][data69['UnixUtcTime']==Tim].values,
                                             data70['KT_post180'][data70['UnixUtcTime']==Tim].values,
                                             data71['KT_post180'][data71['UnixUtcTime']==Tim].values,
                                             data72['KT_post180'][data72['UnixUtcTime']==Tim].values,
                                             data73['KT_post180'][data73['UnixUtcTime']==Tim].values,
                                             data74['KT_post180'][data74['UnixUtcTime']==Tim].values])))

        data67['NearbyKt_prev15'] = Nearby
        data67['NearbyKt_prev15std'] = SpatialPre15Std
        data67['NearbyKt_post15std'] = SpatialPost15Std
        data67['NearbyKt_post60std'] = SpatialPost60Std
        data67['NearbyKt_post120std'] = SpatialPost120Std
        data67['NearbyKt_post180std'] = SpatialPost180Std


    if t == 1:
        data = data68
        sLength = len(data)
        for x in xrange(0,sLength):
            Tim = data['UnixUtcTime'][x]
            Nearby.append(np.mean(np.concatenate([data67['KT_prev15'][data67['UnixUtcTime']==Tim].values,
                                   data69['KT_prev15'][data69['UnixUtcTime']==Tim].values,
                                   data70['KT_prev15'][data70['UnixUtcTime']==Tim].values,
                                   data71['KT_prev15'][data71['UnixUtcTime']==Tim].values,
                                   data72['KT_prev15'][data72['UnixUtcTime']==Tim].values,
                                   data73['KT_prev15'][data73['UnixUtcTime']==Tim].values,
                                   data74['KT_prev15'][data74['UnixUtcTime']==Tim].values])))
            SpatialPre15Std.append(np.std(np.concatenate([data67['KT_prev15'][data67['UnixUtcTime']==Tim].values,
                                           data69['KT_prev15'][data69['UnixUtcTime']==Tim].values,
                                           data70['KT_prev15'][data70['UnixUtcTime']==Tim].values,
                                           data71['KT_prev15'][data71['UnixUtcTime']==Tim].values,
                                           data72['KT_prev15'][data72['UnixUtcTime']==Tim].values,
                                           data73['KT_prev15'][data73['UnixUtcTime']==Tim].values,
                                           data74['KT_prev15'][data74['UnixUtcTime']==Tim].values])))
            SpatialPost15Std.append(np.std(np.concatenate([data67['KT_post15'][data67['UnixUtcTime']==Tim].values,
                                            data69['KT_post15'][data69['UnixUtcTime']==Tim].values,
                                            data70['KT_post15'][data70['UnixUtcTime']==Tim].values,
                                            data71['KT_post15'][data71['UnixUtcTime']==Tim].values,
                                            data72['KT_post15'][data72['UnixUtcTime']==Tim].values,
                                            data73['KT_post15'][data73['UnixUtcTime']==Tim].values,
                                            data74['KT_post15'][data74['UnixUtcTime']==Tim].values])))
            SpatialPost60Std.append(np.std(np.concatenate([data67['KT_post60'][data67['UnixUtcTime']==Tim].values,
                                            data69['KT_post60'][data69['UnixUtcTime']==Tim].values,
                                            data70['KT_post60'][data70['UnixUtcTime']==Tim].values,
                                            data71['KT_post60'][data71['UnixUtcTime']==Tim].values,
                                            data72['KT_post60'][data72['UnixUtcTime']==Tim].values,
                                            data73['KT_post60'][data73['UnixUtcTime']==Tim].values,
                                            data74['KT_post60'][data74['UnixUtcTime']==Tim].values])))
            SpatialPost120Std.append(np.std(np.concatenate([data67['KT_post120'][data67['UnixUtcTime']==Tim].values,
                                             data69['KT_post120'][data69['UnixUtcTime']==Tim].values,
                                             data70['KT_post120'][data70['UnixUtcTime']==Tim].values,
                                             data71['KT_post120'][data71['UnixUtcTime']==Tim].values,
                                             data72['KT_post120'][data72['UnixUtcTime']==Tim].values,
                                             data73['KT_post120'][data73['UnixUtcTime']==Tim].values,
                                             data74['KT_post120'][data74['UnixUtcTime']==Tim].values])))
            SpatialPost180Std.append(np.std(np.concatenate([data67['KT_post180'][data67['UnixUtcTime']==Tim].values,
                                             data69['KT_post180'][data69['UnixUtcTime']==Tim].values,
                                             data70['KT_post180'][data70['UnixUtcTime']==Tim].values,
                                             data71['KT_post180'][data71['UnixUtcTime']==Tim].values,
                                             data72['KT_post180'][data72['UnixUtcTime']==Tim].values,
                                             data73['KT_post180'][data73['UnixUtcTime']==Tim].values,
                                             data74['KT_post180'][data74['UnixUtcTime']==Tim].values])))

        data68['NearbyKt_prev15'] = Nearby
        data68['NearbyKt_prev15std'] = SpatialPre15Std
        data68['NearbyKt_post15std'] = SpatialPost15Std
        data68['NearbyKt_post60std'] = SpatialPost60Std
        data68['NearbyKt_post120std'] = SpatialPost120Std
        data68['NearbyKt_post180std'] = SpatialPost180Std

    if t == 2:
        data = data69
        sLength = len(data)
        for x in xrange(0,sLength):
            Tim = data['UnixUtcTime'][x]
            Nearby.append(np.mean(np.concatenate([data67['KT_prev15'][data67['UnixUtcTime']==Tim].values,
                                   data68['KT_prev15'][data68['UnixUtcTime']==Tim].values,
                                   data70['KT_prev15'][data70['UnixUtcTime']==Tim].values,
                                   data71['KT_prev15'][data71['UnixUtcTime']==Tim].values,
                                   data72['KT_prev15'][data72['UnixUtcTime']==Tim].values,
                                   data73['KT_prev15'][data73['UnixUtcTime']==Tim].values,
                                   data74['KT_prev15'][data74['UnixUtcTime']==Tim].values])))
            SpatialPre15Std.append(np.std(np.concatenate([data67['KT_prev15'][data67['UnixUtcTime']==Tim].values,
                                           data68['KT_prev15'][data68['UnixUtcTime']==Tim].values,
                                           data70['KT_prev15'][data70['UnixUtcTime']==Tim].values,
                                           data71['KT_prev15'][data71['UnixUtcTime']==Tim].values,
                                           data72['KT_prev15'][data72['UnixUtcTime']==Tim].values,
                                           data73['KT_prev15'][data73['UnixUtcTime']==Tim].values,
                                           data74['KT_prev15'][data74['UnixUtcTime']==Tim].values])))
            SpatialPost15Std.append(np.std(np.concatenate([data67['KT_post15'][data67['UnixUtcTime']==Tim].values,
                                            data68['KT_post15'][data68['UnixUtcTime']==Tim].values,
                                            data70['KT_post15'][data70['UnixUtcTime']==Tim].values,
                                            data71['KT_post15'][data71['UnixUtcTime']==Tim].values,
                                            data72['KT_post15'][data72['UnixUtcTime']==Tim].values,
                                            data73['KT_post15'][data73['UnixUtcTime']==Tim].values,
                                            data74['KT_post15'][data74['UnixUtcTime']==Tim].values])))
            SpatialPost60Std.append(np.std(np.concatenate([data67['KT_post60'][data67['UnixUtcTime']==Tim].values,
                                            data68['KT_post60'][data68['UnixUtcTime']==Tim].values,
                                            data70['KT_post60'][data70['UnixUtcTime']==Tim].values,
                                            data71['KT_post60'][data71['UnixUtcTime']==Tim].values,
                                            data72['KT_post60'][data72['UnixUtcTime']==Tim].values,
                                            data73['KT_post60'][data73['UnixUtcTime']==Tim].values,
                                            data74['KT_post60'][data74['UnixUtcTime']==Tim].values])))
            SpatialPost120Std.append(np.std(np.concatenate([data67['KT_post120'][data67['UnixUtcTime']==Tim].values,
                                             data68['KT_post120'][data68['UnixUtcTime']==Tim].values,
                                             data70['KT_post120'][data70['UnixUtcTime']==Tim].values,
                                             data71['KT_post120'][data71['UnixUtcTime']==Tim].values,
                                             data72['KT_post120'][data72['UnixUtcTime']==Tim].values,
                                             data73['KT_post120'][data73['UnixUtcTime']==Tim].values,
                                             data74['KT_post120'][data74['UnixUtcTime']==Tim].values])))
            SpatialPost180Std.append(np.std(np.concatenate([data67['KT_post180'][data67['UnixUtcTime']==Tim].values,
                                             data68['KT_post180'][data68['UnixUtcTime']==Tim].values,
                                             data70['KT_post180'][data70['UnixUtcTime']==Tim].values,
                                             data71['KT_post180'][data71['UnixUtcTime']==Tim].values,
                                             data72['KT_post180'][data72['UnixUtcTime']==Tim].values,
                                             data73['KT_post180'][data73['UnixUtcTime']==Tim].values,
                                             data74['KT_post180'][data74['UnixUtcTime']==Tim].values])))

        data69['NearbyKt_prev15'] = Nearby
        data69['NearbyKt_prev15std'] = SpatialPre15Std
        data69['NearbyKt_post15std'] = SpatialPost15Std
        data69['NearbyKt_post60std'] = SpatialPost60Std
        data69['NearbyKt_post120std'] = SpatialPost120Std
        data69['NearbyKt_post180std'] = SpatialPost180Std
    if t == 3:
        data = data70
        sLength = len(data)
        for x in xrange(0,sLength):
            Tim = data['UnixUtcTime'][x]
            Nearby.append(np.mean(np.concatenate([data67['KT_prev15'][data67['UnixUtcTime']==Tim].values,
                                   data68['KT_prev15'][data68['UnixUtcTime']==Tim].values,
                                   data69['KT_prev15'][data69['UnixUtcTime']==Tim].values,
                                   data71['KT_prev15'][data71['UnixUtcTime']==Tim].values,
                                   data72['KT_prev15'][data72['UnixUtcTime']==Tim].values,
                                   data73['KT_prev15'][data73['UnixUtcTime']==Tim].values,
                                   data74['KT_prev15'][data74['UnixUtcTime']==Tim].values])))
            SpatialPre15Std.append(np.std(np.concatenate([data67['KT_prev15'][data67['UnixUtcTime']==Tim].values,
                                           data68['KT_prev15'][data68['UnixUtcTime']==Tim].values,
                                           data69['KT_prev15'][data69['UnixUtcTime']==Tim].values,
                                           data71['KT_prev15'][data71['UnixUtcTime']==Tim].values,
                                           data72['KT_prev15'][data72['UnixUtcTime']==Tim].values,
                                           data73['KT_prev15'][data73['UnixUtcTime']==Tim].values,
                                           data74['KT_prev15'][data74['UnixUtcTime']==Tim].values])))
            SpatialPost15Std.append(np.std(np.concatenate([data67['KT_post15'][data67['UnixUtcTime']==Tim].values,
                                            data68['KT_post15'][data68['UnixUtcTime']==Tim].values,
                                            data69['KT_post15'][data69['UnixUtcTime']==Tim].values,
                                            data71['KT_post15'][data71['UnixUtcTime']==Tim].values,
                                            data72['KT_post15'][data72['UnixUtcTime']==Tim].values,
                                            data73['KT_post15'][data73['UnixUtcTime']==Tim].values,
                                            data74['KT_post15'][data74['UnixUtcTime']==Tim].values])))
            SpatialPost60Std.append(np.std(np.concatenate([data67['KT_post60'][data67['UnixUtcTime']==Tim].values,
                                            data68['KT_post60'][data68['UnixUtcTime']==Tim].values,
                                            data69['KT_post60'][data69['UnixUtcTime']==Tim].values,
                                            data71['KT_post60'][data71['UnixUtcTime']==Tim].values,
                                            data72['KT_post60'][data72['UnixUtcTime']==Tim].values,
                                            data73['KT_post60'][data73['UnixUtcTime']==Tim].values,
                                            data74['KT_post60'][data74['UnixUtcTime']==Tim].values])))
            SpatialPost120Std.append(np.std(np.concatenate([data67['KT_post120'][data67['UnixUtcTime']==Tim].values,
                                             data68['KT_post120'][data68['UnixUtcTime']==Tim].values,
                                             data69['KT_post120'][data69['UnixUtcTime']==Tim].values,
                                             data71['KT_post120'][data71['UnixUtcTime']==Tim].values,
                                             data72['KT_post120'][data72['UnixUtcTime']==Tim].values,
                                             data73['KT_post120'][data73['UnixUtcTime']==Tim].values,
                                             data74['KT_post120'][data74['UnixUtcTime']==Tim].values])))
            SpatialPost180Std.append(np.std(np.concatenate([data67['KT_post180'][data67['UnixUtcTime']==Tim].values,
                                             data68['KT_post180'][data68['UnixUtcTime']==Tim].values,
                                             data69['KT_post180'][data69['UnixUtcTime']==Tim].values,
                                             data71['KT_post180'][data71['UnixUtcTime']==Tim].values,
                                             data72['KT_post180'][data72['UnixUtcTime']==Tim].values,
                                             data73['KT_post180'][data73['UnixUtcTime']==Tim].values,
                                             data74['KT_post180'][data74['UnixUtcTime']==Tim].values])))

        data70['NearbyKt_prev15'] = Nearby
        data70['NearbyKt_prev15std'] = SpatialPre15Std
        data70['NearbyKt_post15std'] = SpatialPost15Std
        data70['NearbyKt_post60std'] = SpatialPost60Std
        data70['NearbyKt_post120std'] = SpatialPost120Std
        data70['NearbyKt_post180std'] = SpatialPost180Std

    if t == 4:
        data = data71
        sLength = len(data)
        for x in xrange(0,sLength):
            Tim = data['UnixUtcTime'][x]
            Nearby.append(np.mean(np.concatenate([data67['KT_prev15'][data67['UnixUtcTime']==Tim].values,
                                   data68['KT_prev15'][data68['UnixUtcTime']==Tim].values,
                                   data69['KT_prev15'][data69['UnixUtcTime']==Tim].values,
                                   data70['KT_prev15'][data70['UnixUtcTime']==Tim].values,
                                   data72['KT_prev15'][data72['UnixUtcTime']==Tim].values,
                                   data73['KT_prev15'][data73['UnixUtcTime']==Tim].values,
                                   data74['KT_prev15'][data74['UnixUtcTime']==Tim].values])))
            SpatialPre15Std.append(np.std(np.concatenate([data67['KT_prev15'][data67['UnixUtcTime']==Tim].values,
                                           data68['KT_prev15'][data68['UnixUtcTime']==Tim].values,
                                           data69['KT_prev15'][data69['UnixUtcTime']==Tim].values,
                                           data70['KT_prev15'][data70['UnixUtcTime']==Tim].values,
                                           data72['KT_prev15'][data72['UnixUtcTime']==Tim].values,
                                           data73['KT_prev15'][data73['UnixUtcTime']==Tim].values,
                                           data74['KT_prev15'][data74['UnixUtcTime']==Tim].values])))
            SpatialPost15Std.append(np.std(np.concatenate([data67['KT_post15'][data67['UnixUtcTime']==Tim].values,
                                            data68['KT_post15'][data68['UnixUtcTime']==Tim].values,
                                            data69['KT_post15'][data69['UnixUtcTime']==Tim].values,
                                            data70['KT_post15'][data70['UnixUtcTime']==Tim].values,
                                            data72['KT_post15'][data72['UnixUtcTime']==Tim].values,
                                            data73['KT_post15'][data73['UnixUtcTime']==Tim].values,
                                            data74['KT_post15'][data74['UnixUtcTime']==Tim].values])))
            SpatialPost60Std.append(np.std(np.concatenate([data67['KT_post60'][data67['UnixUtcTime']==Tim].values,
                                            data68['KT_post60'][data68['UnixUtcTime']==Tim].values,
                                            data69['KT_post60'][data69['UnixUtcTime']==Tim].values,
                                            data70['KT_post60'][data70['UnixUtcTime']==Tim].values,
                                            data72['KT_post60'][data72['UnixUtcTime']==Tim].values,
                                            data73['KT_post60'][data73['UnixUtcTime']==Tim].values,
                                            data74['KT_post60'][data74['UnixUtcTime']==Tim].values])))
            SpatialPost120Std.append(np.std(np.concatenate([data67['KT_post120'][data67['UnixUtcTime']==Tim].values,
                                             data68['KT_post120'][data68['UnixUtcTime']==Tim].values,
                                             data69['KT_post120'][data69['UnixUtcTime']==Tim].values,
                                             data70['KT_post120'][data70['UnixUtcTime']==Tim].values,
                                             data72['KT_post120'][data72['UnixUtcTime']==Tim].values,
                                             data73['KT_post120'][data73['UnixUtcTime']==Tim].values,
                                             data74['KT_post120'][data74['UnixUtcTime']==Tim].values])))
            SpatialPost180Std.append(np.std(np.concatenate([data67['KT_post180'][data67['UnixUtcTime']==Tim].values,
                                             data68['KT_post180'][data68['UnixUtcTime']==Tim].values,
                                             data69['KT_post180'][data69['UnixUtcTime']==Tim].values,
                                             data70['KT_post180'][data70['UnixUtcTime']==Tim].values,
                                             data72['KT_post180'][data72['UnixUtcTime']==Tim].values,
                                             data73['KT_post180'][data73['UnixUtcTime']==Tim].values,
                                             data74['KT_post180'][data74['UnixUtcTime']==Tim].values])))

        data71['NearbyKt_prev15'] = Nearby
        data71['NearbyKt_prev15std'] = SpatialPre15Std
        data71['NearbyKt_post15std'] = SpatialPost15Std
        data71['NearbyKt_post60std'] = SpatialPost60Std
        data71['NearbyKt_post120std'] = SpatialPost120Std
        data71['NearbyKt_post180std'] = SpatialPost180Std

    if t == 5:
        data = data72
        sLength = len(data)
        for x in xrange(0,sLength):
            Tim = data['UnixUtcTime'][x]
            Nearby.append(np.mean(np.concatenate([data67['KT_prev15'][data67['UnixUtcTime']==Tim].values,
                                   data68['KT_prev15'][data68['UnixUtcTime']==Tim].values,
                                   data69['KT_prev15'][data69['UnixUtcTime']==Tim].values,
                                   data70['KT_prev15'][data70['UnixUtcTime']==Tim].values,
                                   data71['KT_prev15'][data71['UnixUtcTime']==Tim].values,
                                   data73['KT_prev15'][data73['UnixUtcTime']==Tim].values,
                                   data74['KT_prev15'][data74['UnixUtcTime']==Tim].values])))
            SpatialPre15Std.append(np.std(np.concatenate([data67['KT_prev15'][data67['UnixUtcTime']==Tim].values,
                                           data68['KT_prev15'][data68['UnixUtcTime']==Tim].values,
                                           data69['KT_prev15'][data69['UnixUtcTime']==Tim].values,
                                           data70['KT_prev15'][data70['UnixUtcTime']==Tim].values,
                                           data71['KT_prev15'][data71['UnixUtcTime']==Tim].values,
                                           data73['KT_prev15'][data73['UnixUtcTime']==Tim].values,
                                           data74['KT_prev15'][data74['UnixUtcTime']==Tim].values])))
            SpatialPost15Std.append(np.std(np.concatenate([data67['KT_post15'][data67['UnixUtcTime']==Tim].values,
                                            data68['KT_post15'][data68['UnixUtcTime']==Tim].values,
                                            data69['KT_post15'][data69['UnixUtcTime']==Tim].values,
                                            data70['KT_post15'][data70['UnixUtcTime']==Tim].values,
                                            data71['KT_post15'][data71['UnixUtcTime']==Tim].values,
                                            data73['KT_post15'][data73['UnixUtcTime']==Tim].values,
                                            data74['KT_post15'][data74['UnixUtcTime']==Tim].values])))
            SpatialPost60Std.append(np.std(np.concatenate([data67['KT_post60'][data67['UnixUtcTime']==Tim].values,
                                            data68['KT_post60'][data68['UnixUtcTime']==Tim].values,
                                            data69['KT_post60'][data69['UnixUtcTime']==Tim].values,
                                            data70['KT_post60'][data70['UnixUtcTime']==Tim].values,
                                            data71['KT_post60'][data71['UnixUtcTime']==Tim].values,
                                            data73['KT_post60'][data73['UnixUtcTime']==Tim].values,
                                            data74['KT_post60'][data74['UnixUtcTime']==Tim].values])))
            SpatialPost120Std.append(np.std(np.concatenate([data67['KT_post120'][data67['UnixUtcTime']==Tim].values,
                                             data68['KT_post120'][data68['UnixUtcTime']==Tim].values,
                                             data69['KT_post120'][data69['UnixUtcTime']==Tim].values,
                                             data70['KT_post120'][data70['UnixUtcTime']==Tim].values,
                                             data71['KT_post120'][data71['UnixUtcTime']==Tim].values,
                                             data73['KT_post120'][data73['UnixUtcTime']==Tim].values,
                                             data74['KT_post120'][data74['UnixUtcTime']==Tim].values])))
            SpatialPost180Std.append(np.std(np.concatenate([data67['KT_post180'][data67['UnixUtcTime']==Tim].values,
                                             data68['KT_post180'][data68['UnixUtcTime']==Tim].values,
                                             data69['KT_post180'][data69['UnixUtcTime']==Tim].values,
                                             data70['KT_post180'][data70['UnixUtcTime']==Tim].values,
                                             data71['KT_post180'][data71['UnixUtcTime']==Tim].values,
                                             data73['KT_post180'][data73['UnixUtcTime']==Tim].values,
                                             data74['KT_post180'][data74['UnixUtcTime']==Tim].values])))

        data72['NearbyKt_prev15'] = Nearby
        data72['NearbyKt_prev15std'] = SpatialPre15Std
        data72['NearbyKt_post15std'] = SpatialPost15Std
        data72['NearbyKt_post60std'] = SpatialPost60Std
        data72['NearbyKt_post120std'] = SpatialPost120Std
        data72['NearbyKt_post180std'] = SpatialPost180Std

    if t == 6:
        data = data73
        sLength = len(data)
        for x in xrange(0,sLength):
            Tim = data['UnixUtcTime'][x]
            Nearby.append(np.mean(np.concatenate([data67['KT_prev15'][data67['UnixUtcTime']==Tim].values,
                                   data68['KT_prev15'][data68['UnixUtcTime']==Tim].values,
                                   data69['KT_prev15'][data69['UnixUtcTime']==Tim].values,
                                   data70['KT_prev15'][data70['UnixUtcTime']==Tim].values,
                                   data71['KT_prev15'][data71['UnixUtcTime']==Tim].values,
                                   data72['KT_prev15'][data72['UnixUtcTime']==Tim].values,
                                   data74['KT_prev15'][data74['UnixUtcTime']==Tim].values])))
            SpatialPre15Std.append(np.std(np.concatenate([data67['KT_prev15'][data67['UnixUtcTime']==Tim].values,
                                           data68['KT_prev15'][data68['UnixUtcTime']==Tim].values,
                                           data69['KT_prev15'][data69['UnixUtcTime']==Tim].values,
                                           data70['KT_prev15'][data70['UnixUtcTime']==Tim].values,
                                           data71['KT_prev15'][data71['UnixUtcTime']==Tim].values,
                                           data72['KT_prev15'][data72['UnixUtcTime']==Tim].values,
                                           data74['KT_prev15'][data74['UnixUtcTime']==Tim].values])))
            SpatialPost15Std.append(np.std(np.concatenate([data67['KT_post15'][data67['UnixUtcTime']==Tim].values,
                                            data68['KT_post15'][data68['UnixUtcTime']==Tim].values,
                                            data69['KT_post15'][data69['UnixUtcTime']==Tim].values,
                                            data70['KT_post15'][data70['UnixUtcTime']==Tim].values,
                                            data71['KT_post15'][data71['UnixUtcTime']==Tim].values,
                                            data72['KT_post15'][data72['UnixUtcTime']==Tim].values,
                                            data74['KT_post15'][data74['UnixUtcTime']==Tim].values])))
            SpatialPost60Std.append(np.std(np.concatenate([data67['KT_post60'][data67['UnixUtcTime']==Tim].values,
                                            data68['KT_post60'][data68['UnixUtcTime']==Tim].values,
                                            data69['KT_post60'][data69['UnixUtcTime']==Tim].values,
                                            data70['KT_post60'][data70['UnixUtcTime']==Tim].values,
                                            data71['KT_post60'][data71['UnixUtcTime']==Tim].values,
                                            data72['KT_post60'][data72['UnixUtcTime']==Tim].values,
                                            data74['KT_post60'][data74['UnixUtcTime']==Tim].values])))
            SpatialPost120Std.append(np.std(np.concatenate([data67['KT_post120'][data67['UnixUtcTime']==Tim].values,
                                             data68['KT_post120'][data68['UnixUtcTime']==Tim].values,
                                             data69['KT_post120'][data69['UnixUtcTime']==Tim].values,
                                             data70['KT_post120'][data70['UnixUtcTime']==Tim].values,
                                             data71['KT_post120'][data71['UnixUtcTime']==Tim].values,
                                             data72['KT_post120'][data72['UnixUtcTime']==Tim].values,
                                             data74['KT_post120'][data74['UnixUtcTime']==Tim].values])))
            SpatialPost180Std.append(np.std(np.concatenate([data67['KT_post180'][data67['UnixUtcTime']==Tim].values,
                                             data68['KT_post180'][data68['UnixUtcTime']==Tim].values,
                                             data69['KT_post180'][data69['UnixUtcTime']==Tim].values,
                                             data70['KT_post180'][data70['UnixUtcTime']==Tim].values,
                                             data71['KT_post180'][data71['UnixUtcTime']==Tim].values,
                                             data72['KT_post180'][data72['UnixUtcTime']==Tim].values,
                                             data74['KT_post180'][data74['UnixUtcTime']==Tim].values])))

        data73['NearbyKt_prev15'] = Nearby
        data73['NearbyKt_prev15std'] = SpatialPre15Std
        data73['NearbyKt_post15std'] = SpatialPost15Std
        data73['NearbyKt_post60std'] = SpatialPost60Std
        data73['NearbyKt_post120std'] = SpatialPost120Std
        data73['NearbyKt_post180std'] = SpatialPost180Std

    if t == 7:
        data = data74
        sLength = len(data)
        for x in xrange(0,sLength):
            Tim = data['UnixUtcTime'][x]
            Nearby.append(np.mean(np.concatenate([data67['KT_prev15'][data67['UnixUtcTime']==Tim].values,
                                   data68['KT_prev15'][data68['UnixUtcTime']==Tim].values,
                                   data69['KT_prev15'][data69['UnixUtcTime']==Tim].values,
                                   data70['KT_prev15'][data70['UnixUtcTime']==Tim].values,
                                   data71['KT_prev15'][data71['UnixUtcTime']==Tim].values,
                                   data72['KT_prev15'][data72['UnixUtcTime']==Tim].values,
                                   data73['KT_prev15'][data73['UnixUtcTime']==Tim].values])))
            SpatialPre15Std.append(np.std(np.concatenate([data67['KT_prev15'][data67['UnixUtcTime']==Tim].values,
                                           data68['KT_prev15'][data68['UnixUtcTime']==Tim].values,
                                           data69['KT_prev15'][data69['UnixUtcTime']==Tim].values,
                                           data70['KT_prev15'][data70['UnixUtcTime']==Tim].values,
                                           data71['KT_prev15'][data71['UnixUtcTime']==Tim].values,
                                           data72['KT_prev15'][data72['UnixUtcTime']==Tim].values,
                                           data73['KT_prev15'][data73['UnixUtcTime']==Tim].values])))
            SpatialPost15Std.append(np.std(np.concatenate([data67['KT_post15'][data67['UnixUtcTime']==Tim].values,
                                            data68['KT_post15'][data68['UnixUtcTime']==Tim].values,
                                            data69['KT_post15'][data69['UnixUtcTime']==Tim].values,
                                            data70['KT_post15'][data70['UnixUtcTime']==Tim].values,
                                            data71['KT_post15'][data71['UnixUtcTime']==Tim].values,
                                            data72['KT_post15'][data72['UnixUtcTime']==Tim].values,
                                            data73['KT_post15'][data73['UnixUtcTime']==Tim].values])))
            SpatialPost60Std.append(np.std(np.concatenate([data67['KT_post60'][data67['UnixUtcTime']==Tim].values,
                                            data68['KT_post60'][data68['UnixUtcTime']==Tim].values,
                                            data69['KT_post60'][data69['UnixUtcTime']==Tim].values,
                                            data70['KT_post60'][data70['UnixUtcTime']==Tim].values,
                                            data71['KT_post60'][data71['UnixUtcTime']==Tim].values,
                                            data72['KT_post60'][data72['UnixUtcTime']==Tim].values,
                                            data73['KT_post60'][data73['UnixUtcTime']==Tim].values])))
            SpatialPost120Std.append(np.std(np.concatenate([data67['KT_post120'][data67['UnixUtcTime']==Tim].values,
                                             data68['KT_post120'][data68['UnixUtcTime']==Tim].values,
                                             data69['KT_post120'][data69['UnixUtcTime']==Tim].values,
                                             data70['KT_post120'][data70['UnixUtcTime']==Tim].values,
                                             data71['KT_post120'][data71['UnixUtcTime']==Tim].values,
                                             data72['KT_post120'][data72['UnixUtcTime']==Tim].values,
                                             data73['KT_post120'][data73['UnixUtcTime']==Tim].values])))
            SpatialPost180Std.append(np.std(np.concatenate([data67['KT_post180'][data67['UnixUtcTime']==Tim].values,
                                             data68['KT_post180'][data68['UnixUtcTime']==Tim].values,
                                             data69['KT_post180'][data69['UnixUtcTime']==Tim].values,
                                             data70['KT_post180'][data70['UnixUtcTime']==Tim].values,
                                             data71['KT_post180'][data71['UnixUtcTime']==Tim].values,
                                             data72['KT_post180'][data72['UnixUtcTime']==Tim].values,
                                             data73['KT_post180'][data73['UnixUtcTime']==Tim].values])))

        data74['NearbyKt_prev15'] = Nearby
        data74['NearbyKt_prev15std'] = SpatialPre15Std
        data74['NearbyKt_post15std'] = SpatialPost15Std
        data74['NearbyKt_post60std'] = SpatialPost60Std
        data74['NearbyKt_post120std'] = SpatialPost120Std
        data74['NearbyKt_post180std'] = SpatialPost180Std

## Convert UTC Hour to Local Hour
data67['LocalHour'] = data67['UtcHour']
data68['LocalHour'] = data68['UtcHour']
data69['LocalHour'] = data69['UtcHour']
data70['LocalHour'] = data70['UtcHour']
data71['LocalHour'] = data71['UtcHour']
data72['LocalHour'] = data72['UtcHour']
data73['LocalHour'] = data73['UtcHour']
data74['LocalHour'] = data74['UtcHour']

cond = data67.UtcHour < 8
data67.LocalHour[cond] = data67['UtcHour'][cond]+16
cond2 = data67.UtcHour > 7
data67.LocalHour[cond2] = data67['UtcHour'][cond2]-8
cond = data68.UtcHour < 8
data68.LocalHour[cond] = data68['UtcHour'][cond]+16
cond2 = data68.UtcHour > 7
data68.LocalHour[cond2] = data68['UtcHour'][cond2]-8
cond = data69.UtcHour < 8
data69.LocalHour[cond] = data69['UtcHour'][cond]+16
cond2 = data69.UtcHour > 7
data69.LocalHour[cond2] = data69['UtcHour'][cond2]-8
cond = data70.UtcHour < 8
data70.LocalHour[cond] = data70['UtcHour'][cond]+16
cond2 = data70.UtcHour > 7
data70.LocalHour[cond2] = data70['UtcHour'][cond2]-8
cond = data71.UtcHour < 8
data71.LocalHour[cond] = data71['UtcHour'][cond]+16
cond2 = data71.UtcHour > 7
data71.LocalHour[cond2] = data71['UtcHour'][cond2]-8
cond = data72.UtcHour < 8
data72.LocalHour[cond] = data72['UtcHour'][cond]+16
cond2 = data72.UtcHour > 7
data72.LocalHour[cond2] = data72['UtcHour'][cond2]-8
cond = data73.UtcHour < 8
data73.LocalHour[cond] = data73['UtcHour'][cond]+16
cond2 = data73.UtcHour > 7
data73.LocalHour[cond2] = data73['UtcHour'][cond2]-8
cond = data74.UtcHour < 8
data74.LocalHour[cond] = data74['UtcHour'][cond]+16
cond2 = data74.UtcHour > 7
data74.LocalHour[cond2] = data74['UtcHour'][cond2]-8

data67['SineJulianDay'] = np.sin((data67['JulianDay'].values)*np.pi/180.)
data68['SineJulianDay'] = np.sin((data68['JulianDay'].values)*np.pi/180.)
data69['SineJulianDay'] = np.sin((data69['JulianDay'].values)*np.pi/180.)
data70['SineJulianDay'] = np.sin((data70['JulianDay'].values)*np.pi/180.)
data71['SineJulianDay'] = np.sin((data71['JulianDay'].values)*np.pi/180.)
data72['SineJulianDay'] = np.sin((data72['JulianDay'].values)*np.pi/180.)
data73['SineJulianDay'] = np.sin((data73['JulianDay'].values)*np.pi/180.)
data74['SineJulianDay'] = np.sin((data74['JulianDay'].values)*np.pi/180.)
data67['CosJulianDay'] = np.cos((data67['JulianDay'].values)*np.pi/180.)
data68['CosJulianDay'] = np.cos((data68['JulianDay'].values)*np.pi/180.)
data69['CosJulianDay'] = np.cos((data69['JulianDay'].values)*np.pi/180.)
data70['CosJulianDay'] = np.cos((data70['JulianDay'].values)*np.pi/180.)
data71['CosJulianDay'] = np.cos((data71['JulianDay'].values)*np.pi/180.)
data72['CosJulianDay'] = np.cos((data72['JulianDay'].values)*np.pi/180.)
data73['CosJulianDay'] = np.cos((data73['JulianDay'].values)*np.pi/180.)
data74['CosJulianDay'] = np.cos((data74['JulianDay'].values)*np.pi/180.)
print data74['CosJulianDay']
## Slope multipled by R^2  ##
data67['Slope_R2'] = data67.KT_Slope*data67.R2
data68['Slope_R2'] = data68.KT_Slope*data68.R2
data69['Slope_R2'] = data69.KT_Slope*data69.R2
data70['Slope_R2'] = data70.KT_Slope*data70.R2
data71['Slope_R2'] = data71.KT_Slope*data71.R2
data72['Slope_R2'] = data72.KT_Slope*data72.R2
data73['Slope_R2'] = data73.KT_Slope*data73.R2
data74['Slope_R2'] = data74.KT_Slope*data74.R2

## Last change in KT... ##
data67['KT15_KT30'] = data67['KT_prev15']-data67['KT_prev30']
data68['KT15_KT30'] = data68['KT_prev15']-data68['KT_prev30']
data69['KT15_KT30'] = data69['KT_prev15']-data69['KT_prev30']
data70['KT15_KT30'] = data70['KT_prev15']-data70['KT_prev30']
data71['KT15_KT30'] = data71['KT_prev15']-data71['KT_prev30']
data72['KT15_KT30'] = data72['KT_prev15']-data72['KT_prev30']
data73['KT15_KT30'] = data73['KT_prev15']-data73['KT_prev30']
data74['KT15_KT30'] = data74['KT_prev15']-data74['KT_prev30']

data67 = data67.dropna().reset_index(drop=True)
data68 = data68.dropna().reset_index(drop=True)
data69 = data69.dropna().reset_index(drop=True)
data70 = data70.dropna().reset_index(drop=True)
data71 = data71.dropna().reset_index(drop=True)
data72 = data72.dropna().reset_index(drop=True)
data73 = data73.dropna().reset_index(drop=True)
data74 = data74.dropna().reset_index(drop=True)

#plt.figure()
#data67['KT_post15'].hist(bins=100); plt.title('KT 15-Min Ahead');
#savefig('SMUD67_KT15Hist.tif')
#plt.figure()
#data67['KT_post60'].hist(bins=100); plt.title('KT 60-Min Ahead');
#savefig('SMUD67_KT60Hist.tif')
#plt.figure()
#data67['KT_post120'].hist(bins=100); plt.title('KT 120-Min Ahead');
#savefig('SMUD67_KT120Hist.tif')
#plt.figure()
#data67['KT_post180'].hist(bins=100); plt.title('KT 180-Min Ahead');
#savefig('SMUD67_KT180Hist.tif')
#plt.figure()
#data67['KT15_KT30'].hist(bins=100); plt.title('Most Recent Change in 15-min KT');
#savefig('SMUD67_KT15_KT30_Hist.tif')
#plt.figure()
#data67['KT_Slope'].hist(bins=100); plt.title('Linear Regression Equation: Slope of KT');
#savefig('SMUD67_KT_Slope_Hist.tif')
#plt.figure()
#data67['T_Td'].hist(bins=100); plt.title('Dewpoint Depression');
#savefig('SMUD67_T_TD_Hist.tif')
#plt.figure()
#data67['R2'].hist(bins=100); plt.title('Correlation Coefficient');
#savefig('SMUD67_R2_Hist.tif')
#plt.figure()
#data67['T_AVG'].hist(bins=100); plt.title('Temperature');
#savefig('SMUD67_T_AVG_Hist.tif')
#plt.figure()
#data67['dewpt_AVG'].hist(bins=100); plt.title('Dewpoint Temperature');
#savefig('SMUD67_Td_Hist.tif')
#plt.figure()
#data67['wind_speed_AVG'].hist(bins=100); plt.title('Wind Speed');
#savefig('SMUD67_wind_speed_Hist.tif')
#plt.figure()
#data67['prob_precip01_AVG'].hist(bins=100); plt.title('Probability of Precipitation (1-hr)');
#savefig('SMUD67_ProbPrecip_Hist.tif')
#plt.figure()
#data67['qpf01_AVG'].hist(bins=100); plt.title('QPF (1-hr)');
#savefig('SMUD67_QPF_Hist.tif')
#plt.figure()
#data67['cloud_cov_AVG'].hist(bins=100); plt.title('Cloud Cover');
#savefig('SMUD67_Cloud_Cover_Hist.tif')
#plt.figure()
#data67['cloud_cov_STD'].hist(bins=100); plt.title('Cloud Cover Temporal Variability (Standard Deviation Previous Hour)');
#savefig('SMUD67_cloud_cover_std_Hist.tif')
#plt.figure()
#data67['cloud_cov_SQRD'].hist(bins=100); plt.title('Cloud Cover Squared');
#savefig('SMUD67_CloudCoverSquared_Hist.tif')
#data67['std_prev1Hr'].hist(bins=100); plt.title('Clearness Index Temporal Variability (Standard Deviation Previous Hour)');
#savefig('SMUD67_KT_Stdev_PrevHr_Hist.tif')
#plt.figure()
#data67['NearbyKt_prev15'].hist(bins=100); plt.title('15-Min Average Clearness Index Across Nearby Sites');
#savefig('SMUD67_KTPrev15NearbyAVG_Hist.tif')
#plt.figure()
#data67['Slope_R2'].hist(bins=100); plt.title('Slope Multiplied by Correlation Coefficient');
#savefig('SMUD67_Slope_R2_Hist.tif')
#plt.figure()
#data67['NearbyKt_prev15std'].hist(bins=100); plt.title('Clearness Index Spatial Variability (Previous 15-Min)');
#savefig('SMUD67_Nearby15minVariability_Hist.tif')


SM67One, SM67Two, SM67Three = splitData(data67, trainPerc=0.34, cvPerc=0.33, testPerc=0.33)
SM68One, SM68Two, SM68Three = splitData(data68, trainPerc=0.34, cvPerc=0.33, testPerc=0.33)
SM69One, SM69Two, SM69Three = splitData(data69, trainPerc=0.34, cvPerc=0.33, testPerc=0.33)
SM70One, SM70Two, SM70Three = splitData(data70, trainPerc=0.34, cvPerc=0.33, testPerc=0.33)
SM71One, SM71Two, SM71Three = splitData(data71, trainPerc=0.34, cvPerc=0.33, testPerc=0.33)
SM72One, SM72Two, SM72Three = splitData(data72, trainPerc=0.34, cvPerc=0.33, testPerc=0.33)
SM73One, SM73Two, SM73Three = splitData(data73, trainPerc=0.34, cvPerc=0.33, testPerc=0.33)
SM74One, SM74Two, SM74Three = splitData(data74, trainPerc=0.34, cvPerc=0.33, testPerc=0.33)

data67.to_csv('SM67.csv', header = False)
data68.to_csv('SM68.csv', header = False)
data69.to_csv('SM69.csv', header = False)
data70.to_csv('SM70.csv', header = False)
data71.to_csv('SM71.csv', header = False)
data72.to_csv('SM72.csv', header = False)
data73.to_csv('SM73.csv', header = False)
data74.to_csv('SM74.csv', header = False)

# First 1/3 split...
SM67One.to_csv('SM671.csv', header = False)
SM68One.to_csv('SM681.csv', header = False)
SM69One.to_csv('SM691.csv', header = False)
SM70One.to_csv('SM701.csv', header = False)
SM71One.to_csv('SM711.csv', header = False)
SM72One.to_csv('SM721.csv', header = False)
SM73One.to_csv('SM731.csv', header = False)
SM74One.to_csv('SM741.csv', header = False)
# Second 1/3 split
SM67Two.to_csv('SM672.csv', header = False)
SM68Two.to_csv('SM682.csv', header = False)
SM69Two.to_csv('SM692.csv', header = False)
SM70Two.to_csv('SM702.csv', header = False)
SM71Two.to_csv('SM712.csv', header = False)
SM72Two.to_csv('SM722.csv', header = False)
SM73Two.to_csv('SM732.csv', header = False)
SM74Two.to_csv('SM742.csv', header = False)
# Third 1/3 split
SM67Three.to_csv('SM673.csv', header = False)
SM68Three.to_csv('SM683.csv', header = False)
SM69Three.to_csv('SM693.csv', header = False)
SM70Three.to_csv('SM703.csv', header = False)
SM71Three.to_csv('SM713.csv', header = False)
SM72Three.to_csv('SM723.csv', header = False)
SM73Three.to_csv('SM733.csv', header = False)
SM74Three.to_csv('SM743.csv', header = False)

## header = ["UnixUtcTime", "UtcYear", "UtcMonth", "UtcDay", "UtcHour", "JulianDay", "MV72483000_T", "MV72483009_T", "MV72483016_T", "MV72483000_dewpt", "MV72483009_dewpt", "MV72483016_dewpt", "MV72483000_cloud_cov", "MV72483009_cloud_cov", "MV72483016_cloud_cov", "MV72483000_prob_precip01", "MV72483009_prob_precip01", "MV72483016_prob_precip01", "MV72483000_qpf01", "MV72483009_qpf01", "MV72483016_qpf01", "MV72483000_wind_speed", "MV72483009_wind_speed", "MV72483016_wind_speed", "GHI_prev75", "GHI_prev60", "GHI_prev45", "GHI_prev30", "GHI_prev15", "GHI_post15", "GHI_post30", "GHI_post45", "GHI_post60", "GHI_post75", "GHI_post90", "GHI_post105", "GHI_post120", "GHI_post135", "GHI_post150", "GHI_post165", "GHI_post180", "T_AVG", "dewpt_AVG", "wind_speed_AVG", "prob_precip01_AVG", "qpf01_AVG", "cloud_cov_AVG", "T_Td", "cloud_cov_STD", "cloud_cov_SQRD", "TOA_prev75", "KT_prev75", "TOA_prev60", "KT_prev60", "TOA_prev45", "KT_prev45", "TOA_prev30", "KT_prev30", "TOA_prev15", "KT_prev15", "TOA_post15", "KT_post15", "TOA_post30", "KT_post30", "TOA_post45", "KT_post45", "TOA_post60", "KT_post60", "TOA_post75", "KT_post75", "TOA_post90", "KT_post90", "TOA_post105", "KT_post105", "TOA_post120", "KT_post120", "TOA_post135", "KT_post135", "TOA_post150", "KT_post150", "TOA_post165", "KT_post165", "TOA_post180", "KT_post180", "KT_Slope", "R2", "std_prev1Hr", "std_post1Hr", "std_post2Hr", "std_post3Hr", "NearbyKt_prev15", "NearbyKt_post15std", "NearbyKt_post60std", "NearbyKt_post120std", "NearbyKt_post180std","LocalHour", "SineJulianDay", "Slope_R2", "KT15_KT30"]
## data67.to_csv('data67out.csv', cols = header)
## data68.to_csv('data67out.csv', cols = header)
## data69.to_csv('data68out.csv', cols = header)
## data70.to_csv('data69out.csv', cols = header)

## data71.to_csv('data70out.csv', cols = header)

## data72.to_csv('data71out.csv', cols = header)

## data72.to_csv('data72out.csv', cols = header)

## data73.to_csv('data73out.csv', cols = header)
## data74.to_csv('data74out.csv', cols = header)
