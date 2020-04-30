#-------------------------------------------------Libraries
import pandas as pd
import numpy as np
import os

#-------------------------------------------------Read data functions
#loadData Load vintage of data from file and format as structure
#
# Description:
#
#   Load data from file
#
# Input Arguments:
#
#   datafile - filename of Microsoft Excel workbook file
#
# Output Arguments:
#
# Data - structure with the following fields:
#   .    X : T x N numeric array, transformed dataset
#   . Time : T x 1 numeric array, date number with observation dates
#   .    Z : T x N numeric array, raw (untransformed) dataset

def load_data(datafile,Spec,sample = None,loadExcel = 0):
    if not os.path.splitext(datafile)[1] in [".xlsx",".xls"]:
        ValueError("MUST BE EXCEL FILE")

    Z,Time,Mnem = readData(datafile)

    # Sort data based on model specification
    Z,Mnem = sortData(Z.copy(),Mnem.copy(),Spec)
    del Mnem # since now Mnem == Spec.SeriesID

    # Transform data based on model specification
    X,Time,Z = transformData(Z.copy(),Time.copy(),Spec)

    # Drop data not in estimation sample
    if sample != None:
        X,Time,Z = dropData(X.copy(),Time.copy(),Z.copy(),sample)

    return X,Time,Z

# readData Read data from Microsoft Excel workbook file
def readData(datafile):
    dat  = pd.read_excel(datafile)
    Mnem = np.array([i for i in list(dat.columns) if i != "Date"])
    Z    = dat[Mnem].to_numpy(copy=True)
    Time = dat.Date.apply(lambda x: x.toordinal()+366).to_numpy(copy= True)

    return Z,Time,Mnem

# sortData Sort series by order of model specification
def sortData(Z,Mnem,Spec):

    # Drop series not in Spec
    inSpec = np.in1d(Mnem,Spec.SeriesID)
    Mnem   = Mnem[inSpec]
    Z      = Z[:,inSpec]

    # Sort series by ordering of Spec
    permutation = np.array([np.where(Mnem == i)[0][0] for i in Spec.SeriesID])
    Mnem        = Mnem[permutation]
    Z           = Z[:,permutation]

    return Z, Mnem

def transformData(Z,Time,Spec):
    # transformData Transforms each data series based on Spec.Transformation
    #
    # Input Arguments:
    #
    #      Z : T x N numeric array, raw (untransformed) observed data
    #   Spec : structure          , model specification
    #
    # Output Arguments:
    #
    #      X : T x N numeric array, transformed data (stationary to enter DFM)

    T,N          = Z.shape
    X            = np.empty((T, N))
    X[:]         = np.nan
    Freq_dict    = {"m":1,"q":3}
    formula_dict = {"lin":lambda x:x*2,
                    "chg":lambda x:np.append(np.nan,x[t1+step::step] - x[t1:-1-t1:step]),
                    "ch1":lambda x:x[12+t1::step] - x[t1:-12:step],
                    "pch":lambda x:(np.append(np.nan,x[t1+step::step]/x[t1:-1-t1:step]) - 1)*100,
                    "pc1":lambda x:((x[12+t1::step]/x[t1:-12:step])-1)*100,
                    "pca":lambda x:(np.append(np.nan,x[t1+step::step]/x[t1:-step:step])**(1/n) - 1)*100,
                    "log":lambda x:np.log(x)
    }

    for i in range(N):
        formula = Spec.Transformation[i]
        freq    = Spec.Frequency[i]
        step    = Freq_dict[freq] # time step for different frequencies based on monthly time
        t1      = step -1         # assume monthly observations start at beginning of quarter (subtracted 1 for indexing)
        n       = step/12         # number of years, needed to compute annual % changes
        series  = Spec.SeriesName[i]

        if formula == 'lin':
            X[:,i] = Z[:,i].copy()
        elif formula == 'chg':
            X[t1::step,i] = formula_dict['chg'](Z[:,i].copy())
        elif formula == 'ch1':
            X[12+t1::step, i] = formula_dict['ch1'](Z[:, i].copy())
        elif formula == 'pch':
            X[t1::step, i] = formula_dict['pch'](Z[:, i].copy())
        elif formula == 'pc1':
            X[12+t1::step, i] = formula_dict['pc1'](Z[:, i].copy())
        elif formula == 'pca':
            X[t1::step, i] = formula_dict['pca'](Z[:, i].copy())
        elif formula == 'log':
            X[:, i] = formula_dict['log'](Z[:, i].copy())
        else:
            ValueError("{}: Transformation is unknown".format(formula))

    # Drop first quarter of observations
    # since transformations cause missing values
    return X[3:,:],Time[3:],Z[3:,:]

# dropData Remove data not in estimation sample
def dropData(X,Time,Z,sample):
    filter_index = Time >= sample
    X            = X[filter_index,:].copy()
    Time         = Time[filter_index].copy()
    Z            = Z[filter_index, :].copy()

    return X,Time,Z