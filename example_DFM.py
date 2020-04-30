#-------------------------------------------------Libraries
import os
from datetime import datetime as dt
from Functions.load_spec import load_spec
from Functions.load_data import load_data
from Functions.dfm import dfm
import pickle


#-------------------------------------------------User Inputs
vintage      = '2016-06-29'                                                   # vintage dataset to use for estimation
country      = 'US'                                                           # United States macroeconomic data
sample_start = dt.strptime("2000-01-01", '%Y-%m-%d').date().toordinal() + 366 # estimation sample


#-------------------------------------------------Load model specification and dataset.
# Load model specification structure `Spec`
Spec = load_spec('Spec_US_example.xls')

# Parse `Spec`
SeriesID         = Spec.SeriesID
SeriesName       = Spec.SeriesName
Units            = Spec.Units
UnitsTransformed = Spec.UnitsTransformed

# Load data
datafile   = os.path.join('data',country,vintage + '.xls')
X,Time,Z   = load_data(datafile,Spec,sample_start)


#-------------------------------------------------Run dynamic factor model (DFM) and save estimation output as 'ResDFM'.
threshold = 1e-4 # Set to 1e-5 for more robust estimates

Res = dfm(X,Spec,threshold)
Res = {"Res": Res,"Spec":Spec}

with open('ResDFM.pickle', 'wb') as handle:
    pickle.dump(Res, handle)