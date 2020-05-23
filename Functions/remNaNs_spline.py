#-------------------------------------------------Libraries
import numpy as np
from scipy.signal import lfilter
from scipy.interpolate import CubicSpline


#-------------------------------------------------remNaNs_spline
def remNaNs_spline(X,options):
    ###  Replication files for:
    ###  ""Nowcasting", 2010, (by Marta Banbura, Domenico Giannone and Lucrezia Reichlin),
    ### in Michael P. Clements and David F. Hendry, editors, Oxford Handbook on Economic Forecasting.
    ###
    ### The software can be freely used in applications.
    ### Users are kindly requested to add acknowledgements to published work and
    ### to cite the above reference in any resulting publications
    #
    #Description:
    #
    #remNaNs    Treats NaNs in dataset for use in DFM.
    #
    #  Syntax:
    #    [X,indNaN] = remNaNs(X,options)
    #
    #  Description:
    #    remNaNs() processes NaNs in a data matrix X according to 5 cases (see
    #    below for details). These are useful for running functions in the
    #    'DFM.m' file that do not take missing value inputs.
    #
    #  Input parameters:
    #    X (T x n): Input data where T gives time and n gives the series.
    #    options: A structure with two elements:
    #      options.method (numeric):
    #      - 1: Replaces all missing values using filter().
    #      - 2: Replaces missing values after removing trailing and leading
    #           zeros (a row is 'missing' if >80# is NaN)
    #      - 3: Only removes rows with leading and closing zeros
    #      - 4: Replaces missing values after removing trailing and leading
    #           zeros (a row is 'missing' if all are NaN)
    #      - 5: Replaces missing values with spline() then runs filter().
    #
    #      options.k (numeric): remNaNs() relies on MATLAB's filter function
    #      for the 1-D filter. k controls the rational transfer function
    #      argument's numerator (the denominator is set to 1). More
    #      specifically, the numerator takes the form 'ones(2*k+1,1)/(2*k+1)'
    #      For additional help, see MATLAB's documentation for filter().
    #
    #  Output parameters:
    #    X: Outputted data.
    #    indNaN: A matrix indicating the location for missing values (1 for NaN).

    T,N    = X.shape
    k      = options["k"]
    indNaN = np.isnan(X)

    if options["method"] == 1:  # replace all the missing values
        for i in range(N): # Loop through columns
            x              = X[:,i].copy()
            x[indNaN[:,i]] = np.nanmedian(x)
            x_MA           = lfilter(np.ones((2*k+1))/(2*k+1),1,np.append(np.append(x[0]*np.ones((k,1)),x),x[-1]*np.ones((k,1))))
            x_MA           = x_MA[(2*k+1) -1:] # Match dimensions
            # replace all the missing values
            x[indNaN[:,i]] = x_MA[indNaN[:,i]]
            X[:,i]         = x # Replace vector

    elif options["method"] == 2: # replace missing values after removing leading and closing zeros
        # Returns row sum for NaN values. Marks true for rows with more than 80% NaN
        rem1    = np.nansum(indNaN, axis =1) > (N * 0.8)
        nanLead = np.cumsum(rem1) == np.arange(1,(T+1))
        nanEnd  = np.cumsum(rem1) == np.arange(T,0,-1)
        nanLE   = nanLead|nanEnd

        # Subsets X
        X      = X[~nanLE,:]
        indNaN = np.isnan(X) # Index for missing values

        for i in range(N): # Loop for each series
            x          = X[:,i].copy()
            isnanx     = np.isnan(x)
            t1         = np.min(np.where(~isnanx)) # First non-NaN entry
            t2         = np.max(np.where(~isnanx)) # Last non-NaN entry

            # Interpolates without NaN entries in beginning and end
            x[t1:t2+1] = CubicSpline(np.where(~isnanx)[0],x[~isnanx])(np.arange(t1,t2+1))
            isnanx     = np.isnan(x)

            # replace NaN observations with median
            x[isnanx]  = np.nanmedian(x)

            # Apply filter
            x_MA       = lfilter(np.ones((2*k+1))/(2*k+1),1,np.append(np.append(x[0]*np.ones((k,1)),x),x[-1]*np.ones((k,1))))
            x_MA       = x_MA[(2*k+1) -1:]

            # Replace nanx wih filtered observations
            x[isnanx]  = x_MA[isnanx]
            X[:,i]     = x

    elif options["method"] == 3:
        rem1    = np.sum(indNaN, axis = 1) == N
        nanLead = np.cumsum(rem1) == np.arange(1,(T+1))
        nanEnd  = np.cumsum(rem1) == np.arange(T,0,-1)
        nanLE   = nanLead|nanEnd

        X      = X[~nanLE,:]
        indNaN = np.isnan(X)

    elif options["method"] == 4: # remove rows with leading and closing zeros & replace missing values
        rem1    = np.sum(indNaN, axis = 1) == N
        nanLead = np.cumsum(rem1) == np.arange(1,(T+1))
        nanEnd  = np.cumsum(rem1) == np.arange(T,0,-1)
        nanLE   = nanLead|nanEnd

        X      = X[~nanLE,:]
        indNaN = np.isnan(X)

        for i in range(N):
            x            = X[:, i].copy()
            isnanx       = np.isnan(x)
            t1           = np.min(np.where(~isnanx))
            t2           = np.max(np.where(~isnanx))
            x[t1:t2 + 1] = CubicSpline(np.where(~isnanx)[0],x[~isnanx])(np.arange(t1,t2+1))
            isnanx       = np.isnan(x)
            x[isnanx]    = np.nanmedian(x)
            x_MA         = lfilter(np.ones((2 * k + 1)) / (2 * k + 1), 1,
                                   np.append(np.append(x[0] * np.ones((k, 1)), x), x[-1] * np.ones((k, 1))))
            x_MA      = x_MA[(2 * k + 1) - 1:]
            x[isnanx] = x_MA[isnanx]
            X[:, i]   = x

    elif options["method"] == 5: # replace missing values
        indNaN = np.isnan(X)

        for i in range(N):
            x            = X[:, i].copy()
            isnanx       = np.isnan(x)
            t1           = np.min(np.where(~isnanx))
            t2           = np.max(np.where(~isnanx))
            x[t1:t2 + 1] = CubicSpline(np.where(~isnanx)[0],x[~isnanx])(np.arange(t1,t2+1))
            isnanx       = np.isnan(x)
            x[isnanx]    = np.nanmedian(x)
            x_MA         = lfilter(np.ones((2 * k + 1)) / (2 * k + 1), 1,
                                   np.append(np.append(x[0] * np.ones((k, 1)), x), x[-1] * np.ones((k, 1))))
            x_MA         = x_MA[(2 * k + 1) - 1:]
            x[isnanx]    = x_MA[isnanx]
            X[:, i]      = x

    return X,indNaN