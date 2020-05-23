#-------------------------------------------------Libraries
from datetime import datetime as dt
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from Functions.dfm import SKF,FIS


#-------------------------------------------------update_nowcast
def update_nowcast(X_old,X_new,Time,Spec,Res,series,period,vintage_old,vintage_new):

    if not type(vintage_old) == type(0):
        vintage_old = dt.strptime(vintage_old, '%Y-%m-%d').date().toordinal() + 366

    if not type(vintage_new) == type(0):
        vintage_new = dt.strptime(vintage_new, '%Y-%m-%d').date().toordinal() + 366

    # Make sure datasets are the same size
    N     = np.shape(X_new)[1]

    # append 1 year (12 months) of data to each dataset to allow for
    # forecasting at different horizons
    T_old = np.shape(X_old)[0]
    T_new = np.shape(X_new)[0]

    if T_new > T_old:

        temp    = np.zeros((T_new - T_old,N))
        temp[:] = np.nan
        X_old   = np.vstack([X_old,temp])

    temp    = np.zeros((12, N))
    temp[:] = np.nan

    # append 1 year (12 months) of data to each dataset to allow for
    # forecasting at different horizons
    X_old = np.vstack([X_old, temp])
    X_new = np.vstack([X_new, temp])

    future = np.array([(dt.fromordinal(Time[-1] - 366) +
                           relativedelta(months=+ i)).toordinal() + 366 for i in range(1,13)])

    Time = np.hstack([Time,future])

    i_series    = np.where(series == Spec.SeriesID)[0]
    series_name = Spec.SeriesName[i_series]
    freq        = Spec.Frequency[i_series][0]

    if freq == 'm':
        y, m      = period.split(freq)
        y         = int(y)
        m         = int(m)
        d         = 1
        t_nowcast = np.where((dt(y,m,d).toordinal()+366) == Time)[0]

    elif freq == 'q':
        y, q      = period.split(freq)
        y         = int(y)
        m         = 3 * int(q)
        d         = 1
        t_nowcast = np.where((dt(y,m,d).toordinal()+366) == Time)[0]

    else:
        return "Freq's value is not appropiate"

    if t_nowcast.size == 0:
        ValueError('Period is out of nowcasting horizon (up to one year ahead).')

    # Update nowcast for target variable 'series' (i) at horizon 'period' (t)
    #   > Relate nowcast update into news from data releases:
    #     a. Compute the impact from data revisions
    #     b. Compute the impact from new data releases

    X_rev                  = X_new.copy()
    X_rev[np.isnan(X_old)] = np.nan

    # Compute news --------------------------------------------------------

    # Compute impact from data revisions
    y_old,_,_,_,_,_,_,_,_ = News_DFM(X_old, X_rev, Res, t_nowcast, i_series)

    # Display output
    y_rev,y_new,_,actual,forecast,weight,_,_,_ = News_DFM(X_rev,X_new,Res,t_nowcast,i_series)

    print("\n Nowcast Update: {}".format(dt.fromordinal(vintage_new-366).isoformat().split('T')[0]))
    print("\n Nowcast for: {} ({}), {}".format(Spec.SeriesName[i_series][0],
                                               Spec.UnitsTransformed[i_series][0],
                                               pd.to_datetime(dt.fromordinal(Time[t_nowcast]-366)).to_period('Q')))

    # Only display table output if a forecast is made
    if forecast.shape[0] == 0:
        print("\n No forecast was made \n")
    else:
        impact_revisions = y_rev - y_old     # Impact from revisions
        news             = actual - forecast # News from releases
        impact_releases  = weight * news     # Impact of releases

        # Store results
        news_table = pd.DataFrame({'Forecast': forecast.flatten('F'),
                                   'Actual'  : actual.flatten('F'),
                                   'Weight'  : weight.flatten('F'),
                                   'Impact'  : impact_releases.flatten('F')},
                                  index = Spec.SeriesID)

        # Select only series with updates
        data_released = np.any(np.isnan(X_old) & ~np.isnan(X_new),0)

        # Display the impact decomposition
        print('\n Nowcast Impact Decomposition')
        print(' Note: The displayed output is subject to rounding error\n\n')
        print('              {} nowcast:              {}'.format(dt.fromordinal(vintage_old-366).isoformat().split('T')[0],y_old[0]))
        print('      Impact from data revisions:      {}'.format(impact_revisions[0]))
        print('       Impact from data releases:      {}'.format(np.nansum(news_table.Impact)))
        print('                                     +_________')
        print('                    Total impact:      {}'.format(impact_revisions[0] + np.nansum(news_table.Impact)))
        print('              {} nowcast:              {}'.format(dt.fromordinal(vintage_new-366).isoformat().split('T')[0],
                                                                                y_new[0]))

        print('\n  Nowcast Detail Table \n')
        print(news_table.iloc[np.where(data_released)[0],:])

def News_DFM(X_old,X_new,Res,t_fcst,v_news):
    # News_DFM()    Calculates changes in news
    # Syntax:
    # [y_old, y_new, singlenews, actual, fore, weight ,t_miss, v_miss, innov] = ...
    #   News_DFM(X_old, X_new, Q, t_fcst, v_news)
    #
    # Description:
    #  News DFM() inputs two datasets, DFM parameters, target time index, and
    #  target variable index. The function then produces Nowcast updates and
    #  decomposes the changes into news.
    #
    # Input Arguments:
    #   X_old:  Old data matrix (old vintage)
    #   X_new:  New data matrix (new vintage)
    #   Res:    DFM() output results (see DFM for more details)
    #   t_fcst: Index for target time
    #   v_news: Index for target variable
    #
    # Output Arguments:
    #   y_old:       Old nowcast
    #   y_new:       New nowcast
    #   single_news: News for each data series
    #   actual:      Observed series release values
    #   fore:        Forecasted series values
    #   weight:      News weight
    #   t_miss:      Time index for data releases
    #   v_miss:      Series index for data releases
    #   innov:       Difference between observed and predicted series values ("innovation")

    # Initialize variables
    r          = Res["C"].shape[1]
    N          = X_new.shape[1]
    singlenews = np.zeros((1,N)) # Initialize news vector (will store news for each series)


    # NO FORECAST CASE: Already values for variables v_news at time t_fcst
    if ~np.isnan(X_new[t_fcst,v_news])[0]:

        Res_old = para_const(X_old,Res,0) # Apply Kalman filter for old data

        y_old   = np.zeros((1,v_news.shape[0]))
        y_new   = np.zeros((1, v_news.shape[0]))
        for i in range(v_news.shape[0]): # Loop for each target variable

            # (Observed value) - (predicted value)
            singlenews[:,v_news[i]] = X_new[t_fcst,v_news[i]] - Res_old["X_sm"][t_fcst,v_news[i]]

            # Set predicted and observed y values
            y_old[0,i] = Res_old["X_sm"][t_fcst,v_news[i]].copy()
            y_new[0,i] = X_new[t_fcst, v_news[i]].copy()

        # Forecast-related output set to empty
        return y_old,y_new,singlenews,None,None,None,None,None,None

    else:
        # FORECAST CASE (these are broken down into (A) and (B))

        # Initialize series mean/standard deviation respectively
        Mx = Res["Mx"].reshape((-1,1))
        Wx = Res["Wx"].reshape((-1,1))

        # Calculate indicators for missing values (1 if missing, 0 otherwise)
        miss_old = np.isnan(X_old).astype(np.int64)
        miss_new = np.isnan(X_new).astype(np.int64)

        # Indicator for missing--combine above information to single matrix where:
        # (i) -1: Value is in the old data, but missing in new data
        # (ii) 1: Value is in the new data, but missing in old data
        # (iii) 0: Values are missing from/available in both datasets
        i_miss = miss_old - miss_new

        # Time/variable indicies where case (b) is true
        t_miss = np.where(i_miss == 1)[0]
        v_miss = np.where(i_miss == 1)[1]

        # FORECAST SUBCASE (A): NO NEW INFORMATION
        if v_miss.shape[0] == 0:

            # Fill in missing variables using a Kalman filter
            Res_old = para_const(X_old, Res, 0)
            Res_new = para_const(X_new, Res, 0) # CHECK: Why isn't this being used?

            # Set predicted and observed y values. New y value is set to old
            y_old = Res_old["X_sm"][t_fcst,v_news]
            y_new = y_old.copy()

            # No news, so nothing returned for news-related output
            return y_old,y_new,singlenews,None,None,None,None,None,None

        else:
            #----------------------------------------------------------------------
            #     v_miss=[1:size(X_new,2)]';
            #     t_miss=t_miss(1)*ones(size(X_new,2),1);
            #----------------------------------------------------------------------
            # FORECAST SUBCASE (B): NEW INFORMATION

            # Difference between forecast time and new data time
            lag = t_fcst - t_miss

            # Gives biggest time interval between forecast and new data
            k   = np.max(np.hstack([np.abs(lag),np.max(lag) - np.min(lag)]))

            C = Res["C"].copy() # Observation matrix
            R = Res["R"].copy() # Covariance for observation matrix residuals

            # Number of new events
            n_news = lag.shape[0]

            # Smooth old dataset
            Res_old = para_const(X_old, Res, k)
            Plag    = Res_old["Plag"].copy()

            # Smooth new dataset
            Res_new = para_const(X_new, Res, 0)

            # Subset for target variable and forecast time
            y_old = Res_old["X_sm"][t_fcst,v_news]
            y_new = Res_new["X_sm"][t_fcst,v_news]

            P  = Res_old["P"][1:].copy()

            for i in range(n_news): # Cycle through total number of updates
                h = abs(t_fcst-t_miss[i])[0]
                m = max(t_miss[i],t_fcst)[0]

                # If location of update is later than the forecasting date
                if t_miss[i] > t_fcst:
                    Pp = Plag[h][m].copy()
                else:
                    Pp = Plag[h][m].T.copy()
                if i == 0:
                    # Initialize projection onto updates
                    P1 = np.matmul(Pp,C[[v_miss[i]]][:,:r].T)
                else:
                    # Projection on updates
                    P1 = np.hstack([P1,np.matmul(Pp,C[[v_miss[i]]][:,:r].T)])

            for i in range(t_miss.shape[0]):
                # Standardize predicted and observed values
                X_new_norm = (X_new[t_miss[i],v_miss[i]] - Mx[v_miss[i]])/Wx[v_miss[i]]
                X_sm_norm  = (Res_old["X_sm"][t_miss[i],v_miss[i]] - Mx[v_miss[i]])/Wx[v_miss[i]]

                # Innovation: Gives [observed] data - [predicted data]
                if i == 0:
                    innov = X_new_norm - X_sm_norm
                else:
                    innov = np.hstack([innov,X_new_norm - X_sm_norm])
            innov = innov.reshape((1,-1))

            ins   = len(innov)
            WW    = np.zeros((v_miss[-1]+1,v_miss[-1]+1))
            WW[:] = np.nan

            # Gives non-standardized series weights
            for i in range(lag.shape[0]):
                for j in range(lag.shape[0]):
                    h = abs(lag[i] - lag[j])
                    m = max(t_miss[i],t_miss[j])

                    if t_miss[j] > t_miss[j]:
                        Pp = Plag[h][m].copy()
                    else:
                        Pp = Plag[h][m].T.copy()

                    if v_miss[i] == v_miss[j] and t_miss[i] != t_miss[j]:
                        WW[v_miss[i],v_miss[j]] = 0
                    else:
                        WW[v_miss[i], v_miss[j]] = R[v_miss[i],v_miss[j]].copy()

                    if j == 0:
                        p2 = np.matmul(np.matmul(C[[v_miss[i]]][:,:r],Pp),C[[v_miss[j]]][:,:r].T) + WW[v_miss[i], v_miss[j]]
                    else:
                        p2 = np.hstack([p2,np.matmul(np.matmul(C[[v_miss[i]]][:,:r],Pp),C[[v_miss[j]]][:,:r].T) + WW[v_miss[i], v_miss[j]]])
                if i == 0:
                    P2 = p2.copy()
                else:
                    P2 = np.vstack([P2,p2])

            try:
                del temp
            except NameError:
                pass

            # CHECK: can this be written better?
            for i in range(v_news.shape[0]): # loop on v_news
                # Convert to real units (unstadardized data)
                if i == 0:
                    totnews = np.matmul(np.matmul(np.matmul(np.matmul(Wx[[v_news[i]]],C[[v_news[i]]][:,:r]),P1),np.linalg.inv(P2)),innov.T)
                    temp    = np.matmul(np.matmul(np.matmul(Wx[[v_news[i]]],C[[v_news[i]]][:,:r]),P1),np.linalg.inv(P2)) * innov
                    gain    = np.matmul(np.matmul(np.matmul(Wx[[v_news[i]]],C[[v_news[i]]][:,:r]),P1),np.linalg.inv(P2))

                    temp = temp.reshape((1,*temp.shape))
                    gain = gain.reshape((1,*gain.shape))
                else:
                    temp_A = np.matmul(np.matmul(np.matmul(np.matmul(Wx[v_news[i]],C[[v_news[i]]][:,:r]),P1),np.linalg.inv(P2)),innov.T)
                    temp_B = np.matmul(np.matmul(np.matmul(Wx[v_news[i]],C[[v_news[i]]][:,:r]),P1),np.linalg.inv(P2)) * innov
                    temp_C = np.matmul(np.matmul(np.matmul(Wx[v_news[i]],C[[v_news[i]]][:,:r]),P1),np.linalg.inv(P2))

                    totnews = np.hstack([totnews, temp_A])
                    temp    = np.vstack([temp,temp_B[np.newaxis,]])
                    gain    = np.vstack([gain, temp_C[np.newaxis,]])

            # Initialize output objects
            singlenews = np.zeros((v_news.shape[0],np.max(t_miss) - np.min(t_miss)+1,N))
            actual     = np.zeros((N,1))
            forecast   = np.zeros((N,1))
            weight     = np.zeros((v_news.shape[0],N,1))
            singlenews[:], actual[:], forecast[:], weight[:] = np.nan,np.nan,np.nan,np.nan

            # Fill in output values
            for i in range(innov.shape[1]):
                actual[v_miss[i],0]   = X_new[t_miss[i],v_miss[i]].copy()
                forecast[v_miss[i],0] = Res_old["X_sm"][t_miss[i],v_miss[i]].copy()

                for j in range(v_news.shape[0]):
                    singlenews[j,t_miss[i]-min(t_miss),v_miss[i]] = temp[j,0,i].copy()
                    weight[j,v_miss[i],:] = gain[j,:,i]/Wx[v_miss[i]]

            singlenews = np.sum(singlenews, axis = 0) # Returns total news
            v_miss     = np.sort(np.unique(v_miss))

    # CHECK: weight seems suspicious
    return y_old,y_new,singlenews,actual,forecast,weight[0],t_miss,v_miss,innov

def para_const(X,P,lag):
    # para_const()    Implements Kalman filter for "News_DFM.m"
    #
    #   Syntax:
    #     Res = para_const(X,P,lag)
    #
    #   Description:
    #     para_const() implements the Kalman filter for the news calculation
    #     step. This procedure smooths and fills in missing data for a given
    #     data matrix X. In contrast to runKF(), this function is used when
    #     model parameters are already estimated.
    #
    #   Input parameters:
    #     X: Data matrix.
    #     P: Parameters from the dynamic factor model.
    #     lag: Number of lags
    #
    #   Output parameters:
    #     Res [struc]: A structure containing the following:
    #       Res.Plag: Smoothed factor covariance for transition matrix
    #       Res.P:    Smoothed factor covariance matrix
    #       Res.X_sm: Smoothed data matrix
    #       Res.F:    Smoothed factors
    #
    #
    #
    #  Kalman filter with specified paramaters
    #  written for
    #  "MAXIMUM LIKELIHOOD ESTIMATION OF FACTOR MODELS ON DATA SETS WITH
    #  ARBITRARY PATTERN OF MISSING DATA."
    #  by Marta Banbura and Michele Modugno
    #
    #   Set model parameters and data preparation

    # Set model parameters
    Z_0 = P["Z_0"].copy()
    V_0 = P["V_0"].copy()
    A   = P["A"].copy()
    C   = P["C"].copy()
    Q   = P["Q"].copy()
    R   = P["R"].copy()
    Mx  = P["Mx"].copy()
    Wx  = P["Wx"].copy()

    # Prepare data
    T = X.shape[0]

    # Standardise x
    Y = ((X - np.tile(Mx,(T,1)))/np.tile(Wx,(T,1))).T

    # Apply Kalman filter and smoother
    # See runKF() for details about FIS and SKF
    Sf = SKF(Y,A,C,Q,R,Z_0,V_0) # Kalman filter
    Ss = FIS(A,Sf)              # Smoothing step

    # Calculate parameter output
    Vs      = Ss["VmT"][1:,:,:].copy() # Smoothed factor covariance for transition matrix
    Vf      = Ss["VmU"][1:,:,:].copy() # Filtered factor posterior covariance
    Zsmooth = Ss["ZmT"].copy()         # Smoothed factors
    Vsmooth = Ss["VmT"].copy()         # Smoothed covariance values

    Plag    = [Vs.copy()]

    for jk in range(1,lag+1):
        reset = True
        for jt in range(Plag[0].shape[0]-1,lag-1,-1):
            As = np.matmul(np.matmul(Vf[jt-jk],A.T),
                           np.linalg.pinv(np.matmul(np.matmul(A,Vf[jt-jk]),A.T) + Q))
            if reset:
                reset      = False
                base       = np.matmul(As,Plag[jk-1][jt])
                fill       = np.zeros((Plag[0].shape[0]-(lag),*base.shape))
                fill[:]    = np.nan
                fill[jt-1] = base
                Plag.append(fill)
            else:
                Plag[jk][jt] = np.matmul(As, Plag[jk-1][jt])

    # Prepare data for output
    Zsmooth = Zsmooth.T

    x_sm = np.matmul(Zsmooth[1:,:],C.T)                 # Factors to series representation
    X_sm = np.tile(Wx,(T,1)) * x_sm + np.tile(Mx,(T,1)) # Standardized to unstandardized

    # Loading dictionary with the results
    Res = {"Plag" : Plag,
           "P"    : Vsmooth,
           "X_sm" : X_sm,
           "F"    : Zsmooth[1:,:]
    }

    return Res