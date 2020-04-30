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

    N     = np.shape(X_new)[1]
    T_old = np.shape(X_old)[0]
    T_new = np.shape(X_new)[0]

    if T_new > T_old:

        temp    = np.zeros((T_new - T_old,N))
        temp[:] = np.nan
        X_old   = np.vstack([X_old,temp])

    temp    = np.zeros((12, N))
    temp[:] = np.nan

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

    X_rev                  = X_new.copy()
    X_rev[np.isnan(X_old)] = np.nan

    y_old,_,_,_,_,_,_,_,_ = News_DFM(X_old, X_rev, Res, t_nowcast, i_series)

    y_rev,y_new,_,actual,forecast,weight,_,_,_ = News_DFM(X_rev,X_new,Res,t_nowcast,i_series)

    print("\n Nowcast Update: {}".format(dt.fromordinal(vintage_new-366).isoformat().split('T')[0]))
    print("\n Nowcast for: {} ({}), {}".format(Spec.SeriesName[i_series][0],
                                               Spec.UnitsTransformed[i_series][0],
                                               pd.to_datetime(dt.fromordinal(Time[t_nowcast]-366)).to_period('Q')))

    if forecast.shape[0] == 0:
        print("\n No forecast was made \n")
    else:
        impact_revisions = y_rev - y_old
        news             = actual - forecast
        impact_releases  = weight * news

        news_table = pd.DataFrame({'Forecast': forecast.flatten('F'),
                                   'Actual'  : actual.flatten('F'),
                                   'Weight'  : weight.flatten('F'),
                                   'Impact'  : impact_releases.flatten('F')},
                                  index = Spec.SeriesID)

        data_released = np.any(np.isnan(X_old) & ~np.isnan(X_new),0)

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

    r          = Res["C"].shape[1]
    N          = X_new.shape[1]
    singlenews = np.zeros((1,N))


    # CHECK: NEEDS TO BE CHECKED
    if ~np.isnan(X_new[t_fcst,v_news])[0]:

        Res_old = para_const(X_old,Res,0)
        y_old   = np.zeros((1,v_news.shape[0]))
        y_new   = np.zeros((1, v_news.shape[0]))

        for i in range(v_news.shape[0]):
            singlenews[:,v_news[i]] = X_new[t_fcst,v_news[i]] - Res_old["X_sm"][t_fcst,v_news[i]]
            y_old[0,i] = Res_old["X_sm"][t_fcst,v_news[i]].copy()
            y_new[0,i] = X_new[t_fcst, v_news[i]].copy()


        return y_old,y_new,singlenews,None,None,None,None,None,None

    else:
        Mx = Res["Mx"].reshape((-1,1))
        Wx = Res["Wx"].reshape((-1,1))

        miss_old = np.isnan(X_old).astype(np.int64)
        miss_new = np.isnan(X_new).astype(np.int64)

        i_miss = miss_old - miss_new

        t_miss = np.where(i_miss == 1)[0]
        v_miss = np.where(i_miss == 1)[1]

        if v_miss.shape[0] == 0:
            Res_old = para_const(X_old, Res, 0)
            Res_new = para_const(X_new, Res, 0)

            y_old = Res_old["X_sm"][t_fcst,v_news]
            y_new = y_old.copy()

            return y_old,y_new,singlenews,None,None,None,None,None,None

        else:
            lag = t_fcst - t_miss
            k   = np.max(np.hstack([np.abs(lag),np.max(lag) - np.min(lag)]))

            C = Res["C"].copy()
            R = Res["R"].copy()

            n_news = lag.shape[0]

            Res_old = para_const(X_old, Res, k)
            Plag    = Res_old["Plag"].copy()

            Res_new = para_const(X_new, Res, 0)

            y_old = Res_old["X_sm"][t_fcst,v_news]
            y_new = Res_new["X_sm"][t_fcst,v_news]

            P  = Res_old["P"][1:].copy()

            for i in range(n_news):
                h = abs(t_fcst-t_miss[i])[0]
                m = max(t_miss[i],t_fcst)[0]

                if t_miss[i] > t_fcst:
                    Pp = Plag[h][m].copy()
                else:
                    Pp = Plag[h][m].T.copy()
                if i == 0:
                    P1 = np.matmul(Pp,C[[v_miss[i]]][:,:r].T)
                else:
                    P1 = np.hstack([P1,np.matmul(Pp,C[[v_miss[i]]][:,:r].T)])

            for i in range(t_miss.shape[0]):
                X_new_norm = (X_new[t_miss[i],v_miss[i]] - Mx[v_miss[i]])/Wx[v_miss[i]]
                X_sm_norm  = (Res_old["X_sm"][t_miss[i],v_miss[i]] - Mx[v_miss[i]])/Wx[v_miss[i]]

                if i == 0:
                    innov = X_new_norm - X_sm_norm
                else:
                    innov = np.hstack([innov,X_new_norm - X_sm_norm])

            innov = innov.reshape((1,-1))
            ins   = len(innov)

            WW    = np.zeros((v_miss[-1]+1,v_miss[-1]+1))
            WW[:] = np.nan

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

            for i in range(v_news.shape[0]):
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

            singlenews = np.zeros((v_news.shape[0],np.max(t_miss) - np.min(t_miss)+1,N))
            actual     = np.zeros((N,1))
            forecast   = np.zeros((N,1))
            weight     = np.zeros((v_news.shape[0],N,1))

            singlenews[:], actual[:], forecast[:], weight[:] = np.nan,np.nan,np.nan,np.nan

            for i in range(innov.shape[1]):
                actual[v_miss[i],0]   = X_new[t_miss[i],v_miss[i]].copy()
                forecast[v_miss[i],0] = Res_old["X_sm"][t_miss[i],v_miss[i]].copy()

                for j in range(v_news.shape[0]):
                    singlenews[j,t_miss[i]-min(t_miss),v_miss[i]] = temp[j,0,i].copy()
                    weight[j,v_miss[i],:] = gain[j,:,i]/Wx[v_miss[i]]

            singlenews = np.sum(singlenews, axis = 0)
            v_miss     = np.sort(np.unique(v_miss))

    # CHECK: weight seems suspicious
    return y_old,y_new,singlenews,actual,forecast,weight[0],t_miss,v_miss,innov

def para_const(X,P,lag):
    Z_0 = P["Z_0"].copy()
    V_0 = P["V_0"].copy()
    A   = P["A"].copy()
    C   = P["C"].copy()
    Q   = P["Q"].copy()
    R   = P["R"].copy()
    Mx  = P["Mx"].copy()
    Wx  = P["Wx"].copy()

    T = X.shape[0]
    Y = ((X - np.tile(Mx,(T,1)))/np.tile(Wx,(T,1))).T

    Sf = SKF(Y,A,C,Q,R,Z_0,V_0)
    Ss = FIS(A,Sf)

    Vs      = Ss["VmT"][1:,:,:].copy()
    Vf      = Ss["VmU"][1:,:,:].copy()
    Zsmooth = Ss["ZmT"].copy()
    Vsmooth = Ss["VmT"].copy()

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

    Zsmooth = Zsmooth.T

    x_sm = np.matmul(Zsmooth[1:,:],C.T)
    X_sm = np.tile(Wx,(T,1)) * x_sm + np.tile(Mx,(T,1))

    Res = {"Plag" : Plag,
           "P"    : Vsmooth,
           "X_sm" : X_sm,
           "F"    : Zsmooth[1:,:]
    }

    return Res