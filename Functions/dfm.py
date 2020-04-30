#-------------------------------------------------Libraries
import numpy as np
import pandas as pd
from remNaNs_spline import remNaNs_spline
from scipy.linalg import eig
from scipy.linalg import block_diag


#-------------------------------------------------Dynamic Factor Modeling functions
def dfm(X,Spec,threshold = 1e-5,max_iter = 5000):
    # DFM()    Runs the dynamic factor model
    #
    #  Syntax:
    #    Res = DFM(X,Par)
    #
    #  Description:
    #   DFM() inputs the organized and transformed data X and parameter structure Par.
    #   Then, the function outputs dynamic factor model structure Res and data
    #   summary statistics (mean and standard deviation).
    #
    #  Input arguments:
    #    X: Kalman-smoothed data where missing values are replaced by their expectation
    #    Par: A structure containing the following parameters:
    #      Par.blocks: Block loadings.
    #      Par.nQ: Number of quarterly series
    #      Par.p: Number of lags in transition matrix
    #      Par.r: Number of common factors for each block
    #
    # Output Arguments:
    #
    #   Res - structure of model results with the following fields
    #       . X_sm | Kalman-smoothed data where missing values are replaced by their expectation
    #       . Z | Smoothed states. Rows give time, and columns are organized according to Res.C.
    #       . C | Observation matrix. The rows correspond
    #          to each series, and the columns are organized as shown below:
    #         - 1-20: These columns give the factor loa dings. For example, 1-5
    #              give loadings for the first block and are organized in
    #              reverse-chronological order (f^G_t, f^G_t-1, f^G_t-2, f^G_t-3,
    #              f^G_t-4). Columns 6-10, 11-15, and 16-20 give loadings for
    #              the second, third, and fourth blocks respectively.
    #       .R: Covariance for observation matrix residuals
    #       .A: Transition matrix. This is a square matrix that follows the
    #      same organization scheme as Res.C's columns. Identity matrices are
    #      used to account for matching terms on the left and righthand side.
    #      For example, we place an I4 matrix to account for matching
    #      (f_t-1; f_t-2; f_t-3; f_t-4) terms.
    #       .Q: Covariance for transition equation residuals.
    #       .Mx: Series mean
    #       .Wx: Series standard deviation
    #       .Z_0: Initial value of state
    #       .V_0: Initial value of covariance matrix
    #       .r: Number of common factors for each block
    #       .p: Number of lags in transition equation
    #
    # References:
    #
    #   Marta Banbura, Domenico Giannone and Lucrezia Reichlin
    #   Nowcasting (2010)
    #   Michael P. Clements and David F. Hendry, editors,
    #   Oxford Handbook on Economic Forecasting.

    ## Store model parameters ------------------------------------------------

    # DFM input specifications: See documentation for details
    Par = {}
    Par["blocks"] = Spec.Blocks.copy()                                  # Block loading structure
    Par["nQ"]     = (Spec.Frequency == "q").sum()                       # Number of quarterly series
    Par["p"]      = 1                                                   # Number of lags in autoregressive of factor (same for all factors)
    Par["r"]      = np.ones((1,Spec.Blocks.shape[1])).astype(np.int64)  # Number of common factors for each block
    # Par.r(1) =2;
    # Display blocks

    print(pd.DataFrame(data=Spec.Blocks, index=Spec.SeriesName, columns=Spec.BlockNames))
    print("Estimating the dynamic factor model (DFM)")

    T, N   = X.shape
    r      = Par["r"].copy()
    p      = Par["p"]
    nQ     = Par["nQ"]
    blocks = Par["blocks"].copy()

    i_idio = np.append(np.ones(N-nQ),np.zeros(nQ)).reshape((-1,1),order="F") == 1

    # R*Lambda = q; Contraints on the loadings of the quartrly variables

    R_mat = np.array([2,-1,0,0,0,3,0,-1,0,0,2,0,0,-1,0,1,0,0,0,-1]).reshape((4,5))
    q     = np.zeros((4,1))

    # Prepare data
    Mx   = np.nanmean(X,axis = 0)
    Wx   = np.nanstd(X,axis = 0,ddof = 1)
    xNaN = (X - np.tile(Mx,(T,1))) / np.tile(Wx,(T,1))

    # Initial Conditions-------------------------------------------------------
    optNaN           = {}
    optNaN["method"] = 2 # Remove leading and closing zeros
    optNaN["k"]      = 3 # Setting for filter(): See remNaN_spline

    A,C,Q,R,Z_0,V_0  = InitCond(xNaN.copy(), r.copy(), p, blocks.copy(), optNaN, R_mat.copy(), q, nQ, i_idio.copy())

    previous_loglik = -np.inf
    num_iter        = 0
    LL              = [-np.inf]
    converged       = 0

    y                = xNaN.copy().T
    optNaN["method"] = 3
    y_est,_          = remNaNs_spline(xNaN.copy(),optNaN)
    y_est            = y_est.T

    max_iter = 5000
    while num_iter < max_iter and not converged:
        C_new, R_new, A_new, Q_new, Z_0, V_0, loglik = EMstep(y_est, A, C, Q, R, Z_0, V_0, r,p,R_mat,q,nQ,i_idio,blocks)

        C = C_new.copy()
        R = R_new.copy()
        A = A_new.copy()
        Q = Q_new.copy()

        if num_iter > 2:
            converged, decrease = em_converged(loglik,previous_loglik,threshold,1)

        if (num_iter % 10) == 0 and num_iter > 0:
            print("Now running the {}th iteration of max {}".format(num_iter,max_iter))
            print('Loglik: {} (% Change: {})'.format(loglik, 100*((loglik-previous_loglik)/previous_loglik)))

        LL.append(loglik)
        previous_loglik = loglik
        num_iter        += 1

    if num_iter < max_iter:
        print('Successful: Convergence at {} interations'.format(num_iter))
    else:
        print('Stopped because maximum iterations reached')

    Zsmooth,_,_,_  = runKF(y,A,C,Q,R,Z_0,V_0)
    Zsmooth        = Zsmooth.T
    x_sm           = np.matmul(Zsmooth[1:,:],C.T)

    Res = { "x_sm"     : x_sm.copy(),
            "X_sm" : np.tile(Wx,(T,1)) * x_sm + np.tile(Mx,(T,1)),
            "Z"        : Zsmooth[1:,:].copy(),
            "C"        : C.copy(),
            "R"        : R.copy(),
            "A"        : A.copy(),
            "Q"        : Q.copy(),
            "Mx"       : Mx.copy(),
            "Wx"       : Wx.copy(),
            "Z_0"      : Z_0.copy(),
            "V_0"      : V_0.copy(),
            "r"        : r,
            "p"        : p
    }

    nQ       = Par["nQ"]
    nM       = Spec.SeriesID.shape[0] - nQ
    nLags    = max(Par["p"],5)
    nFactors = np.sum(Par["r"])

    print("\n Table 4: Factor Loadings for Monthly Series")
    print(pd.DataFrame(Res["C"][:nM,np.arange(0,nFactors*5,5)],
                       columns = Spec.BlockNames,
                       index   = Spec.SeriesName[:nM]))

    print("\n Table 5: Quarterly Loadings Sample (Global Factor)")
    print(pd.DataFrame(Res["C"][(-1-nQ+1):,:5],
                       columns = ['f1_lag0', 'f1_lag1', 'f1_lag2', 'f1_lag3', 'f1_lag4'],
                       index   = Spec.SeriesName[-1-nQ+1:]))

    A_terms = np.diag(Res["A"]).copy()
    Q_terms = np.diag(Res["Q"]).copy()

    print('\n Table 6: Autoregressive Coefficients on Factors')
    print(pd.DataFrame({'AR_Coefficient'    : A_terms[np.arange(0,nFactors*5,5)].copy(),
                        'Variance_Residual' : Q_terms[np.arange(0,nFactors*5,5)].copy()},
                       index   = Spec.BlockNames))

    print('\n Table 7: Autoregressive Coefficients on Idiosyncratic Component')
    A_len = A.shape[0]
    Q_len = Q.shape[0]

    A_index = np.hstack([np.arange(nFactors*5,nFactors*5+nM),np.arange(nFactors*5+nM,A_len,5)])
    Q_index = np.hstack([np.arange(nFactors*5,nFactors*5+nM),np.arange(nFactors*5+nM,Q_len,5)])

    print(pd.DataFrame({'AR_Coefficient'    : A_terms[A_index].copy(),
                        'Variance_Residual' : Q_terms[Q_index].copy()},
                       index   = Spec.SeriesName))

    return Res

def InitCond(x,r,p,blocks,optNaN,Rcon,q,nQ,i_idio):
    #InitCond()      Calculates initial conditions for parameter estimation
    #
    #  Description:
    #    Given standardized data and model information, InitCond() creates
    #    initial parameter estimates. These are intial inputs in the EM
    #    algorithm, which re-estimates these parameters using Kalman filtering
    #    techniques.
    #
    #Inputs:
    #  - x:      Standardized data
    #  - r:      Number of common factors for each block
    #  - p:      Number of lags in transition equation
    #  - blocks: Gives series loadings
    #  - optNaN: Option for missing values in spline. See remNaNs_spline() for details.
    #  - Rcon:   Incorporates estimation for quarterly series (i.e. "tent structure")
    #  - q:      Constraints on loadings for quarterly variables
    #  - NQ:     Number of quarterly variables
    #  - i_idio: Logical. Gives index for monthly variables (1) and quarterly (0)
    #
    #Output:
    #  - A:   Transition matrix
    #  - C:   Observation matrix
    #  - Q:   Covariance for transition equation residuals
    #  - R:   Covariance for observation equation residuals
    #  - Z_0: Initial value of state
    #  - V_0: Initial value of covariance matrix

    pC  = Rcon.shape[1]   # Gives 'tent' structure size (quarterly to monthly)
    ppC = max(p,pC)
    n_b = blocks.shape[1] # Number of blocks

    xBal,indNaN = remNaNs_spline(x.copy(),optNaN)  # Spline without NaNs
    
    T,N = xBal.shape  # Time T series number N
    nM  = N-nQ        # Number of monthly series
    
    xNaN         = xBal.copy()
    xNaN[indNaN] = np.nan        # Set missing values equal to NaNs
    res          = xBal.copy()   # Spline output equal to res Later this is used for residuals
    resNaN       = xNaN.copy()   # Later used for residuals

    # Initialize model coefficient output
    C   = None
    A   = None
    Q   = None
    V_0 = None

    indNaN[:pC-1, :] = np.True_

    # Set the first observations as NaNs: For quarterly-monthly aggreg. scheme
    for i in range(n_b):
        r_i = r[0,i].copy() # r_i = 1 when block is loaded

        # Observation equation -----------------------------------------------

        C_i = np.zeros((N, r_i * ppC))  # Initialize state variable matrix helper
        idx_i = np.where(blocks[:, i])[0]  # Returns series index loading block i
        idx_iM = idx_i[idx_i < nM]  # Monthly series indicies for loaded blocks
        idx_iQ = idx_i[idx_i >= nM]  # Quarterly series indicies for loaded blocks

        # Returns eigenvector v w/largest eigenvalue d
        d, v = eig(np.cov(res[:, idx_iM], rowvar=False))
        e_idx = np.where(d == np.max(d))[0]
        d = d[e_idx]
        v = v[:, e_idx]

        # Flip sign for cleaner output. Gives equivalent results without this section
        if np.sum(v) < 0:
            v = -v

        # For monthly series with loaded blocks (rows), replace with eigenvector
        # This gives the loading
        C_i[idx_iM,0:r_i] = v.copy()
        f                 = np.matmul(res[:,idx_iM],v) # Data projection for eigenvector direction
        F                 = np.array(f[(pC-1):f.shape[0],:]).reshape((-1,1))

        # Lag matrix using loading. This is later used for quarterly series
        for kk in range(1,max(p+1,pC)):
            F = np.concatenate((F,f[(pC-1)-kk:f.shape[0]-kk,:]), axis =1)

        Rcon_i = np.kron(Rcon,np.eye(r_i)) # Quarterly-monthly aggregation scheme
        q_i    = np.kron(q,np.zeros((r_i,1)))

        # Produces projected data with lag structure (so pC-1 fewer entries)
        ff = F[:, 0:(r_i*pC)].copy()

        for j in idx_iQ:

            xx_j = resNaN[(pC-1):,j].copy()

            if sum(~np.isnan(xx_j)) < (ff.shape[1] + 2):
                xx_j = res[(pC-1):,j].copy()

            ff_j = ff[~np.isnan(xx_j),:].copy()
            xx_j = xx_j[~np.isnan(xx_j)].reshape((-1,1)).copy()

            iff_j = np.linalg.inv(np.matmul(ff_j.T,ff_j))
            Cc    = np.matmul(np.matmul(iff_j,ff_j.T),xx_j)

            a1 = np.matmul(iff_j,Rcon_i.T)
            a2 = np.linalg.inv(np.matmul(np.matmul(Rcon_i,iff_j),Rcon_i.T))
            a3 = np.matmul(Rcon_i,Cc)-q_i
            Cc = Cc - np.matmul(np.matmul(a1,a2),a3)

            C_i[j,0:pC*r_i] = Cc.T.copy()

        ff = np.concatenate([np.zeros((pC-1,pC*r_i)),ff], axis = 0)

        res            = res - np.matmul(ff,C_i.T)
        resNaN         = res.copy()
        resNaN[indNaN] = np.nan

        if i == 0:
            C = C_i.copy()
        else:
            C = np.hstack([C,C_i.copy()])

        z = F[:,r_i-1].copy()
        Z = F[:,r_i:(r_i*(p+1))].copy()

        A_i    = np.zeros((r_i*ppC,r_i*ppC)).T
        A_temp = np.matmul(np.matmul(np.linalg.inv(np.matmul(Z.T,Z)),Z.T),z)

        A_i[:r_i,:r_i*p]       = A_temp.T.copy()
        A_i[r_i:,:r_i*(ppC-1)] = np.eye(r_i*(ppC-1))

        Q_i            = np.zeros((ppC*r_i,ppC*r_i))
        e              = z - np.matmul(Z,A_temp)
        Q_i[:r_i,:r_i] = np.cov(e, rowvar=False)

        initV_i = np.reshape(np.matmul(np.linalg.inv(np.eye((r_i*ppC)**2) - np.kron(A_i,A_i)),Q_i.flatten('F').reshape((-1,1))),(r_i*ppC,r_i*ppC))

        if i == 0:
            A   = A_i.copy()
            Q   = Q_i.copy()
            V_0 = initV_i.copy()
        else:
            A   = block_diag(A,A_i)
            Q   = block_diag(Q,Q_i)
            V_0 = block_diag(V_0,initV_i)

    eyeN = np.eye(N)[:,i_idio.flatten('F')]
    C    = np.hstack([C,eyeN])
    C    = np.hstack([C,np.vstack([np.zeros((nM,5*nQ)),np.kron(np.eye(nQ),np.array([1,2,3,2,1]).reshape((1,-1)))])])
    R    = np.diag(np.nanvar(resNaN,ddof = 1,axis = 0))

    ii_idio = np.where(i_idio)[0]
    n_idio  = ii_idio.shape[0]
    BM      = np.zeros((n_idio,n_idio))
    SM      = np.zeros((n_idio, n_idio))

    for i in range(n_idio):

        R[ii_idio[i],ii_idio[i]] = 1e-4

        res_i = resNaN[:,ii_idio[i]].copy()

        try:
            leadZero = np.max(np.where(np.arange(1,T+1) == np.cumsum(np.isnan(res_i)))) + 1
        except ValueError:
            leadZero = None

        try:
            endZero  = -(np.max(np.where(np.arange(1,T+1) == np.cumsum(np.isnan(res_i[::-1])))[0]) + 1)
        except ValueError:
            endZero = None

        res_i = res[:,ii_idio[i]].copy()
        res_i = res_i[:endZero]
        res_i = res_i[leadZero:].reshape((-1,1),order="F")

        BM[i,i] = np.matmul(np.matmul(np.linalg.inv(np.matmul(res_i[:-1].T,res_i[:-1])),res_i[:-1].T),res_i[1:])
        SM[i,i] = np.cov(res_i[1:] - (res_i[:-1]*BM[i,i]),rowvar=False)

    Rdiag       = np.diag(R).copy()
    sig_e       = (Rdiag[nM:]/19)
    Rdiag[nM:]  = 1e-4
    R           = np.diag(Rdiag).copy()

    rho0      = np.array([[.1]])
    temp      = np.zeros((5,5))
    temp[0,0] = 1

    SQ = np.kron(np.diag((1 - rho0[0,0]**2)*sig_e),temp)
    BQ = np.kron(np.eye(nQ),np.vstack([np.hstack([rho0,np.zeros((1,4))]),np.hstack([np.eye(4),np.zeros((4,1))])]))

    initViQ = np.matmul(np.linalg.inv(np.eye((5*nQ)**2) - np.kron(BQ,BQ)),SQ.reshape((-1,1))).reshape((5*nQ,5*nQ))
    initViM = np.diag(1/np.diag(np.eye(BM.shape[0]) - BM**2))*SM

    A   = block_diag(A,BM,BQ)
    Q   = block_diag(Q,SM,SQ)
    Z_0 = np.zeros((A.shape[0],1))
    V_0 = block_diag(V_0,initViM,initViQ)

    return A, C, Q, R, Z_0, V_0

def EMstep(y, A, C, Q, R, Z_0, V_0, r,p,R_mat,q,nQ,i_idio,blocks):
    #EMstep    Applies EM algorithm for parameter reestimation
    #
    #  Syntax:
    #    [C_new, R_new, A_new, Q_new, Z_0, V_0, loglik]
    #    = EMstep(y, A, C, Q, R, Z_0, V_0, r, p, R_mat, q, nQ, i_idio, blocks)
    #
    #  Description:
    #    EMstep reestimates parameters based on the Estimation Maximization (EM)
    #    algorithm. This is a two-step procedure:
    #    (1) E-step: the expectation of the log-likelihood is calculated using
    #        previous parameter estimates.
    #    (2) M-step: Parameters are re-estimated through the maximisation of
    #        the log-likelihood (maximize result from (1)).
    #
    #    See "Maximum likelihood estimation of factor models on data sets with
    #    arbitrary pattern of missing data" for details about parameter
    #    derivation (Banbura & Modugno, 2010). This procedure is in much the
    #    same spirit.
    #
    #  Input:
    #    y:      Series data
    #    A:      Transition matrix
    #    C:      Observation matrix
    #    Q:      Covariance for transition equation residuals
    #    R:      Covariance for observation matrix residuals
    #    Z_0:    Initial values of factors
    #    V_0:    Initial value of factor covariance matrix
    #    r:      Number of common factors for each block (e.g. vector [1 1 1 1])
    #    p:      Number of lags in transition equation
    #    R_mat:  Estimation structure for quarterly variables (i.e. "tent")
    #    q:      Constraints on loadings
    #    nQ:     Number of quarterly series
    #    i_idio: Indices for monthly variables
    #    blocks: Block structure for each series (i.e. for a series, the structure
    #            [1 0 0 1] indicates loadings on the first and fourth factors)
    #
    #  Output:
    #    C_new: Updated observation matrix
    #    R_new: Updated covariance matrix for residuals of observation matrix
    #    A_new: Updated transition matrix
    #    Q_new: Updated covariance matrix for residuals for transition matrix
    #    Z_0:   Initial value of state
    #    V_0:   Initial value of covariance matrix
    #    loglik: Log likelihood
    #
    # References:
    #   "Maximum likelihood estimation of factor models on data sets with
    #   arbitrary pattern of missing data" by Banbura & Modugno (2010).
    #   Abbreviated as BM2010
    #
    #

    # Initialize preliminary values

    # Store series/model values
    n,T        = y.shape
    nM         = n - nQ
    pC         = R_mat.shape[1]
    ppC        = max(p,pC)
    num_blocks = blocks.shape[1]

    Zsmooth,Vsmooth,VVsmooth,loglik = runKF(y, A, C, Q, R, Z_0, V_0)

    A_new   = A.copy()
    Q_new   = Q.copy()
    V_0_new = V_0.copy()

    for i in range(num_blocks):
        r_i      = r[0,i].copy()
        rp       = r_i*p
        rp1      = np.sum(r[0,:i]) *ppC
        b_subset = np.arange(rp1,rp1+rp)
        t_start  = rp1
        t_end    = (rp1+r_i*ppC)

        EZZ    = np.matmul(Zsmooth[b_subset,1:],Zsmooth[b_subset,1:].T) + \
                   np.sum(Vsmooth[1:,[b_subset],[b_subset]], axis =0)

        EZZ_BB = np.matmul(Zsmooth[b_subset,:-1],Zsmooth[b_subset,:-1].T) + \
                   np.sum(Vsmooth[:-1,[b_subset],[b_subset]],axis =0)

        EZZ_FB = np.matmul(Zsmooth[b_subset,1:],Zsmooth[b_subset,:-1].T) + \
                    np.sum(VVsmooth[:,b_subset,b_subset], axis = 0)

        A_i = A[t_start:t_end, t_start:t_end].copy()
        Q_i = Q[t_start:t_end, t_start:t_end].copy()

        A_i[:r_i,:rp] = np.matmul(EZZ_FB[:r_i,:rp],np.linalg.inv(EZZ_BB[:rp,:rp]))

        Q_i[:r_i,:r_i] = (EZZ[:r_i,:r_i] - np.matmul(A_i[:r_i,:rp],EZZ_FB[:r_i,:rp].T))/T

        A_new[t_start:t_end, t_start:t_end]   = A_i.copy()
        Q_new[t_start:t_end, t_start:t_end]   = Q_i.copy()
        V_0_new[t_start:t_end, t_start:t_end] = Vsmooth[0,t_start:t_end,t_start:t_end].copy()

    rp1      = np.sum(r)*ppC
    niM      = np.sum(i_idio[:nM])
    t_start  = rp1
    i_subset = np.arange(t_start,rp1+niM)

    EZZ = np.diag(np.diag(np.matmul(Zsmooth[t_start:,1:],Zsmooth[t_start:,1:].T))) + \
            np.diag(np.diag(np.sum(Vsmooth[1:,t_start:,t_start:], axis = 0)))

    EZZ_BB = np.diag(np.diag(np.matmul(Zsmooth[t_start:,:-1],Zsmooth[t_start:,:-1].T))) + \
                np.diag(np.diag(np.sum(Vsmooth[:-1,t_start:,t_start:], axis =0)))

    EZZ_FB = np.diag(np.diag(np.matmul(Zsmooth[t_start:,1:], Zsmooth[t_start:,:-1].T))) + \
                np.diag(np.diag(np.sum(VVsmooth[:,t_start:,t_start:], axis = 0)))

    A_i = np.matmul(EZZ_FB,np.diag(1/np.diag((EZZ_BB))))
    Q_i = (EZZ - np.matmul(A_i,EZZ_FB.T))/T

    A_new[np.ix_(i_subset,i_subset)]   = A_i[:niM,:niM].copy()
    Q_new[np.ix_(i_subset,i_subset)]   = Q_i[:niM,:niM].copy()
    V_0_new[np.ix_(i_subset,i_subset)] = np.diag(np.diag(Vsmooth[0,i_subset,i_subset].copy()))

    Z_0 = Zsmooth[:,[0]].copy()

    y              = y.copy()
    nanY           = np.isnan(y).astype(np.int64)
    y[np.isnan(y)] = 0

    C_new = C.copy()

    bl   = np.unique(blocks,axis =0)
    n_bl = bl.shape[0]

    for i in range(num_blocks):
        if i == 0:
            bl_idxQ = np.tile(bl[:,[i]],(1,r[0,i]*ppC))
            bl_idxM = np.hstack([np.tile(bl[:,[i]],(1,r[0,i])),np.zeros((n_bl,r[0,i]*(ppC-1)))])
            R_con   = np.kron(R_mat,np.eye(r[0,i]))
            q_con   = np.zeros((r[0,i]*R_mat.shape[0],1))
        else:
            bl_idxQ = np.hstack([bl_idxQ, np.tile(bl[:,[i]],(1,r[0,i]*ppC))])
            bl_idxM = np.hstack([np.hstack([bl_idxM, np.tile(bl[:,[i]],(1,r[0,i]))]),np.zeros((n_bl,r[0,i]*(ppC-1)))])
            R_con   = block_diag(R_con,np.kron(R_mat,np.eye(r[0,i])))
            q_con   = np.vstack([q_con,np.zeros((r[0,i]*R_mat.shape[0],1))])

    bl_idxM = bl_idxM == 1
    bl_idxQ = bl_idxQ == 1

    i_idio_M = i_idio[:nM].copy()
    n_idio_M = np.where(i_idio_M)[0].shape[0]
    c_i_idio = np.cumsum(i_idio)

    for i in range(n_bl):
        bl_i   = bl[[i],:].copy()
        rs     = np.sum(r[np.where(bl_i == 1)])
        idx_i  = np.where((blocks == bl_i).all(axis =1))[0]
        idx_iM = idx_i[idx_i < nM]
        n_i    = len(idx_iM)

        denom = np.zeros((n_i*rs,n_i*rs))
        nom   = np.zeros((n_i,rs))

        i_idio_i  = i_idio_M[idx_iM,:].flatten('F').copy()
        i_idio_ii = c_i_idio[idx_iM].copy()
        i_idio_ii = i_idio_ii[i_idio_i].copy() - 1

        bl_idxM_ind = np.where(bl_idxM[i, :])[0]

        for t in range(T):
            Wt          = np.diag(np.logical_not(nanY[idx_iM,t]).astype(np.int64))
            
            denom += np.kron(np.matmul(Zsmooth[bl_idxM_ind][:,[t+1]], Zsmooth[bl_idxM_ind][:,[t+1]].T) + \
                                        Vsmooth[t+1][np.ix_(bl_idxM_ind,bl_idxM_ind)],
                                    Wt)
            nom += np.matmul(y[idx_iM][:,[t]],Zsmooth[bl_idxM_ind][:,[t+1]].T) - \
                   np.matmul(Wt[:,i_idio_i],
                                    np.matmul(Zsmooth[rp1+i_idio_ii][:,[t+1]],Zsmooth[bl_idxM_ind][:,[t+1]].T) + \
                                    Vsmooth[t+1][rp1+i_idio_ii,:][:,bl_idxM_ind])

        # POSSIBLE WEAK POINT FOUND: NEED TO TEST ON INDEXING AS NUMPY DOES NOT MAINTAIN PROPER MATRIX FORM DEPENDING ON HOW ITS INDEXED: CHECK
        vec_C = np.matmul(np.linalg.inv(denom),nom.flatten('F').reshape((-1,1)))
        C_new[np.ix_(idx_iM,bl_idxM_ind)] = vec_C.copy().reshape((n_i,rs),order = "F") # CHECK: RESHAPE NEEDS TO BE VERIFIED

        idx_iQ = idx_i[idx_i >=nM].copy()
        rps = rs*ppC

        R_con_i = R_con[:,bl_idxQ[i,:]]
        q_con_i = q_con.copy()

        no_c = np.where(~(R_con_i.any(axis = 1)))[0]
        R_con_i = np.delete(R_con_i,no_c,axis = 0)
        q_con_i = np.delete(q_con_i, no_c, axis=0)

        for j in idx_iQ:

            denom = np.zeros((rps,rps))
            nom   = np.zeros((1,rps))

            idx_jQ = j - nM

            i_idio_jQ = np.arange(rp1 + n_idio_M + 5*(idx_jQ),rp1 + n_idio_M + 5*(idx_jQ+1))

            V_0_new[np.ix_(i_idio_jQ,i_idio_jQ)] = Vsmooth[0][np.ix_(i_idio_jQ,i_idio_jQ)].copy()
            A_new[i_idio_jQ[0],i_idio_jQ[0]]     = A_i[i_idio_jQ[0]-rp1,i_idio_jQ[0]-rp1].copy()
            Q_new[i_idio_jQ[0],i_idio_jQ[0]]     = Q_i[i_idio_jQ[0]-rp1,i_idio_jQ[0]-rp1].copy()

            bl_idxQ_ind = np.where(bl_idxQ[i,:])[0]
            
            for t in range(T):
                Wt = np.diag(np.logical_not(nanY[[j]][:,[t]]).astype(np.int64))

                denom += np.kron(np.matmul(Zsmooth[bl_idxQ_ind][:,[t+1]],Zsmooth[bl_idxQ_ind][:,[t+1]].T) + \
                                 Vsmooth[t+1][np.ix_(bl_idxQ_ind,bl_idxQ_ind)],
                                 Wt)
                nom += y[j,t]*Zsmooth[bl_idxQ_ind,t+1].T
                nom -= np.matmul(Wt,np.matmul(np.matmul(np.array([[1,2,3,2,1]]), Zsmooth[i_idio_jQ][:,[t+1]]),
                                              Zsmooth[bl_idxQ_ind][:,[t+1]].T) + \
                                    np.matmul(np.array([[1,2,3,2,1]]),Vsmooth[t+1][np.ix_(i_idio_jQ,bl_idxQ_ind)]))

            C_i = np.matmul(np.linalg.inv(denom),nom.T)
            C_i_constr = C_i - np.matmul(np.matmul(np.matmul(np.linalg.inv(denom),R_con_i.T),
                                                   np.linalg.inv(np.matmul(np.matmul(R_con_i,np.linalg.inv(denom)),R_con_i.T))),
                                         np.matmul(R_con_i,C_i)-q_con_i)

            C_new[j,bl_idxQ_ind] = C_i_constr.flatten('F')

    R_new = np.zeros((n,n))

    for t in range(T):
        Wt = np.diag(np.logical_not(nanY[:,t])).astype(np.int64)

        R_new += np.matmul(y[:,[t]] - np.matmul(np.matmul(Wt,C_new),Zsmooth[:,[t+1]]),
                           (y[:,[t]] - np.matmul(np.matmul(Wt,C_new),Zsmooth[:,[t+1]])).T) + \
                 np.matmul(np.matmul(np.matmul(np.matmul(Wt,C_new),Vsmooth[t+1][:,:]),C_new.T),Wt) + \
                 np.matmul(np.matmul((np.eye(n) - Wt),R),(np.eye(n)-Wt))

    i_idio_M = np.where(i_idio_M.flatten('F'))[0]
    R_new        = R_new/T
    RR           = np.diag(R_new).copy()
    RR[i_idio_M] = 1e-4
    RR[nM:]      = 1e-4
    R_new        = np.diag(RR).copy()
    # CHECK: np.diag to ensure no read only and
    return C_new, R_new, A_new, Q_new, Z_0, V_0, loglik

def em_converged(loglik, previous_loglik, threshold = 1e-4, check_decreased = 1):
    converged = 0
    decrease  = 0

    if check_decreased == 1:
        if (loglik - previous_loglik) < -1e-3:
            print('******likelihood decreased from {} to {}').format(previous_loglik,loglik)
            decrease = 1

    delta_loglik = np.abs(loglik - previous_loglik)
    avg_loglik   = (np.abs(loglik) + np.abs(previous_loglik) + np.finfo(float).eps)/2

    if (delta_loglik/avg_loglik) < threshold:
        converged = 1

    return converged, decrease

def runKF(Y,A,C,Q,R,Z_0,V_0):
    #runKF()    Applies Kalman filter and fixed-interval smoother
    #
    #  Syntax:
    #    [zsmooth, Vsmooth, VVsmooth, loglik] = runKF(Y, A, C, Q, R, Z_0, V_0)
    #
    #  Description:
    #    runKF() applies a Kalman filter and fixed-interval smoother. The
    #    script uses the following model:
    #           Y_t = C_t Z_t + e_t for e_t ~ N(0, R)
    #           Z_t = A Z_{t-1} + mu_t for mu_t ~ N(0, Q)

    #  Throughout this file:
    #    'm' denotes the number of elements in the state vector Z_t.
    #    'k' denotes the number of elements (observed variables) in Y_t.
    #    'nobs' denotes the number of time periods for which data are observed.
    #
    #  Input parameters:
    #    Y: k-by-nobs matrix of input data
    #    A: m-by-m transition matrix
    #    C: k-by-m observation matrix
    #    Q: m-by-m covariance matrix for transition equation residuals (mu_t)
    #    R: k-by-k covariance for observation matrix residuals (e_t)
    #    Z_0: 1-by-m vector, initial value of state
    #    V_0: m-by-m matrix, initial value of state covariance matrix
    #
    #  Output parameters:
    #    zsmooth: k-by-(nobs+1) matrix, smoothed factor estimates
    #             (i.e. zsmooth(:,t+1) = Z_t|T)
    #    Vsmooth: k-by-k-by-(nobs+1) array, smoothed factor covariance matrices
    #             (i.e. Vsmooth(:,:,t+1) = Cov(Z_t|T))
    #    VVsmooth: k-by-k-by-nobs array, lag 1 factor covariance matrices
    #              (i.e. Cov(Z_t,Z_t-1|T))
    #    loglik: scalar, log-likelihood
    #
    #  References:
    #  - QuantEcon's "A First Look at the Kalman Filter"
    #  - Adapted from replication files for:
    #    "Nowcasting", 2010, (by Marta Banbura, Domenico Giannone and Lucrezia
    #    Reichlin), in Michael P. Clements and David F. Hendry, editors, Oxford
    #    Handbook on Economic Forecasting.
    #
    # The software can be freely used in applications.
    # Users are kindly requested to add acknowledgements to published work and
    # to cite the above reference in any resulting publications

    S = SKF(Y, A, C, Q, R, Z_0, V_0)  # Kalman filter
    S = FIS(A, S)                     # Fixed interval smoother

    # Organize output
    zsmooth  = S["ZmT"].copy()
    Vsmooth  = S["VmT"].copy()
    VVsmooth = S["VmT_1"].copy()
    loglik   = S["loglik"].copy()

    return zsmooth,Vsmooth,VVsmooth,loglik

def SKF(Y, A, C, Q, R, Z_0, V_0):
    # SKF    Applies Kalman filter
    #
    #  Syntax:
    #    S = SKF(Y, A, C, Q, R, Z_0, V_0)
    #
    #  Description:
    #    SKF() applies the Kalman filter

    #  Input parameters:
    #    Y: k-by-nobs matrix of input data
    #    A: m-by-m transition matrix
    #    C: k-by-m observation matrix
    #    Q: m-by-m covariance matrix for transition equation residuals (mu_t)
    #    R: k-by-k covariance for observation matrix residuals (e_t)
    #    Z_0: 1-by-m vector, initial value of state
    #    V_0: m-by-m matrix, initial value of state covariance matrix
    #
    #  Output parameters:
    #    S.Zm: m-by-nobs matrix, prior/predicted factor state vector
    #          (S.Zm(:,t) = Z_t|t-1)
    #    S.ZmU: m-by-(nobs+1) matrix, posterior/updated state vector
    #           (S.Zm(t+1) = Z_t|t)
    #    S.Vm: m-by-m-by-nobs array, prior/predicted covariance of factor
    #          state vector (S.Vm(:,:,t) = V_t|t-1)
    #    S.VmU: m-by-m-by-(nobs+1) array, posterior/updated covariance of
    #           factor state vector (S.VmU(:,:,t+1) = V_t|t)
    #    S.loglik: scalar, value of likelihood function
    #    S.k_t: k-by-m Kalman gain

    m    = C.shape[1]
    nobs = Y.shape[1]
    S    = {}

    S["Zm"]     = np.zeros((m,nobs))
    S["Vm"]     = np.zeros((nobs,m,m))
    S["ZmU"]    = np.zeros((m,nobs + 1))
    S["VmU"]    = np.zeros((nobs + 1,m,m))
    S["loglik"] = 0

    S["Zm"][:]  = np.nan
    S["Vm"][:]  = np.nan
    S["ZmU"][:] = np.nan
    S["VmU"][:] = np.nan

    Zu = Z_0.copy()
    Vu = V_0.copy()

    S["ZmU"][:,[0]] = Zu.copy()
    S["VmU"][0,:,:] = Vu.copy()

    for t in range(nobs):
        Z = np.matmul(A,Zu)

        V = np.matmul(np.matmul(A,Vu),A.T) + Q
        V = .5 * (V + V.T)

        Y_t,C_t,R_t,_ = MissData(Y[:,[t]],C,R)

        if Y_t.shape[0] == 0:
            Zu = Z.copy()
            Vu = V.copy()
        else:
            VC = np.matmul(V,C_t.T)
            iF = np.linalg.inv(np.matmul(C_t,VC) + R_t)

            VCF = np.matmul(VC,iF)

            innov = Y_t - np.matmul(C_t,Z)

            Zu = Z + np.matmul(VCF,innov)

            Vu = V - np.matmul(VCF,VC.T)
            Vu = .5 * (Vu + Vu.T)

            S["loglik"] = S["loglik"] + .5*(np.log(np.linalg.det(iF)) - np.matmul(np.matmul(innov.T,iF), innov))[0,0]

        S["Zm"][:,[t]]   = Z.copy()
        S["Vm"][[t],:,:] = V.copy()

        S["ZmU"][:,[t+1]]   = Zu.copy()
        S["VmU"][t+1,:,:] = Vu.copy()

    if Y_t.shape[0] == 0:
        S["k_t"] = np.zeros((m,m))
    else:
        S["k_t"] = np.matmul(VCF,C_t)

    return S

def FIS(A,S):
    #FIS()    Applies fixed-interval smoother
    #
    #  Syntax:
    #    S = FIS(A, S)
    #
    #  Description:
    #    SKF() applies a fixed-interval smoother, and is used in conjunction
    #    with SKF(). See  page 154 of 'Forecasting, structural time series models
    #    and the Kalman filter' for more details (Harvey, 1990).
    #
    #  Input parameters:
    #    A: m-by-m transition matrix
    #    S: structure returned by SKF()
    #
    #  Output parameters:
    #    S: FIS() adds the following smoothed estimates to the S structure:
    #    - S.ZmT: m-by-(nobs+1) matrix, smoothed states
    #             (S.ZmT(:,t+1) = Z_t|T)
    #    - S.VmT: m-by-m-by-(nobs+1) array, smoothed factor covariance
    #             matrices (S.VmT(:,:,t+1) = V_t|T = Cov(Z_t|T))
    #    - S.VmT_1: m-by-m-by-nobs array, smoothed lag 1 factor covariance
    #               matrices (S.VmT_1(:,:,t) = Cov(Z_t Z_t-1|T))
    #
    #  Model:
    #   Y_t = C_t Z_t + e_t for e_t ~ N(0, R)
    #   Z_t = A Z_{t-1} + mu_t for mu_t ~ N(0, Q)

    m,nobs   = S["Zm"].shape
    S["ZmT"] = np.zeros((m,nobs+1))
    S["VmT"] = np.zeros((nobs+1,m,m))

    S["ZmT"][:,nobs]   = np.squeeze(S["ZmU"][:,nobs])
    S["VmT"][nobs,:,:] = np.squeeze(S["VmU"][nobs,:,:])

    VmT_1_init             = np.matmul(np.matmul(np.eye(m) - S["k_t"],A),np.squeeze(S["VmU"][nobs-1,:,:]))
    S["VmT_1"]             = np.zeros((nobs,VmT_1_init.shape[0],VmT_1_init.shape[1]))
    S["VmT_1"][nobs-1,:,:] = VmT_1_init

    J_2 = np.matmul(np.matmul(np.squeeze(S["VmU"][nobs-1,:,:]),A.T),np.linalg.pinv(np.squeeze(S["Vm"][nobs-1,:,:])))

    for t in range(nobs)[::-1]:
        VmU = np.squeeze(S["VmU"][t,:,:])
        Vm1 = np.squeeze(S["Vm"][t,:,:])

        V_T  = np.squeeze(S["VmT"][t+1,:,:])
        V_T1 = np.squeeze(S["VmT_1"][t,:,:])

        J_1 = J_2.copy()

        S["ZmT"][:,[t]] = S["ZmU"][:,[t]] + np.matmul(J_1,S["ZmT"][:,[t+1]] - np.matmul(A,S["ZmU"][:,[t]]))

        S["VmT"][t,:,:] = VmU + np.matmul(J_1,np.matmul((V_T - Vm1),J_1.T))

        if t>0:
            J_2  = np.matmul(np.matmul(np.squeeze(S["VmU"][t-1,:,:]),A.T),np.linalg.pinv(np.squeeze(S["Vm"][t-1,:,:])))
            S["VmT_1"][t-1,:,:] = np.matmul(VmU,J_2.T) + np.matmul(J_1,np.matmul(V_T1 - np.matmul(A,VmU),J_2.T))

    return S

def MissData(y,C,R):
    # Syntax:
    # Description:
    #   Eliminates the rows in y & matrices C, R that correspond to missing
    #   data (NaN) in y
    #
    # Input:
    #   y: Vector of observations at time t
    #   C: Observation matrix
    #   R: Covariance for observation matrix residuals
    #
    # Output:
    #   y: Vector of observations at time t (reduced)
    #   C: Observation matrix (reduced)
    #   R: Covariance for observation matrix residuals
    #   L: Used to restore standard dimensions(n x #) where # is the nr of
    #      available data in y

    # Returns 1 for nonmissing series
    ix = np.where(~np.isnan(y).flatten('F'))[0]

    # Index for columns with nonmissing variables
    e  = np.eye(y.shape[0])
    L  = e[:,ix].copy()

    # Removes missing series
    y  = y[ix].copy()

    # Removes missing series from observation matrix
    C  =  C[ix,:].copy()

    # Removes missing series from transition matrix
    R  =  R[np.ix_(ix,ix)].copy()

    return y,C,R,L