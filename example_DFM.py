#-------------------------------------------------Libraries
import os
from datetime import datetime as dt
from Functions.load_spec import load_spec
from Functions.load_data import load_data
from Functions.dfm import dfm
import pickle
from Functions.summarize import summarize
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np


#-------------------------------------------------Set dataframe to full view
pd.set_option('display.expand_frame_repr', False)


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

# Summarize dataset
summarize(X,Time,Spec)


#-------------------------------------------------Plot data
# Raw vs transformed
idxSeries = np.where(Spec.SeriesID == "INDPRO")[0][0]
t_obs     = ~np.isnan(X[:,idxSeries])

fig = make_subplots(rows=2, cols=1,
                    subplot_titles=("Raw Observed Data", "Transformed Data"))

fig.append_trace(go.Scatter(
    x=[dt.fromordinal(i - 366).strftime('%Y-%m-%d') for i in Time[t_obs]],
    y=Z[t_obs,idxSeries],
), row=1, col=1)

fig.append_trace(go.Scatter(
    x=[dt.fromordinal(i - 366).strftime('%Y-%m-%d') for i in Time[t_obs]],
    y=X[t_obs,idxSeries],
), row=2, col=1)


fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'} ,
                  title_text="Raw vs Transformed Data",
                  showlegend=False)
fig.update_yaxes(title_text=Spec.Units[idxSeries], row=1, col=1)
fig.update_yaxes(title_text=Spec.UnitsTransformed[idxSeries], row=2, col=1)
fig.show()


#-------------------------------------------------Run dynamic factor model (DFM) and save estimation output as 'ResDFM'.
threshold = 1e-4 # Set to 1e-5 for more robust estimates
Res = dfm(X,Spec,threshold)
Res = {"Res": Res,"Spec":Spec}

with open('ResDFM.pickle', 'wb') as handle:
    pickle.dump(Res, handle)
# TODO: Res and Spec should be separate, this will be fixed after the unit tests are created


#-------------------------------------------------Plot Loglik across number of steps
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(1,len(Res["Res"]["loglik"][1:])+1),
                         y=Res["Res"]["loglik"][1:],
                         mode='lines',
                         name="LogLik")
)
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'} ,
                  title_text="LogLik across number of steps taken",
                  showlegend=False
)
fig.update_yaxes(title_text="LogLik")
fig.update_xaxes(title_text="Number of steps")
fig.show()


#-------------------------------------------------Plot common factor and standardized data.
# select INDPRO data series
idxSeries = np.where(Spec.SeriesID == "INDPRO")[0][0]

# Create traces
fig = go.Figure()
for i in range(Res["Res"]["x_sm"].shape[1]):
    fig.add_trace(go.Scatter(x=[dt.fromordinal(i - 366).strftime('%Y-%m-%d') for i in Time],
                             y=Res["Res"]["x_sm"][:,i],
                             mode='lines',
                             name=Spec.SeriesID[i],
                             line={'width':.9})
)
fig.add_trace(go.Scatter(x=[dt.fromordinal(i - 366).strftime('%Y-%m-%d') for i in Time],
                         y=Res["Res"]["Z"][:,0]*Res["Res"]["C"][idxSeries,0],
                         mode='lines',
                         name="Common Factor",
                         line=dict(color='black', width=1.5))
)

# Plot common factor and standardized data
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'} ,
                  title_text="Common Factor and Standardized Data"
)
fig.show()


#-------------------------------------------------Plot projection of common factor onto Payroll Employment and GDP
# Two plots in one graph
fig = make_subplots(rows=2, cols=1,
                    subplot_titles=("Payroll Employment", "Real Gross Domestic Product"))

# Create an array of the data series that we are interested in looping through to plot the projection
series = ["PAYEMS","GDPC1"]

# For a particular series:
#       1.) plot the common factor
#       2.) plot the data series (with NAs removed)
for i in range(len(series)):

    idxSeries    = np.where(Spec.SeriesID == series[i])[0][0]
    t_obs        = ~np.isnan(X[:,idxSeries])

    CommonFactor = np.matmul(Res["Res"]["C"][idxSeries,:5].reshape(1,-1),Res["Res"]["Z"][:,:5].T) * \
                   Res["Res"]["Wx"][idxSeries] + Res["Res"]["Mx"][idxSeries]

    fig.append_trace(go.Scatter(
        x=[dt.fromordinal(i - 366).strftime('%Y-%m-%d') for i in Time],
        y=CommonFactor[0,:],
        name="Common Factor ({})".format(series[i])
    ), row=i+1, col=1)

    fig.append_trace(go.Scatter(
        x=[dt.fromordinal(i - 366).strftime('%Y-%m-%d') for i in Time[t_obs]],
        y=X[t_obs,idxSeries],
        name="Data ({})".format(series[i])
    ), row=i+1, col=1)

    fig.update_yaxes(title_text=Spec.Units[idxSeries] + " ({})".format(Spec.UnitsTransformed[idxSeries]), row=i+1, col=1)

fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'} ,
                  title_text="Projection of Common Factor")
fig.show()