#-------------------------------------------------Libraries
import pandas as pd
import numpy as np
from datetime import datetime as dt


#-------------------------------------------------Functions
def summarize(X, Time, Spec):
    """
    summarize Display the detail table for data entering the DFM

    Description:
        Display the detail table for the nowcast, decomposing nowcast changes
        into news and impacts for released data series.
    """
    print("\n\n\n")
    print('Table 2: Data Summary \n')

    # Number of rows and data series
    T,N = X.shape

    print("N =     {} data series".format(N))
    print("T =     {} observations from {} to {} \n".format(T,
                                                         dt.fromordinal(Time[0] - 366).strftime('%Y-%m-%d'),
                                                         dt.fromordinal(Time[-1] - 366).strftime('%Y-%m-%d')))

    # create base table to add additional columns
    base = pd.DataFrame(X, columns=Spec.SeriesName)

    # Create additional columns
    time_range    = base.apply(lambda x: data_table_prep(1,x,Time = Time)).values
    max_min_range = base.apply(lambda x: data_table_prep(2,x,Time = Time)).values
    Frequency     = data_table_prep(3,freq_dict={'m':"Monthly", "q":"Quarterly"},Spec=Spec)
    Units         = data_table_prep(4, N=N,Spec=Spec)

    # Transform base dataframe
    base = base.describe().T
    base["Observation Time Range"] = time_range
    base["Min/Max Dates"] = max_min_range
    base["Frequency"] = Frequency
    base["Units"] = Units

    # Rename columns and set the order of columns
    base.rename(columns={"count": "Observations"}, inplace=True)
    orderColumns = ["Observations",
                    "Observation Time Range",
                    "Units",
                    "Frequency",
                    "mean",
                    "std",
                    "min",
                    "Min/Max Dates"]

    # Rename index values of dataframe
    base.rename(index = {base.index[i]:base.index[i] + " [{}]".format(Spec.SeriesID[i])
                         for i in range(N)},
                inplace = True)

    # Print dataframe
    print(base[orderColumns])

def data_table_prep(option,x=None,freq_dict=None,Time=None,N=None,Spec=None):
    """
    The param option takes values from 1 to 4

    Option 1:
        :param x: a pd.Series
        :param Time: an array containing ordinal values

        1.) Find values that are not NA
        2.) Do a cumsum
        3.) Find the first value that is not NA (will be 1)
        4.) Find the max value
        5.) First value and max value are the first and last date
        :return: a string containing the start and last date values that are not NA

    Option 2:
        :param x: a pd.Series
        :param Time: an array containing ordinal values

        1.) Get the index of the min and max values of x
        2.) Get the Time value in respect to the two indices
        :return: a string containing dates that represent min and max values

    Option 3:
        :param freq_dict: dictionary containing abbreviated as keys with values that are full names: e.g. "q" -> "Quarterly"
        1.) Convert the abbreviated form of frequency to full name
         :return: a string containing full name of the frequency

    Option 4:
        :param N: Number of data series
        :param Spec: Spec class object
        :return: returns a string representing unit transformed
    """

    if option == 1:
        truth   = (np.isnan(x) == False).cumsum()
        first   = Time[np.where(truth == 1)[0][0]]
        last    = Time[np.max(truth) - 1] # subtracted by 1 to get index value
        return "{} to {}".format(dt.fromordinal(first - 366).strftime('%b-%Y'),
                                dt.fromordinal(last - 366).strftime('%b-%Y'))
    elif option == 2:
        min = Time[np.where(x == np.min(x))[0][0]]
        max = Time[np.where(x == np.max(x))[0][0]]

        return "{} / {}".format(dt.fromordinal(min - 366).strftime('%b-%Y'),
                                dt.fromordinal(max - 366).strftime('%b-%Y'))

    elif option == 3:
        return [freq_dict[i] for i in Spec.Frequency]

    elif option == 4:
        return [unitTransformed(Spec.Units[i],
                                Spec.Transformation[i],
                                Spec.Frequency[i])
                for i in range(N)]

    else:
        ValueError("Option needed: must be 1 or 2")

def unitTransformed(unit,transform,freq):
    """
    :param unit: a data series' unit type
    :param transform: what transformation was applied to the data series
    :param freq: the frequency of the data series
    :return: returns a string representing unit transformed
    """
    if unit == "Index":
        unit_transformed = "Index"

    elif transform == "chg":
        if "%" in unit:
            unit_transformed = "Ppt. change"
        else:
            unit_transformed = "Level change"

    elif "pch" == transform and freq == "m":
        unit_transformed = "MoM %"
    elif "pca" == transform and freq == "q":
        unit_transformed = "QoQ AR %"
    else:
        unit_transformed = unit + " [{}]".format(transform)

    return unit_transformed