#-------------------------------------------------Libraries
import pandas as pd
import numpy as np
import re


#-------------------------------------------------load_spec class
class load_spec():

    # loadSpec Load model specification for a dynamic factor model (DFM)
    #
    # Description:
    #
    #   Load model specification  'Spec' from a Microsoft Excel workbook file
    #   given by 'filename'.
    #
    # Input Arguments:
    #
    #   filename -
    #
    # Output Arguments:
    #
    # spec - 1 x 1 structure with the following fields:
    #     . series_id
    #     . name
    #     . frequency
    #     . units
    #     . transformation
    #     . category
    #     . blocks
    #     . BlockNames

    """
    Python Version Notes:

    spec is a dictionary containing the fields:
        . series_id
        . name
        . frequency
        . units
        . transformation
        . category
        . blocks
        . BlockNames
    """


    def __init__(self,filename):

        # Find and drop series from Spec that are not in Model
        raw         = pd.read_excel(filename)
        raw.columns = [i.replace(" ","") for i in  raw.columns]
        raw         = raw[raw["Model"] == 1].reset_index(drop = True)

        # Sort all fields of 'Spec' in order of decreasing frequency
        frequency    = ['d','w','m','q','sa','a']
        permutations = []
        for freq in frequency:
            permutations+= list(raw[raw.Frequency == freq].index)
        raw = raw.loc[permutations,:]

        # Parse fields given by column names in Excel worksheet
        fldnms = ['SeriesID','SeriesName','Frequency','Units','Transformation','Category']
        for field in fldnms:
            if field in raw.columns:
                setattr(self,field,raw[field].to_numpy(copy=True))
            else:
                raise ValueError("{} raise ValueError(column missing from model specification.".format(field))

        # Parse blocks
        jColBlock             = list(raw.columns[raw.columns.str.contains("Block", case = False)])
        Blocks                = raw[jColBlock].copy()
        Blocks[Blocks.isna()] = 0

        if not (Blocks.iloc[:,0] == 1).all():
            raise ValueError("All variables must load on global block.")
        else:
            self.Blocks = Blocks.to_numpy(copy=True)
        self.BlockNames = [re.sub("Block[0-9]+-","",i) for i in jColBlock]

        # Transformations
        transformation = {'lin':'Levels (No Transformation)',
                          'chg':'Change (Difference)',
                          'ch1':'Year over Year Change (Difference)',
                          'pch':'Percent Change',
                          'pc1':'Year over Year Percent Change',
                          'pca':'Percent Change (Annual Rate)',
                          'cch':'Continuously Compounded Rate of Change',
                          'cca':'Continuously Compounded Annual Rate of Change',
                          'log':'Natural Log'}

        self.UnitsTransformed = np.array([transformation[i] for i in self.Transformation])

        # Summarize model specification
        print('\n Table 1: Model specification \n')
        print(pd.DataFrame({"SeriesID"         :self.SeriesID,
                            "SeriesName"       :self.SeriesName,
                            "Units"            :self.Units,
                            "UnitsTransformed" :self.UnitsTransformed}))