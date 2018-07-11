import pandas as pd
import numpy as np 


# Perform all utility taks here
# Etc data loading, data cleaning etc.
# Removing missing values, extra columns

def load_data(filePath):
    df = pd.read_csv(filePath)
    df1 = df.drop(['fyAGE', 'CCCfy98.1'], axis=1) # erroneous column. CCCfy98 repeated in original csv file
    tups = [tuple(x) for x in df1.values]

    data_arr = np.asarray(tups)

    return data_arr





