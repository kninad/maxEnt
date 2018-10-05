import pandas as pd
import numpy as np 

'''
Perform all utility taks here:
- Data loading modules
- Writing clean csv files
- Removing missing values, extra columns, NaN etc.
- Functions to deal with specific files
'''


'''
Function to load a csv file and return it as numpy array
It is required that the data is binary. So convert any value that is > 0 to 1
'''
def load_data(filePath):
    df = pd.read_csv(filePath)
    tups = [tuple(x) for x in df.values]
    data_arr = np.asarray(tups)

    # Map all positive values to 1 since any > 0 indicates the disease
    data_arr[data_arr > 0] = 1
    return data_arr


'''
Function to clean and write the fy, sy and merged csv files from the 
2010-2014 csv file. Only the fy and sy disease prevalences are extracted from
the csv file. All other columns are ignored. This code is taken from the 
data_exploration ipython notebook.
''' 
def write_csv_files_2010_14():
    bigFilePath = '../data/2010-2014.csv'
    big_df = pd.read_csv(bigFilePath)
    col_list = big_df.columns
    # print col_list
    print "Total columns:",  len(col_list)

    print "Extracting relevant column numbers: "

    # first disease is 'CCCfy1'
    i = 0
    for i in range(len(col_list)):
        if col_list[i] == 'CCCfy1':
            print i
            break

    # last disease is 'CCCfy670'
    j = 0
    for j in range(len(col_list)):
        if col_list[j] == 'CCCfy670':
            print j
            break

    first_index = i
    fy_last_index = j
    total_first_year = fy_last_index - first_index + 1
    end_index = first_index + 2 * total_first_year 

    # sanity check: end_index should be the one just after
    # the last disease's sy column
    print first_index, fy_last_index, end_index
    print col_list[first_index], col_list[fy_last_index], col_list[end_index-1]

    # disease_list_2years is the list for all the columns for first year (fy)
    # disease prevalence in the dataset
    disease_list_fy = col_list[first_index:fy_last_index+1]
    print "Fy disease list:", disease_list_fy

    # disease_list_sy is the list for all the columns for second year (sy)
    # disease prevalence in the dataset
    disease_list_sy = col_list[fy_last_index+1:end_index]
    print "Sy disease list:", disease_list_sy


    # disease_list_merge is the list for all the columns for first and second year 
    # (fy and sy) disease prevalence in the dataset
    disease_list_merge = col_list[first_index:end_index]
    print "Merge disease list:", disease_list_merge


    df_fy = big_df.filter(disease_list_fy, axis=1)
    df_sy = big_df.filter(disease_list_sy, axis=1)
    df_merge = big_df.filter(disease_list_merge, axis=1)

    print "Printing the shape of the three data frames:"
    print df_fy.shape, df_sy.shape, df_merge.shape

    # Drop the rows who have NaN in certain columns
    df_fy = df_fy.dropna()
    df_sy = df_sy.dropna()
    df_merge = df_merge.dropna()

    print "Printing the shape of the three data frames after dropping the NaN rows:"
    print df_fy.shape, df_sy.shape, df_merge.shape


    print "Saving the cleaned data frames to csv files"
    fy_csv_file = '../data/2010-2014-fy.csv'
    sy_csv_file = '../data/2010-2014-sy.csv'
    merge_csv_file = '../data/2010-2014-merge.csv'

    df_fy.to_csv(fy_csv_file, encoding='utf-8', index=False)
    df_sy.to_csv(sy_csv_file, encoding='utf-8', index=False)
    df_merge.to_csv(merge_csv_file, encoding='utf-8', index=False)

    return


'''
Function to clean and write the fy, sy and merged csv files from the 
50Age toy dataset csv file. Only the fy and sy disease prevalences are extracted 
from the csv file. All other columns are ignored. This code is taken from the 
data_exploration ipython notebook.
''' 
def write_csv_files_toy_data():
    # Loading the toy-dataset
    filePath = '../data/Age50_DataExtract.csv'
    df = pd.read_csv(filePath)

    print df.columns

    drop_list = ['fyAGE', 'CCCfy98.1']
    df = df.drop(drop_list, axis=1)

    fy_list = col_list[:9]
    sy_list = col_list[9:]

    df_fy = df.filter(fy_list, axis=1)
    df_sy = df.filter(sy_list, axis=1)

    print "Saving the clean csv files"

    fname_merge = '../data/Age50_DataExtract_merge.csv'
    fname_fy = '../data/Age50_DataExtract_fy.csv'
    fname_sy = '../data/Age50_DataExtract_sy.csv'

    df.to_csv(fname_merge, encoding='utf-8', index=False)
    df_fy.to_csv(fname_fy, encoding='utf-8', index=False)
    df_sy.to_csv(fname_sy, encoding='utf-8', index=False)

    return