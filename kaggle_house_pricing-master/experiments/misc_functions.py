import numpy as np
import pandas as pd
import re

def get_na_count(df):
    col_list = df.columns
    final_list = []
    for x in col_list:
        count = df[x].isnull().sum()
        if count > 0:
            final_list.append([x,count])
    return final_list 

if __name__ == '__main__':

    df = pd.read_csv('dataset/train.csv')
    df_test = pd.read_csv('dataset/test.csv')
    df.drop('Id', axis=1, inplace=True)
    id_array_test = df_test['Id'].values
    df_test.drop('Id', axis=1, inplace=True)
    y_train = df['SalePrice'].values
    df.drop('SalePrice', axis=1, inplace=True)

    df_cum = pd.concat([df, df_test])
    a = get_na_count(df_cum)
    print (a)
