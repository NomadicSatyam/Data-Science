import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy import stats

def apply_one_hot_encoding(df_fit, df_trans):
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore')

    df_ohe_fit = df_fit.select_dtypes(include='object')
    one_hot_encoder.fit(df_ohe_fit)

    df_ohe = df_trans.select_dtypes(include='object')

    feat1 = one_hot_encoder.transform(df_ohe).toarray()
    return feat1

def process_df_iqr(df):
    df_apply_iqr = df[['LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'SalePrice']]
    q1 = df_apply_iqr.quantile(0.25)
    q3 = df_apply_iqr.quantile(0.75)
    iqr = q3 - q1

    df_iqr = df[~((df < (q1 - 1.5 * iqr)) |(df > (q3 + 1.5 * iqr))).any(axis=1)]
    df_iqr = df_iqr.drop(df_iqr.columns[0], axis=1)
  
    feat1 = apply_one_hot_encoding(df, df_iqr)
    df_non_ohe = df_iqr.select_dtypes(exclude='object')
    feat2 = df_non_ohe.to_numpy()
    final_feat = np.concatenate([feat1, feat2], axis=1)

    return final_feat[:,0:-1], final_feat[:,-1]

def process_df_z(df):
    df_apply_z= df[['LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'SalePrice']]
    z = np.abs(stats.zscore(df_apply_z))
    df_z = df[(z<3).all(axis=1)]
    df_z = df_z.drop(df_z.columns[0], axis=1)

    feat1 = apply_one_hot_encoding(df, df_z)
    df_non_ohe_z = df_z.select_dtypes(exclude='object')
    feat2 = df_non_ohe_z.to_numpy()
    final_feat = np.concatenate([feat1, feat2], axis=1)

    return final_feat[:,0:-1], final_feat[:,-1]
    
    
