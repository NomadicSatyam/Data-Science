import pandas as pd
import numpy as np
from process_data import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

df = pd.read_csv('dataset/train.csv')

#Handling NA for both numerical and categorical data.
cat_cols = (df.select_dtypes(include='object')).columns
df[cat_cols] = df[cat_cols].fillna('NA')

num_cols = (df.select_dtypes(exclude='object')).columns
df[num_cols] = df[num_cols].fillna(0)

df.drop(df.columns[0], axis=1, inplace=True)

#Convert categorical data using one hot encoding
feat1 = apply_one_hot_encoding(df, df)
df_non_ohe = df.select_dtypes(exclude='object')
feat2 = df_non_ohe.to_numpy()
final_feat = np.concatenate([feat1, feat2], axis=1)
X_train = final_feat[:,0:-1]
Y_train = final_feat[:,-1].reshape(-1,1)

#Scaling
standardScalerX = StandardScaler()
X_train = standardScalerX.fit_transform(X_train)

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

df_sub = pd.read_csv('dataset/test.csv')

#Handling NA for both numerical and categorical data.
cat_cols = (df_sub.select_dtypes(include='object')).columns
df_sub[cat_cols] = df_sub[cat_cols].fillna('NA')

num_cols = (df_sub.select_dtypes(exclude='object')).columns
df_sub[num_cols] = df_sub[num_cols].fillna(0)

id_array = df_sub.as_matrix(columns=['Id'])
df_sub.drop(df_sub.columns[0], axis=1, inplace=True)

#Convert categorical data using one hot encoding
feat1 = apply_one_hot_encoding(df, df_sub)

df_non_ohe = df_sub.select_dtypes(exclude='object')
feat2 = df_non_ohe.to_numpy()

final_feat = np.concatenate([feat1, feat2], axis=1)
X_test = final_feat

#Scaling
X_test = standardScalerX.transform(X_test)
y_pred = regressor.predict(X_test)

final_result = np.concatenate([id_array, y_pred], axis=1)

index_array = np.array([x for x in range(1459)])
submission_df = pd.DataFrame(data=final_result, index=index_array, columns=['Id', 'SalePrice'])
convert_dict = {'Id': int} 
  
submission_df = submission_df.astype(convert_dict) 

submission_df.to_csv('submission.csv',index=False)
