def preprocessing(df, scaler):
    import numpy as np
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import StandardScaler
    categorical_features=[feature for feature in df.columns if df[feature].dtype=='object']
    numerical_features=[feature for feature in df.columns if feature not in categorical_features]
    train_=df.copy()
    train_.drop(['Id','GarageArea','TotalBsmtSF','TotRmsAbvGrd','GarageYrBlt'],axis=1,inplace=True)
    categorical_features=[feature for feature in train_.columns if train_[feature].dtype=='object']
    numerical_features=[feature for feature in train_.columns if feature not in categorical_features]
    df_num=train_[numerical_features]
    df_cat=train_[categorical_features]
    df_num['LotFrontage']=np.where(df_num['LotFrontage'].isnull()==True,df_num['LotFrontage'].median(),df_num['LotFrontage'])
    df_num['MasVnrArea']=np.where(df_num['MasVnrArea'].isnull()==True,0,df_num['MasVnrArea'])
    idx_1=df_num[df_num['LotFrontage']==313].index
    idx_2=df_num[df_num['SalePrice']>700000].index
    #we need to drop the following rows from both numerical dataframe df_num and categorical dataframe df_cat
    df_num.drop(idx_1,inplace=True)
    df_num.drop(idx_2,inplace=True)
    df_num.reset_index(drop=True,inplace=True)

    df_cat.drop(idx_1,inplace=True)
    df_cat.drop(idx_2,inplace=True)
    df_cat.reset_index(drop=True,inplace=True)
    #we will first separate the target variable and independent variables separately
    X=df_num.drop('SalePrice',axis=1) # X for now has numerical features only
    y=df_num['SalePrice']
    imputer_num=SimpleImputer(strategy='median')
    X_=pd.DataFrame(imputer_num.fit_transform(X),columns=X.columns)
    scaler_= scaler
    X_=pd.DataFrame(scaler_.transform(X_),columns=X_.columns) #changed. scaler is argument. X_ is preprocessed num columns
    df_cat.drop('PoolQC',axis=1,inplace=True)
    df_cat.drop(['Heating'],axis=1,inplace=True)
    df_cat['MiscFeature']=np.where(df_cat['MiscFeature'].isnull()==True,'No_misc',df_cat['MiscFeature'])
    df_cat.head()
    df_cat['Alley']=np.where(df_cat['Alley'].isnull()==True,'No_alley',df_cat['Alley'])
    df_cat['Fence']=np.where(df_cat['Fence'].isnull()==True,'No_fence',df_cat['Fence'])
    df_cat['MasVnrType']=np.where(df_cat['MasVnrType'].isnull()==True,'No_venner',df_cat['MasVnrType'])
    df_cat['FireplaceQu']=np.where(df_cat['FireplaceQu'].isnull()==True,'No_fireplace',df_cat['FireplaceQu'])
    df_cat.drop('GarageCond',axis=1,inplace=True)
    df_cat['GarageType']=np.where(df_cat['GarageType'].isnull()==True,'No_garage',df_cat['GarageType'])
    df_cat['GarageFinish']=np.where(df_cat['GarageFinish'].isnull()==True,'No_garage',df_cat['GarageFinish'])
    df_cat['GarageQual']=np.where(df_cat['GarageQual'].isnull()==True,'No_garage',df_cat['GarageQual'])
    df_cat.drop(['BsmtCond','BsmtFinType2'],axis=1,inplace=True)
    df_cat['BsmtQual']=np.where(df_cat['BsmtQual'].isnull()==True,'no_bsmt',df_cat['BsmtQual'])
    df_cat['BsmtFinType1']=np.where(df_cat['BsmtFinType1'].isnull()==True,'no_bsmt',df_cat['BsmtFinType1'])
    df_cat['BsmtExposure']=np.where(df_cat['BsmtExposure'].isnull()==True,'no_bsmt',df_cat['BsmtExposure'])
    df_cat['Electrical']=np.where(df_cat['Electrical'].isnull()==True,df_cat['Electrical'].mode()[0],df_cat['Electrical'])
    imputer_cat=SimpleImputer(strategy='most_frequent')
    df_cat_=pd.DataFrame(imputer_cat.fit_transform(df_cat),columns=df_cat.columns)
    ordinal_features=['ExterQual','ExterCond','BsmtQual','BsmtExposure','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageFinish','CentralAir']
    nominal_features=[feature for feature in df_cat.columns if feature not in ordinal_features]

    df_cat_['ExterQual']=df_cat_['ExterQual'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1})
    df_cat_['ExterCond']=df_cat_['ExterCond'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1})
    df_cat_['BsmtQual']=df_cat_['BsmtQual'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'no_bsmt':0})
    df_cat_['BsmtExposure']=df_cat_['BsmtExposure'].map({'Gd':4,'Av':3,'Mn':2,'No':1,'no_bsmt':0})
    df_cat_['HeatingQC']=df_cat_['HeatingQC'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1})
    df_cat_['KitchenQual']=df_cat_['KitchenQual'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1})
    df_cat_['FireplaceQu']=df_cat_['FireplaceQu'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'No_fireplace':0})
    df_cat_['GarageQual']=df_cat_['GarageQual'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'No_garage':0})
    df_cat_['GarageFinish']=df_cat_['GarageFinish'].map({'Fin':3,'RFn':2,'Unf':1,'No_garage':0})
    df_cat_['CentralAir']=df_cat_['CentralAir'].map({'Y':1,'N':0})

    df_cat_nominal=df_cat_[nominal_features]
    df_cat_ordinal=df_cat_[ordinal_features]

    # we will use one hot encoding technique for nominal features
    encoder_=OneHotEncoder(handle_unknown='ignore')
    df_cat_nominal=pd.DataFrame(encoder_.fit_transform(df_cat_nominal).toarray())
    df_cat_nominal.columns=df_cat_nominal.columns.astype(str)
    data=pd.concat([X_,df_cat_ordinal,df_cat_nominal],axis=1)
    return data, y, encoder_

def input_preprocessor(df,
                      scaler,
                      encoder):
    import pandas as pd
    # Separating numerical and categorical columns
    categorical_features=[feature for feature in df.columns if df[feature].dtype=='object']
    numerical_features=[feature for feature in df.columns if feature not in categorical_features]
    df_num=df[numerical_features]
    df_cat=df[categorical_features]

    # Preprocessing numerical columns
    df_num=pd.DataFrame(scaler.transform(df_num),columns=df_num.columns)
    
    # Separating ordinal and nominal features
    ordinal_features=['ExterQual','ExterCond','BsmtQual','BsmtExposure','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageFinish','CentralAir']
    nominal_features=[feature for feature in df_cat.columns if feature not in ordinal_features]
    df_cat_nominal=df_cat[nominal_features]
    df_cat_ordinal=df_cat[ordinal_features]

    # Preprocessing nominal columns
    df_cat_nominal=pd.DataFrame(encoder.transform(df_cat_nominal).toarray())
    df_cat_nominal.columns=df_cat_nominal.columns.astype(str)

    # Preprocessing ordinal columns
    df_cat_ordinal['ExterQual']=df_cat_ordinal['ExterQual'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1})
    df_cat_ordinal['ExterCond']=df_cat_ordinal['ExterCond'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1})
    df_cat_ordinal['BsmtQual']=df_cat_ordinal['BsmtQual'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'no_bsmt':0})
    df_cat_ordinal['BsmtExposure']=df_cat_ordinal['BsmtExposure'].map({'Gd':4,'Av':3,'Mn':2,'No':1,'no_bsmt':0})
    df_cat_ordinal['HeatingQC']=df_cat_ordinal['HeatingQC'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1})
    df_cat_ordinal['KitchenQual']=df_cat_ordinal['KitchenQual'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1})
    df_cat_ordinal['FireplaceQu']=df_cat_ordinal['FireplaceQu'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'No_fireplace':0})
    df_cat_ordinal['GarageQual']=df_cat_ordinal['GarageQual'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'No_garage':0})
    df_cat_ordinal['GarageFinish']=df_cat_ordinal['GarageFinish'].map({'Fin':3,'RFn':2,'Unf':1,'No_garage':0})
    df_cat_ordinal['CentralAir']=df_cat_ordinal['CentralAir'].map({'Y':1,'N':0})
    
    # Making the final dataframe
    data=pd.concat([df_num,df_cat_ordinal,df_cat_nominal],axis=1)
    
    return data

def import_scaler():
    import joblib
    with open('./scaler.pkl', 'rb') as f:
        scaler = joblib.load(f)
    return scaler

def import_encoder():
    import pickle
    with open("onehotencoder", "rb") as f: 
        encoder = pickle.load(f)

    return encoder

def import_model():
    import xgboost
    # Load the model
    loaded_model = xgboost.XGBRegressor()
    loaded_model.load_model('xgb_model.json')
    return loaded_model