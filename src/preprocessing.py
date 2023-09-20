from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """
    # Print shape of input data
    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # 1. Correct outliers/anomalous values in numerical
    # columns (`DAYS_EMPLOYED` column).
    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    # 2. Encode string categorical features (dytpe `object`):
    #     - If the feature has 2 categories encode using binary encoding,
    #       please use `sklearn.preprocessing.OrdinalEncoder()`. Only 4 columns
    #       from the dataset should have 2 categories.
    #     - If it has more than 2 categories, use one-hot encoding, please use
    #       `sklearn.preprocessing.OneHotEncoder()`. 12 columns
    #       from the dataset should have more than 2 categories.
    #   Take into account that:
    #     - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #       working_test_df).
    #     - In order to prevent overfitting and avoid Data Leakage you must use only
    #       working_train_df DataFrame to fit the OrdinalEncoder and
    #       OneHotEncoder classes, then use the fitted models to transform all the
    #       datasets.

    # Initialize the encoders  
    ordinalencoder = OrdinalEncoder()
    onehotencoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    
    # Get the list for ordinal and onehotencoding encoding
    ordinal_columns=list()
    ohe_columns=list()
    for col in working_train_df.select_dtypes(include='object').columns:
        if working_train_df[col].nunique()==2:
            ordinal_columns.append(col)
        else:
            ohe_columns.append(col)
    
    # ORDINAL ENCODING    
    # Fit the encoding to the dataframe
    ordinalencoder.fit(working_train_df[ordinal_columns])
    # Apply the transform to the dataframes
    # (As it will modify only one column we apply the change directly to the working df's)
    working_train_df[ordinal_columns]=ordinalencoder.transform(working_train_df[ordinal_columns])
    working_test_df[ordinal_columns]=ordinalencoder.transform(working_test_df[ordinal_columns])
    working_val_df[ordinal_columns]=ordinalencoder.transform(working_val_df[ordinal_columns])

    # ONE HOT ENCODING 
    # Fit the encoding to the dataframe
    onehotencoder.fit(working_train_df[ohe_columns])
    # Apply the transform to create the arrays of encoded columns
    train_ohe_array=onehotencoder.transform(working_train_df[ohe_columns])
    test_ohe_array=onehotencoder.transform(working_test_df[ohe_columns])
    val_ohe_array=onehotencoder.transform(working_val_df[ohe_columns])
    
    # Transform the array to DF to join it to the working DF's and drop the original column
    working_train_df=working_train_df.drop(columns=ohe_columns).join(pd.DataFrame(train_ohe_array))
    working_test_df=working_test_df.drop(columns=ohe_columns).join(pd.DataFrame(test_ohe_array))
    working_val_df=working_val_df.drop(columns=ohe_columns).join(pd.DataFrame(val_ohe_array))
    print("Input train data shape: ", working_train_df.shape)
    print("Input val data shape: ", working_val_df.shape)
    print("Input test data shape: ", working_test_df.shape, "\n")
    
    # 3. Impute values for all columns with missing data or, just all the columns.
    #    Use median as imputing value. Please use sklearn.impute.SimpleImputer().
    #    Again, take into account that:
    #      - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #        working_test_df).
    #      - In order to prevent overfitting and avoid Data Leakage you must use only
    #        working_train_df DataFrame to fit the SimpleImputer and then use the fitted
    #        model to transform all the datasets.
    #    

    # Instanciate & fit
    median_imputer = SimpleImputer(strategy="median", missing_values=np.nan)
    median_imputer.fit(working_train_df.values)

    # Apply the transformations
    working_test_df=median_imputer.transform(working_test_df.values)
    working_train_df=median_imputer.transform(working_train_df.values)
    working_val_df=median_imputer.transform(working_val_df.values)

    
    # 4. Feature scaling with Min-Max scaler. Apply this to all the columns.
    #    Please use sklearn.preprocessing.MinMaxScaler().
    #    Again, take into account that:
    #      - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #        working_test_df).
    #      - In order to prevent overfitting and avoid Data Leakage you must use only
    #        working_train_df DataFrame to fit the MinMaxScaler and then use the fitted
    #        model to transform all the datasets.
    
    min_max=MinMaxScaler()
    min_max.fit(working_train_df)

    working_test_df=min_max.transform(working_test_df)
    working_train_df=min_max.transform(working_train_df)
    working_val_df=min_max.transform(working_val_df)

    
    return working_train_df,working_val_df,working_test_df
    
