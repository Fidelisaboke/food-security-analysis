"""This module contains a function for cleaning the dataset."""

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split

FILE_PATH = "../data/dataFS.csv"

def clean_data(path=FILE_PATH):
    """Performs data cleaning on the dataFS dataset."""
    data = pd.read_csv(path)

    # Drop unneccessary columns
    data.drop(['ID','FS'], axis=1, inplace=True)

    # Ensure consistency in 'povline'
    data['povline'] = data['povline'].replace({'no_(2)': 'no'})

    # Fill missing 'hhedu' values with mode
    data['hhedu'] = data['hhedu'].fillna(data['hhedu'].mode()[0])

    # Ordinal Encoding
    ordinal_columns = ['hhedu', 'windex3', 'windex5']
    ordinal_encoder = OrdinalEncoder()
    data[ordinal_columns] = ordinal_encoder.fit_transform(data[ordinal_columns])

    # One-Hot Encoding
    categorical_columns = data.select_dtypes(include='object').columns
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    # Create 'dependency_ratio' feature
    data['dependency_ratio'] = data['u05'] / data['hhsize']

    return data
