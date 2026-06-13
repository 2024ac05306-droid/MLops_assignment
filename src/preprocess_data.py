from tkinter.font import names

import numpy as np
import pandas as pd

import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer



#UCI Heart Disease dataset URL (Cleveland dataset)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

class preprocess_data:
    def __init__(self):
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()
        self.isfitted = False
        
    def load_data(self, url=str):
        column_names = [
        "age","sex","cp","trestbps","chol","fbs","restecg","thalach",
        "exang","oldpeak","slope","ca","thal","target"]
        df = pd.read_csv(url, header = None, names=column_names, na_values='?')
        print (df.head())
        print (df.info())
        #hadling missing values
        df = self.handle_missing_values(df)
        x = df.drop("target", axis=1)
        return df, x
    def handle_missing_values(self, df):
        print("Before imputation:")
        print(df.isnull().sum())
        df = pd.DataFrame(
            self.imputer.fit_transform(df),
            columns=df.columns)
            
        print("After imputation:")
        print(df.isnull().sum())
        return df

    def fit_transform(self, df):
        df_imputed = pd.DataFrame(self.imputer.fit_transform(df), columns=df.columns)
        df_imputed = pd.DataFrame(self.scaler.fit_transform(df_imputed), columns=df.columns)
        self.isfitted = True
        return df_imputed

    def transform(self, df):
        if not self.isfitted:
            raise ValueError("The imputer and scaler must be fitted before calling transform.")
        df_imputed = pd.DataFrame(self.imputer.transform(df), columns=df.columns)
        df_scaled = pd.DataFrame(self.scaler.transform(df_imputed), columns=df.columns)
        return df_scaled
    
    def save(self, path = "models/preprocessor.pkl"):
        joblib.dump(self, path)

    def load(self, path = "models/preprocessor.pkl"):
        return joblib.load(path)


_preprocessor = preprocess_data()
df, x = _preprocessor.load_data(url)
df_preprocessed = _preprocessor.fit_transform(df)
_preprocessor.save()
