import os
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
        
        df = df.apply(pd.to_numeric, errors='coerce')

        print (df.head())
        print (df.info())
        #hadling missing values
        df = self.handle_missing_values(df)
        X = df.drop("target", axis=1)
        X= self.handle_encoding(X)
        return df, X
    def handle_missing_values(self, df):
        print("Before imputation:")
        print(df.isnull().sum())
        df = pd.DataFrame(
            self.imputer.fit_transform(df),
            columns=df.columns)
            
        print("After imputation:")
        print(df.isnull().sum())
        return df
    def handle_encoding(self, df):
        df = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
        return df

    def fit_transform(self, X):
        X = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)
        X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
        self.isfitted = True
        return X

    def transform(self, X):
        if not self.isfitted:
            raise ValueError("The imputer and scaler must be fitted before calling transform.")
        X = pd.DataFrame(self.imputer.transform(X), columns=X.columns)
        X = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
        return X
    
    #save the clean data to a file
    def save(self, path = "models/preprocessor.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"Preprocessor saved to {path}")

    def load(self, path = "models/preprocessor.pkl"):
        return joblib.load(path)


if __name__ == "__main__":
    preprocessor = preprocess_data()
    #load dataset
    df, X = preprocessor.load_data(url)

    #convert to binary classification
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

    #Split features and Target
    X= df.drop("target", axis=1)
    y= df['target']

    # Preprocess features 
    X_processed = preprocessor.fit_transform(X) 
    # Save cleaned dataset 
    processed_df = X_processed.copy() 
    processed_df["target"] = y.values 
    os.makedirs("data/processed", exist_ok=True) 
    processed_df.to_csv( "data/processed/heart_disease_cleaned.csv", index=False ) 
    print("\nCleaned dataset saved:") 
    print("data/processed/heart_disease_cleaned.csv") 
    # Save preprocessor 
    preprocessor.save() 
    print("Preprocessing completed and preprocessor saved.")