import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
PROCESSED_DATA_PATH = "data/processed/heart_disease_cleaned.csv"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
TARGET_COLUMN = "target"


class PreprocessData:
    def __init__(self):
        self.column_names = [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
            "target",
        ]
        self.numeric_columns = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        self.categorical_columns = [
            "sex",
            "cp",
            "fbs",
            "restecg",
            "exang",
            "slope",
            "ca",
            "thal",
        ]
        self.preprocessor = self.build_preprocessor()

    def build_preprocessor(self) -> ColumnTransformer:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, self.numeric_columns),
                ("categorical", categorical_pipeline, self.categorical_columns),
            ]
        )

    def load_data(self, data_path: str = DATA_URL) -> pd.DataFrame:
        data = pd.read_csv(
            data_path,
            header=None,
            names=self.column_names,
            na_values="?",
        )
        data = data.apply(pd.to_numeric, errors="coerce")
        data[TARGET_COLUMN] = data[TARGET_COLUMN].apply(lambda value: 1 if value > 0 else 0)
        return data

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        x = data.drop(columns=[TARGET_COLUMN])
        y = data[TARGET_COLUMN]

        x_processed = self.preprocessor.fit_transform(x)
        feature_names = self.preprocessor.get_feature_names_out()

        processed_data = pd.DataFrame(x_processed, columns=feature_names)
        processed_data[TARGET_COLUMN] = y.values

        return processed_data

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        x_processed = self.preprocessor.transform(data)
        feature_names = self.preprocessor.get_feature_names_out()
        return pd.DataFrame(x_processed, columns=feature_names)

    def save_processed_data(self, data: pd.DataFrame, output_path: str = PROCESSED_DATA_PATH) -> None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_file, index=False)
        print(f"Cleaned dataset saved to: {output_file}")

    def save_preprocessor(self, path: str = PREPROCESSOR_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.preprocessor, path)
        print(f"Preprocessor saved to: {path}")


def main():
    processor = PreprocessData()
    raw_data = processor.load_data()
    processed_data = processor.fit_transform(raw_data)

    processor.save_processed_data(processed_data)
    processor.save_preprocessor()

    print("Preprocessing completed.")
    print(f"Rows: {len(processed_data)}")
    print(f"Columns: {len(processed_data.columns)}")


if __name__ == "__main__":
    main()
