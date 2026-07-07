import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
from mlflow.tracking import MlflowClient


class EDAAnalysis:
    def __init__(
        self,
        datapath="data/processed/heart_disease_cleaned.csv",
        output_dir="outputs/eda",
        target_column="target",
    ):
        self.datapath = datapath
        self.output_dir = output_dir
        self.target_column = target_column

        os.makedirs(self.output_dir, exist_ok=True)
        self.df = pd.read_csv(self.datapath)

    def data_overview(self):
        print("\nDataset Shape")
        print(self.df.shape)

        print("\nDataset Information")
        print(self.df.info())

        print("\nStatistical Summary")
        print(self.df.describe())

        print("\nMissing Values")
        print(self.df.isnull().sum())

        if self.target_column in self.df.columns:
            print("\nClass Distribution")
            print(self.df[self.target_column].value_counts())

        print("\nFirst 5 Rows")
        print(self.df.head())

        self.df.describe(include="all").to_csv(
            os.path.join(self.output_dir, "summary_statistics.csv")
        )
        self.df.isnull().sum().to_frame("missing_values").to_csv(
            os.path.join(self.output_dir, "missing_values.csv")
        )
        self.df.dtypes.astype(str).to_frame("dtype").to_csv(
            os.path.join(self.output_dir, "column_types.csv")
        )

    def plot_feature_distribution_histogram(self):
        numeric_cols = self.df.select_dtypes(include=["float64", "int64"]).columns

        for column in numeric_cols:
            plt.figure(figsize=(6, 4))
            sns.histplot(self.df[column], kde=True)
            plt.title(f"Distribution of {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"{column}_distribution.png"))
            plt.close()

    def plot_target_distribution_or_class_balance(self):
        if self.target_column not in self.df.columns:
            print(f"Target column '{self.target_column}' not found. Skipping target plot.")
            return

        plt.figure(figsize=(6, 4))
        sns.countplot(x=self.target_column, data=self.df, color="skyblue")
        plt.title("Class Balance / Target Variable Distribution")
        plt.xlabel("Target Class")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "target_distribution.png"))
        plt.close()

        self.df[self.target_column].value_counts().to_frame("count").to_csv(
            os.path.join(self.output_dir, "target_distribution.csv")
        )

    def plot_boxplots(self):
        numeric_cols = self.df.select_dtypes(include=["float64", "int64"]).columns

        plt.figure(figsize=(14, 8))
        sns.boxplot(data=self.df[numeric_cols])
        plt.title("Boxplots of Numeric Features")
        plt.xlabel("Features")
        plt.ylabel("Values")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "boxplots.png"))
        plt.close()

    def plot_correlation_heatmap(self):
        numeric_df = self.df.select_dtypes(include=["float64", "int64"])

        plt.figure(figsize=(12, 10))
        correlation_matrix = numeric_df.corr()
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.title("Correlation Heatmap")
        plt.xlabel("Features")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "correlation_heatmap.png"))
        plt.close()

    def plot_target_correlation(self):
        if self.target_column not in self.df.columns:
            print(f"Target column '{self.target_column}' not found. Skipping target correlation.")
            return

        numeric_cols = self.df.select_dtypes(include=["float64", "int64"]).columns
        numeric_cols = numeric_cols.drop(self.target_column, errors="ignore")

        correlations = self.df[numeric_cols].corrwith(self.df[self.target_column])

        plt.figure(figsize=(10, 6))
        sns.barplot(x=correlations.index, y=correlations.values)
        plt.title("Correlation of Numeric Features with Target Variable")
        plt.xlabel("Features")
        plt.ylabel("Correlation Coefficient")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "target_correlation.png"))
        plt.close()

    def save_eda_summary(self):
        summary_path = os.path.join(self.output_dir, "eda_summary.txt")

        with open(summary_path, "w", encoding="utf-8") as file:
            file.write("EDA Observations\n")
            file.write("================\n\n")
            file.write("1. Review target_distribution.png for class balance.\n")
            file.write("2. Review boxplots.png to identify possible outliers.\n")
            file.write("3. Review correlation_heatmap.png for feature relationships.\n")
            file.write("4. Review target_correlation.png for features related to the target.\n")

        print(f"EDA summary saved to: {summary_path}")

    def run_all(self):
        self.data_overview()
        self.plot_feature_distribution_histogram()
        self.plot_target_distribution_or_class_balance()
        self.plot_boxplots()
        self.plot_correlation_heatmap()
        self.plot_target_correlation()
        self.save_eda_summary()


def configure_mlflow(
    db_path="outputs/mlflow/eda_mlflow.db",
    artifact_dir="outputs/mlflow/eda_artifacts",
    experiment_name="heart_disease_eda",
):
    db_file = Path(db_path).resolve()
    artifact_root = Path(artifact_dir).resolve()

    db_file.parent.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    tracking_uri = f"sqlite:///{db_file.as_posix()}"
    artifact_location = artifact_root.as_uri()

    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        client.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location,
        )

    mlflow.set_experiment(experiment_name)

    print(f"EDA MLflow database: {db_file}")
    print(f"EDA MLflow artifacts: {artifact_root}")


def log_eda_to_mlflow(eda: EDAAnalysis):
    mlflow.log_param("data_path", eda.datapath)
    mlflow.log_param("output_dir", eda.output_dir)
    mlflow.log_param("target_column", eda.target_column)

    mlflow.log_metric("rows", eda.df.shape[0])
    mlflow.log_metric("columns", eda.df.shape[1])
    mlflow.log_metric("total_missing_values", int(eda.df.isnull().sum().sum()))

    if eda.target_column in eda.df.columns:
        class_counts = eda.df[eda.target_column].value_counts()
        for class_label, count in class_counts.items():
            mlflow.log_metric(f"class_{int(class_label)}_count", int(count))

    mlflow.log_artifacts(eda.output_dir, artifact_path="eda_outputs")

if __name__ == "__main__":
    configure_mlflow()

    with mlflow.start_run(run_name="eda_analysis"):
        eda = EDAAnalysis()
        eda.run_all()
        log_eda_to_mlflow(eda)

        print(f"MLflow run_id: {mlflow.active_run().info.run_id}")