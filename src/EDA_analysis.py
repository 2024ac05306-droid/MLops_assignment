import os
import pandas as pd
import matplotlib.pyplot as plt
from pyparsing import col
import seaborn as sns

class EDA_analysis:
    def __init__(self, 
                 datapath = "data/processed/heart_disease_cleaned.csv",
                 output_dir = "outputs/eda"):
        self.datapath = datapath
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.df = pd.read_csv(datapath)
    def data_overview(self):
        print("\nDataset Shape")
        print(self.df.shape)

        print("\nDataset Information")
        print(self.df.info())

        print("\nStatistical Summary")
        print(self.df.describe())

        print("\nMissing Values")
        print(self.df.isnull().sum())

        print("\nClass Distribution")
        print(self.df['target'].value_counts())

        print("\nFirst 5 Rows")
        print(self.df.head())

    def plot_feature_distribute_histogram(self):
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            plt.figure(figsize=(6, 4), facecolor='w', edgecolor='k')
            sns.histplot(self.df[col], kde=True)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(self.output_dir, f'{col}_distribution.png'))
            plt.close()

    def plot_target_distr_or_classbalance(self):
        plt.figure(figsize=(6, 4), facecolor='w', edgecolor='k')
        sns.countplot(x='target', data=self.df, color='skyblue' )
        plt.title('Class Balance/Target Variable Distribution')
        plt.xlabel('Target Class')
        plt.ylabel('Count') 
        plt.savefig(os.path.join(self.output_dir, 'target_distribution.png'))   
        plt.close()
    

    def plot_boxplots(self):
        numeric_cols = self.df.select_dtypes(
             include=['float64', 'int64']).columns
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=self.df[0:5])  # Plot first 5 rows
        plt.title('Boxplots of Numeric Features')
        plt.xlabel('Features')
        plt.ylabel('Values')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'boxplots.png'))
        plt.close()

    def plot_correlation_heatmap(self):
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.df.corr()
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
        plt.title('Correlation Heatmap')
        plt.tight_layout()  # Adjust layout to prevent clipping of labels
        plt.savefig(os.path.join(
            self.output_dir, f'{col}_boxplot.png'))
        plt.close()

    def plot_correlation_heatmap(self):
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.df.corr()
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
        plt.title('Correlation Heatmap')
        plt.tight_layout()  # Adjust layout to prevent clipping of labels
        plt.xlabel('Features')  # Add x-axis label
        plt.ylabel('Features')  # Add y-axis label
        plt.savefig(os.path.join(self.output_dir, 'correlation_heatmap.png'))
        plt.close()

    def plot_target_correlation(self):
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = numeric_cols.drop('target')  # Exclude target variable
        correlations = self.df[numeric_cols].corrwith(self.df['target'])
        plt.figure(figsize=(10, 6), facecolor='w', edgecolor='k')
        sns.barplot(x=correlations.index, y=correlations.values, palette='viridis', hue=0.5 )
        plt.title('Correlation of Numeric Features with Target Variable')
        plt.xlabel('Features')
        plt.ylabel('Correlation Coefficient')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'target_correlation.png'))
        plt.close()


if __name__ == "__main__":
    eda = EDA_analysis()
    eda.data_overview()
    eda.plot_feature_distribute_histogram()
    eda.plot_target_distr_or_classbalance()
    eda.plot_boxplots()
    eda.plot_correlation_heatmap()
    eda.plot_target_correlation()
#observations from EDA
    with open(
    "outputs/eda/eda_summary.txt",
    "w"
) as f:
        f.write("EDA Observations\n")
        f.write("================\n\n")

        f.write("1. Class distribution is relatively balanced.\n")
        f.write("2. Cholesterol contains several outliers.\n")
        f.write("3. Chest pain type shows correlation with target.\n")
        f.write("4. Most features are approximately normally distributed.\n")