import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_visualizations(file_path: str):
    """
    Generate production-ready visualizations for the insurance claims dataset.
    
    Args:
        file_path (str): Path to the CSV file containing the data.
    """
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return

    try:
        df = pd.read_csv(file_path)
        logging.info(f"Successfully loaded data from {file_path}")
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return

    output_dir = 'notebooks/figures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    try:
        # 1. Distribution of Total Premium
        plt.figure(figsize=(10, 6))
        sns.histplot(df['TotalPremium'], kde=True)
        plt.title('Distribution of Total Premium')
        plt.xlabel('Total Premium')
        plt.ylabel('Frequency')
        plt.savefig(f'{output_dir}/premium_distribution.png')
        plt.close()
        logging.info("Saved premium_distribution.png")

        # 2. Total Claims by Province
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Province', y='TotalClaims', data=df)
        plt.title('Total Claims by Province')
        plt.xlabel('Province')
        plt.ylabel('Total Claims')
        plt.savefig(f'{output_dir}/claims_by_province.png')
        plt.close()
        logging.info("Saved claims_by_province.png")

        # 3. Correlation Matrix
        plt.figure(figsize=(12, 8))
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.savefig(f'{output_dir}/correlation_matrix.png')
        plt.close()
        logging.info("Saved correlation_matrix.png")

        # 4. Bivariate Analysis: TotalPremium vs TotalClaims stratified by Province
        # Business Insight: This plot helps identify if higher premiums correlate with higher claims across different regions.
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='TotalPremium', y='TotalClaims', hue='Province', data=df, alpha=0.6)
        plt.title('Total Premium vs Total Claims by Province')
        plt.xlabel('Total Premium')
        plt.ylabel('Total Claims')
        plt.legend(title='Province')
        plt.savefig(f'{output_dir}/premium_vs_claims_by_province.png')
        plt.close()
        logging.info("Saved premium_vs_claims_by_province.png")

        # 5. Bivariate Analysis: TotalPremium vs TotalClaims stratified by PostalCode (ZipCode)
        # Note: Using top 10 PostalCodes to avoid clutter
        top_postal_codes = df['PostalCode'].value_counts().nlargest(10).index
        df_top_postal = df[df['PostalCode'].isin(top_postal_codes)]
        
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='TotalPremium', y='TotalClaims', hue='PostalCode', data=df_top_postal, palette='tab10', alpha=0.6)
        plt.title('Total Premium vs Total Claims by Top 10 Postal Codes')
        plt.xlabel('Total Premium')
        plt.ylabel('Total Claims')
        plt.legend(title='Postal Code')
        plt.savefig(f'{output_dir}/premium_vs_claims_by_postalcode.png')
        plt.close()
        logging.info("Saved premium_vs_claims_by_postalcode.png")

    except KeyError as e:
        logging.error(f"Missing column for visualization: {e}")
    except Exception as e:
        logging.error(f"Error generating visualizations: {e}")

    logging.info(f"All visualizations saved to {output_dir}/")


if __name__ == "__main__":
    create_visualizations('data/insurance_claims.csv')
