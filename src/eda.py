import pandas as pd
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def perform_eda(file_path: str) -> pd.DataFrame:
    """
    Perform Exploratory Data Analysis (EDA) on the insurance claims dataset.

    Args:
        file_path (str): Path to the CSV file containing the data.

    Returns:
        pd.DataFrame: The loaded dataframe with added metrics, or None if loading fails.
    """
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return None

    try:
        df = pd.read_csv(file_path)
        logging.info(f"Successfully loaded data from {file_path}")
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return None

    # Basic Data Inspection
    logging.info(f"Data Shape: {df.shape}")
    logging.info(f"Data Types:\n{df.dtypes}")
    
    missing_values = df.isnull().sum()
    if missing_values.any():
        logging.warning(f"Missing Values:\n{missing_values[missing_values > 0]}")
    else:
        logging.info("No missing values found.")

    logging.info(f"Descriptive Statistics:\n{df.describe()}")

    # Feature Engineering: Loss Ratio
    # Business Insight: Loss Ratio (Claims / Premium) is a key metric for profitability.
    # A high loss ratio indicates underpricing or high risk.
    try:
        df['LossRatio'] = df['TotalClaims'] / df['TotalPremium']
        avg_loss_ratio = df['LossRatio'].mean()
        logging.info(f"Overall Loss Ratio: {avg_loss_ratio:.2f}")
        
        # Business Insight: Analyzing Loss Ratio by Province helps identify high-risk regions.
        # ACIS can use this to adjust regional premiums.
        province_loss_ratio = df.groupby('Province')['LossRatio'].mean()
        logging.info(f"Loss Ratio by Province:\n{province_loss_ratio}")
        
    except KeyError as e:
        logging.error(f"Missing required columns for Loss Ratio calculation: {e}")
    except Exception as e:
        logging.error(f"Error calculating Loss Ratio: {e}")

    return df


if __name__ == "__main__":
    perform_eda('data/insurance_claims.csv')
