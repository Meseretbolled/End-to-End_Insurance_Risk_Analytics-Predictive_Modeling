import pandas as pd
from scipy import stats
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def perform_ab_testing(file_path: str):
    """
    Perform A/B testing (Hypothesis Testing) on the insurance claims dataset.
    
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

    # Test 1: Risk Differences across Provinces (ANOVA)
    # Null Hypothesis: There is no significant difference in TotalClaims across Provinces.
    try:
        provinces = df['Province'].unique()
        groups = [df[df['Province'] == p]['TotalClaims'] for p in provinces]
        f_stat, p_value = stats.f_oneway(*groups)
        
        logging.info(f"ANOVA Test for TotalClaims across Provinces: F-stat={f_stat:.4f}, p-value={p_value:.4f}")
        
        if p_value < 0.05:
            logging.info("Result: Reject Null Hypothesis. Significant difference in claims across provinces.")
            recommend_province_strategy(df)
        else:
            logging.info("Result: Fail to Reject Null Hypothesis. No significant difference in claims across provinces.")
            
    except Exception as e:
        logging.error(f"Error performing ANOVA test: {e}")

    # Test 2: Risk Differences between Gender (T-test)
    # Null Hypothesis: There is no significant difference in TotalClaims between Male and Female.
    try:
        male_claims = df[df['Gender'] == 'Male']['TotalClaims']
        female_claims = df[df['Gender'] == 'Female']['TotalClaims']
        
        t_stat, p_value = stats.ttest_ind(male_claims, female_claims, equal_var=False)
        
        logging.info(f"T-test for TotalClaims between Gender: T-stat={t_stat:.4f}, p-value={p_value:.4f}")
        
        if p_value < 0.05:
            logging.info("Result: Reject Null Hypothesis. Significant difference in claims between genders.")
            recommend_gender_strategy(df)
        else:
            logging.info("Result: Fail to Reject Null Hypothesis. No significant difference in claims between genders.")
            
    except Exception as e:
        logging.error(f"Error performing T-test: {e}")


def recommend_province_strategy(df: pd.DataFrame):
    """
    Generate business recommendations based on Province risk analysis.
    """
    avg_claims = df.groupby('Province')['TotalClaims'].mean().sort_values(ascending=False)
    high_risk = avg_claims.head(1).index[0]
    low_risk = avg_claims.tail(1).index[0]
    
    logging.info("--- Business Recommendation (Province) ---")
    logging.info(f"High Risk Province: {high_risk} (Avg Claim: {avg_claims[high_risk]:.2f})")
    logging.info(f"Action: Consider increasing premiums or stricter underwriting in {high_risk}.")
    logging.info(f"Low Risk Province: {low_risk} (Avg Claim: {avg_claims[low_risk]:.2f})")
    logging.info(f"Action: Target marketing campaigns in {low_risk} to acquire low-risk customers.")


def recommend_gender_strategy(df: pd.DataFrame):
    """
    Generate business recommendations based on Gender risk analysis.
    """
    avg_claims = df.groupby('Gender')['TotalClaims'].mean()
    logging.info("--- Business Recommendation (Gender) ---")
    logging.info(f"Average Claims by Gender:\n{avg_claims}")
    
    if avg_claims['Male'] > avg_claims['Female']:
        logging.info("Action: Males have higher average claims. Review pricing models for male drivers.")
    else:
        logging.info("Action: Females have higher average claims. Review pricing models for female drivers.")


if __name__ == "__main__":
    perform_ab_testing('data/insurance_claims.csv')
