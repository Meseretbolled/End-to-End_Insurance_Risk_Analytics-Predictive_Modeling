import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt
import os

def train_and_evaluate_models(file_path):
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    # --- Data Preparation ---
    print("Preparing data...")
    
    # Filter for claims > 0 for Severity Model
    df_claims = df[df['TotalClaims'] > 0].copy()
    
    # Feature Selection (Simplified for this example)
    features = ['VehicleType', 'RegistrationYear', 'Make', 'Model', 
                'Cylinders', 'Cubiccapacity', 'Kilowatts', 'Bodytype', 
                'NumberOfDoors', 'CustomValueEstimate', 'CapitalOutstanding', 
                'SumInsured', 'CalculatedPremiumPerTerm', 'TotalPremium',
                'Province', 'Gender', 'MaritalStatus', 'Age'] # Added Age if available or derive it
    
    # Let's check what columns we actually have from the generator
    available_features = [c for c in features if c in df_claims.columns]
    
    X = df_claims[available_features]
    y = df_claims['TotalClaims']
    
    # Handling Missing Data
    num_cols = X.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns
    
    imputer_num = SimpleImputer(strategy='mean')
    X[num_cols] = imputer_num.fit_transform(X[num_cols])
    
    imputer_cat = SimpleImputer(strategy='most_frequent')
    X[cat_cols] = imputer_cat.fit_transform(X[cat_cols])
    
    # Encoding Categorical Data
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = []
    
    # --- Modeling ---
    
    # 1. Linear Regression
    print("Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    r2_lr = r2_score(y_test, y_pred_lr)
    results.append(f"Linear Regression - RMSE: {rmse_lr:.2f}, R2: {r2_lr:.2f}")
    
    # 2. Random Forest
    print("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)
    results.append(f"Random Forest - RMSE: {rmse_rf:.2f}, R2: {r2_rf:.2f}")
    
    # 3. XGBoost
    print("Training XGBoost...")
    xgb = XGBRegressor(n_estimators=100, random_state=42)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    r2_xgb = r2_score(y_test, y_pred_xgb)
    results.append(f"XGBoost - RMSE: {rmse_xgb:.2f}, R2: {r2_xgb:.2f}")
    
    # --- Feature Importance (SHAP) ---
    print("Calculating SHAP values...")
    explainer = shap.Explainer(xgb)
    shap_values = explainer(X_test)
    
    # Save SHAP summary plot
    if not os.path.exists('notebooks/figures'):
        os.makedirs('notebooks/figures')
        
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig('notebooks/figures/shap_summary.png')
    plt.close()
    
    # Save results
    with open('notebooks/model_results.txt', 'w') as f:
        f.write("\n".join(results))
        
    print("\n".join(results))
    print("SHAP summary plot saved to notebooks/figures/shap_summary.png")

if __name__ == "__main__":
    train_and_evaluate_models('data/insurance_claims.csv')
