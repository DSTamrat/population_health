
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# 1. RAW INGESTION SIMULATION
# ---------------------------------------------------------
def generate_raw_ingestion_data(n_samples=1000):
    """
    Simulates raw ingestion from population health systems,
    clearinghouses, and CMS enrollment files.
    """
    np.random.seed(42)

    member_ids = [f"MEMBER_{10000 + i}" for i in range(n_samples)]
    ages = np.random.randint(18, 85, size=n_samples)

    # Chronic conditions
    diabetes = np.random.choice([0, 1], size=n_samples, p=[0.75, 0.25])
    hypertension = np.random.choice([0, 1], size=n_samples, p=[0.60, 0.40])
    ckd = np.random.choice([0, 1], size=n_samples, p=[0.90, 0.10])

    # Raw cost with outliers
    raw_claims_cost = np.random.exponential(scale=3000, size=n_samples)
    outlier_indices = np.random.choice(n_samples, size=20, replace=False)
    raw_claims_cost[outlier_indices] *= 12

    # SDOH with missing values
    sdoh_score = np.random.uniform(1.0, 5.0, size=n_samples)
    null_indices = np.random.choice(n_samples, size=50, replace=False)
    sdoh_score[null_indices] = np.nan

    # Target variable
    risk_prob = (
        (ages / 100) * 0.2
        + diabetes * 0.25
        + hypertension * 0.15
        + ckd * 0.35
        + (raw_claims_cost / 50000) * 0.1
    )
    risk_prob = np.clip(risk_prob, 0, 1)
    historical_enrollment = np.random.binomial(1, risk_prob)

    df_raw = pd.DataFrame({
        "Member_ID": member_ids,
        "Age": ages,
        "Ind_Diabetes": diabetes,
        "Ind_Hypertension": hypertension,
        "Ind_CKD": ckd,
        "Raw_Total_Cost": raw_claims_cost,
        "SDOH_Risk_Score": sdoh_score,
        "CMS_HEDIS_Enrolled": historical_enrollment,
    })

    return df_raw


# ---------------------------------------------------------
# 2. ETL PIPELINE
# ---------------------------------------------------------
def run_etl_pipeline(df_raw):
    """
    Cleans outliers, imputes missing values, and creates engineered features.
    """
    df_etl = df_raw.copy()

    # Impute missing SDOH
    median_sdoh = df_etl["SDOH_Risk_Score"].median()
    df_etl["SDOH_Risk_Score"] = df_etl["SDOH_Risk_Score"].fillna(median_sdoh)

    # Winsorize cost
    cap_value = df_etl["Raw_Total_Cost"].quantile(0.99)
    df_etl["Cleaned_Total_Cost"] = df_etl["Raw_Total_Cost"].clip(upper=cap_value)

    # Comorbidity index
    df_etl["Comorbidity_Count"] = (
        df_etl["Ind_Diabetes"]
        + df_etl["Ind_Hypertension"]
        + df_etl["Ind_CKD"]
    )

    return df_etl


# ---------------------------------------------------------
# 3. ML MODEL TRAINING
# ---------------------------------------------------------
def train_risk_stratification_model(df_etl):
    """
    Trains a RandomForest model and assigns risk tiers.
    """

    features = [
        "Age",
        "Ind_Diabetes",
        "Ind_Hypertension",
        "Ind_CKD",
        "Cleaned_Total_Cost",
        "SDOH_Risk_Score",
        "Comorbidity_Count",
    ]

    X = df_etl[features]
    y = df_etl["CMS_HEDIS_Enrolled"]

    # Correct parameter: test_size
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    df_etl["Predicted_Risk_Probability"] = model.predict_proba(X)[:, 1]

    df_etl["Risk_Tier"] = pd.cut(
        df_etl["Predicted_Risk_Probability"],
        bins=[-0.01, 0.3, 0.7, 1.01],
        labels=["Low Risk", "Moderate Risk", "High Risk"],
    )

    feature_imp_df = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    return df_etl, feature_imp_df


# ---------------------------------------------------------
# 4. SALESFORCE PAYLOAD GENERATION
# ---------------------------------------------------------
def generate_salesforce_payload(df_processed):
    """
    Creates Salesforce Health Cloud custom object payload.
    """
    sf_df = pd.DataFrame({
        "SF_Account_External_ID__c": df_processed["Member_ID"],
        "Risk_Score_Probability__c": df_processed["Predicted_Risk_Probability"].round(4),
        "Stratification_Tier__c": df_processed["Risk_Tier"],
        "Comorbidity_Index__c": df_processed["Comorbidity_Count"],
        "Total_Medical_Spend__c": df_processed["Cleaned_Total_Cost"].round(2),
        "Sync_Status__c": "Pending",
        "Action_Required__c": df_processed["Risk_Tier"].apply(
            lambda x: "Trigger Outreach Campaign"
            if x == "High Risk"
            else "Routine Check-In"
        ),
    })

    return sf_df
