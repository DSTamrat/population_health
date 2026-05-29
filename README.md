# CareFirst Population Health Analytics Dashboard

End-to-end pseudo-data demo of a CareFirst-style population health engine:
from ingestion and ETL to ML risk stratification and Salesforce Health Cloud payloads,
with a FACETS-style data inspector.

## 1. Features

- Synthetic member-level dataset (age, chronic conditions, SDOH, costs)
- Enterprise ETL: outlier handling, missing value imputation, comorbidity index
- RandomForest-based risk stratification with feature importance
- Salesforce Health Cloud–ready payload (custom object style)
- Multi-page Streamlit UI for senior leadership:
  - Executive Overview
  - Ingestion & ETL Pipeline
  - Predictive ML Stratification
  - Salesforce Health Cloud Sync
  - Google FACETS Data Inspector

## 2. Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
