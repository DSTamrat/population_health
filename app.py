import streamlit as st
import plotly.express as px
import pandas as pd
import pipeline_utils as transform

# Page configuration
st.set_page_config(
    page_title="Population Health Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title { font-size:32px; font-weight:bold; color:#005EA2; margin-bottom:20px; }
    .sub-title { font-size:18px; color:#555555; margin-bottom:30px; }
    .card { background-color:#F0F4F8; padding:20px; border-radius:10px; margin-bottom:15px; }
    </style>
""",
    unsafe_allow_html=True,
)

# Cached pipeline
@st.cache_data
def run_entire_pipeline():
    raw_data = transform.generate_raw_ingestion_data(1200)
    etl_data = transform.run_etl_pipeline(raw_data)
    ml_data, feature_importance = transform.train_risk_stratification_model(etl_data)
    sf_payload = transform.generate_salesforce_payload(ml_data)
    return raw_data, etl_data, ml_data, feature_importance, sf_payload


raw_df, etl_df, ml_df, feat_imp, sf_df = run_entire_pipeline()

# Sidebar
st.sidebar.title("Navigation Hub")
page = st.sidebar.radio(
    "Go to Stage:",
    [
        "Executive Overview",
        "1. Ingestion & ETL Pipeline",
        "2. Predictive ML Stratification",
        "3. Salesforce Health Cloud Sync",
        "4. Google FACETS Data Inspector",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**System Status:** Connected to Data Lakehouse\n\n**Data Build:** 2026-Q2 Active"
)

# ==========================================
# EXECUTIVE OVERVIEW
# ==========================================
if page == "Executive Overview":
    st.markdown(
        "<div class='main-title'>Population Health Analytics Engine</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='sub-title'>End-to-End Machine Learning Stratification and CRM Operationalization Pipeline</div>",
        unsafe_allow_html=True,
    )

    # --- Images Section (Updated: no deprecated params) ---
    colA, colB, colC = st.columns(3)

    with colA:
        st.image(
            "https://img.icons8.com/fluency/240/data-configuration.png",
            caption="End-to-End Data Pipeline",
            width=240
        )

    with colB:
        st.image(
            "https://img.icons8.com/fluency/240/artificial-intelligence.png",
            caption="Predictive Modeling",
            width=240
        )

    with colC:
        st.image(
            "https://img.icons8.com/fluency/240/combo-chart.png",
            caption="Executive Dashboard Insights",
            width=240
        )

    # KPI Metrics
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Monitored Lives", f"{len(ml_df):,}", "Inbound Active")
    with kpi2:
        high_risk_count = len(ml_df[ml_df["Risk_Tier"] == "High Risk"])
        st.metric(
            "High-Risk Population",
            f"{high_risk_count}",
            f"{(high_risk_count/len(ml_df)*100):.1f}% of Pop",
            delta_color="inverse",
        )
    with kpi3:
        avg_cost = f"${ml_df['Cleaned_Total_Cost'].mean():,.2f}"
        st.metric("Average Cleaned Claim Cost", avg_cost)
    with kpi4:
        st.metric(
            "Salesforce Target Ready",
            f"{len(sf_df[sf_df['Action_Required__c']=='Trigger Outreach Campaign'])} Alerts",
        )

    # Pie Chart
    fig_pie = px.pie(
        ml_df,
        names="Risk_Tier",
        values="Cleaned_Total_Cost",
        title="Total Resource Allocation by Predictive Risk Tier",
        color_discrete_sequence=px.colors.sequential.RdBu_r,
        hole=0.4,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ==========================================
# INGESTION & ETL
# ==========================================
elif page == "1. Ingestion & ETL Pipeline":
    st.title("📥 Step 1: Ingestion & ETL Transformation")

    col_raw, col_etl = st.columns(2)

    with col_raw:
        st.subheader("Raw Data Feed")
        st.caption("Contains outliers and missing values.")
        st.dataframe(raw_df.head(10), use_container_width=True)

        fig_raw = px.box(raw_df, y="Raw_Total_Cost", title="Raw Cost Outliers")
        st.plotly_chart(fig_raw, use_container_width=True)

    with col_etl:
        st.subheader("Cleaned ETL Output")
        st.caption("Outliers capped, missing values imputed.")
        st.dataframe(etl_df.head(10), use_container_width=True)

        fig_etl = px.box(etl_df, y="Cleaned_Total_Cost", title="Cleaned Cost Profile")
        st.plotly_chart(fig_etl, use_container_width=True)

# ==========================================
# ML STRATIFICATION
# ==========================================
elif page == "2. Predictive ML Stratification":
    st.title("🤖 Step 2: Predictive Risk Stratification Engine")

    m1, m2 = st.columns([1, 2])

    with m1:
        st.subheader("Feature Importance")
        st.dataframe(feat_imp, use_container_width=True)

        fig_bar = px.bar(
            feat_imp,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Model Feature Importance",
            color="Importance",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with m2:
        st.subheader("Risk Mapping: Age vs Cost")
        fig_scatter = px.scatter(
            ml_df,
            x="Age",
            y="Cleaned_Total_Cost",
            color="Risk_Tier",
            size="Predicted_Risk_Probability",
            hover_data=["Member_ID", "Comorbidity_Count"],
            title="Risk Profile Distribution",
            color_discrete_map={
                "Low Risk": "#2ca02c",
                "Moderate Risk": "#ff7f0e",
                "High Risk": "#d62728",
            },
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# ==========================================
# SALESFORCE SYNC
# ==========================================
elif page == "3. Salesforce Health Cloud Sync":
    st.title("☁️ Step 3: Salesforce Health Cloud Payload")

    st.subheader("Staged Payload Records (PopulationHealth_Member_Risk__c)")
    st.dataframe(sf_df, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.info(
            f"**Action Summary:** {len(sf_df[sf_df['Action_Required__c']=='Trigger Outreach Campaign'])} members require outreach."
        )
    with c2:
        if st.button("🚀 Execute Salesforce Sync"):
            st.success("Payload successfully synced to Salesforce Health Cloud!")

# ==========================================
# FACETS INSPECTOR
# ==========================================
elif page == "4. Google FACETS Data Inspector":
    st.title("🔍 Step 4: FACETS-Style Feature Inspector")

    target_tier = st.selectbox(
        "Select Risk Tier:", ml_df["Risk_Tier"].unique()
    )
    filtered_df = ml_df[ml_df["Risk_Tier"] == target_tier]

    col1, col2, col3 = st.columns(3)

    with col1:
        fig_age = px.histogram(filtered_df, x="Age", nbins=15, title="Age Distribution")
        st.plotly_chart(fig_age, use_container_width=True)

    with col2:
        fig_cost = px.histogram(filtered_df, x="Cleaned_Total_Cost", nbins=15, title="Cost Distribution")
        st.plotly_chart(fig_cost, use_container_width=True)

    with col3:
        fig_como = px.histogram(filtered_df, x="Comorbidity_Count", title="Comorbidity Distribution")
        st.plotly_chart(fig_como, use_container_width=True)

    st.subheader("Raw Feature Table")
    st.dataframe(filtered_df, use_container_width=True)
