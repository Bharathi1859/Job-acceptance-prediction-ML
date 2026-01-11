# =====================================
# Job Acceptance Prediction Dashboard
# Author: Bharathi Jagadeesan
# =====================================

import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------------------
# Page Configuration
# -------------------------------------
st.set_page_config(
    page_title="Job Acceptance Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# -------------------------------------
# Load Data
# -------------------------------------
@st.cache_data
def load_data():
    # Change path if needed
    return pd.read_csv("cleaned_placement_data.csv")

df = load_data()

# -------------------------------------
# Title & Description
# -------------------------------------
st.title("📊 Job Acceptance Prediction System")
st.markdown(
    """
    **Business-focused analytics dashboard** to analyze placement outcomes,
    candidate performance, and future risk using data-driven insights.
    """
)

# =====================================
# SIDEBAR FILTERS (OPTIONAL)
# =====================================
st.sidebar.header("🔍 Filter Candidates")

company_tier = st.sidebar.selectbox(
    "Company Tier",
    ["All"] + sorted(df["company_tier"].dropna().unique().tolist())
)

experience_cat = st.sidebar.selectbox(
    "Experience Category",
    ["All"] + sorted(df["experience_category"].dropna().unique().tolist())
)

competition = st.sidebar.selectbox(
    "Competition Level",
    ["All"] + sorted(df["competition_level"].dropna().unique().tolist())
)

status_filter = st.sidebar.selectbox(
    "Placement Status",
    ["All", "placed", "not placed"]
)

# -------------------------------------
# Apply Filters Only If Selected
# -------------------------------------
filtered_df = df.copy()

if company_tier != "All":
    filtered_df = filtered_df[filtered_df["company_tier"] == company_tier]

if experience_cat != "All":
    filtered_df = filtered_df[filtered_df["experience_category"] == experience_cat]

if competition != "All":
    filtered_df = filtered_df[filtered_df["competition_level"] == competition]

if status_filter != "All":
    filtered_df = filtered_df[filtered_df["status"] == status_filter]

# =====================================
# KPI SECTION
# =====================================
st.subheader("📌 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

total_candidates = len(filtered_df)

placement_rate = (
    filtered_df["status"]
    .value_counts(normalize=True)
    .get("placed", 0) * 100
)

avg_interview = filtered_df["interview_avg_score"].mean()
avg_skills = filtered_df["skills_match_percentage"].mean()

high_risk_pct = (
    filtered_df["placement_probability_score"] < 60
).mean() * 100

acceptance_rate = (
    filtered_df[filtered_df["status"] == "placed"]
    .shape[0] / max(1, total_candidates) * 100
)

dropout_rate = 100 - acceptance_rate



col1.metric("👥 Total Candidates", total_candidates)
col2.metric("🎯 Placement Rate (%)", f"{placement_rate:.2f}")
col3.metric("🤝 Job Acceptance Rate (%)", f"{acceptance_rate:.2f}")
col4.metric("⚠️ Offer Dropout Rate (%)", f"{dropout_rate:.2f}")

col5, col6, col7 = st.columns(3)
col5.metric("🧠 Avg Interview Score", f"{avg_interview:.2f}")
col6.metric("🛠️ Avg Skills Match (%)", f"{avg_skills:.2f}")
col7.metric("🚨 High-Risk Candidates (%)", f"{high_risk_pct:.2f}")

# =====================================
# EDA & ML ANALYTICS
# =====================================
st.divider()
st.subheader("📈 Candidate Performance Analytics")

# -------------------------------------
# Interview Score vs Placement Probability
# -------------------------------------
fig1 = px.scatter(
    filtered_df,
    x="interview_avg_score",
    y="placement_probability_score",
    color="status",
    size="skills_match_percentage",
    hover_data=[
        "company_tier",
        "experience_category",
        "competition_level"
    ],
    title="Interview Performance vs Placement Probability"
)

st.plotly_chart(fig1, use_container_width=True)

# -------------------------------------
# Skills Match vs Placement Outcome
# -------------------------------------
fig2 = px.box(
    filtered_df,
    x="status",
    y="skills_match_percentage",
    color="status",
    title="Skills Match Percentage vs Placement Outcome"
)

st.plotly_chart(fig2, use_container_width=True)

# -------------------------------------
# Certification Impact
# -------------------------------------
cert_df = (
    filtered_df
    .groupby("certifications_count")["status"]
    .value_counts(normalize=True)
    .rename("placement_rate")
    .reset_index()
)

cert_df = cert_df[cert_df["status"] == "placed"]

fig3 = px.line(
    cert_df,
    x="certifications_count",
    y="placement_rate",
    markers=True,
    title="Certification Count Impact on Placement Probability"
)

st.plotly_chart(fig3, use_container_width=True)

# =====================================
# OPERATIONAL INSIGHTS
# =====================================
st.divider()
st.subheader("⚠️ Operational & Risk Insights")

# -------------------------------------
# Dropout Risk Distribution
# -------------------------------------
filtered_df["risk_category"] = pd.cut(
    filtered_df["placement_probability_score"],
    bins=[0, 0.4, 0.7, 1],
    labels=["High Risk", "Medium Risk", "Low Risk"]
)

fig4 = px.pie(
    filtered_df,
    names="risk_category",
    title="Candidate Dropout Risk Distribution"
)

st.plotly_chart(fig4, use_container_width=True)

# -------------------------------------
# Feature Importance (From Logistic Regression)
# -------------------------------------
feature_importance = pd.DataFrame({
    "Feature": df.select_dtypes(include="number").columns,
    "Importance": abs(df.select_dtypes(include="number").corr()["placement_probability_score"])
}).dropna().sort_values("Importance", ascending=False).head(10)

fig5 = px.bar(
    feature_importance,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top Factors Influencing Placement Probability"
)

st.plotly_chart(fig5, use_container_width=True)
import plotly.express as px

fig6 = px.histogram(
    df,
    x="placement_probability_score",
    nbins=25,
    title="Placement Probability Score Distribution"
)
st.plotly_chart(fig6, use_container_width=True)



# =====================================
# BUSINESS RECOMMENDATIONS
# =====================================
st.divider()
st.subheader("💡 Business Recommendations")

st.markdown(
    """
    **✔ Improve Interview Preparation:**  
    Interview performance has the strongest influence on placement success.

    **✔ Focus on Skill Alignment:**  
    Candidates with higher skills match show significantly better acceptance rates.

    **✔ Reduce Offer Dropouts:**  
    High-risk candidates can be identified early using placement probability scores.

    **✔ Certification Strategy:**  
    Encouraging certifications improves placement probability, especially for freshers.

    **✔ Data-Driven Hiring Decisions:**  
    Predictive insights help HR teams reduce hiring cost and time-to-fill.
    """
)

# =====================================
# FOOTER
# =====================================
st.markdown("---")
st.caption("📊 Job Acceptance Prediction System | Built with Streamlit & Plotly")
