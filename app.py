import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Smart Job Market Analyzer",
    page_icon="📊",
    layout="wide"
)

# ─── Functions defined FIRST ───────────────────────────────

@st.cache_resource
def load_model():
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import LabelEncoder

    np.random.seed(42)
    n = 5000
    job_titles = ["data analyst", "software engineer", "data scientist",
                  "ml engineer", "ux/ui designer", "network engineer"]
    locations  = ["India", "USA", "UK", "Canada", "Germany", "Australia"]
    work_types = ["Full-Time", "Part-Time", "Contract", "Temporary", "Intern"]
    companies  = ["Small", "Medium", "Large", "Enterprise"]
    exps       = ["0 to 2 Years", "2 to 5 Years", "5 to 8 Years"]
    quals      = ["B.Com", "B.Tech", "MBA", "MCA", "BCA"]

    df = pd.DataFrame({
        "Job Title":      np.random.choice(job_titles, n),
        "Experience":     np.random.choice(exps, n),
        "location":       np.random.choice(locations, n),
        "Work Type":      np.random.choice(work_types, n),
        "Company Size":   np.random.choice(companies, n),
        "Qualifications": np.random.choice(quals, n),
        "salary_avg":     np.random.uniform(67500, 97500, n).round(0),
    })

    def salary_bracket(sal):
        if sal < 75000: return 0
        elif sal <= 90000: return 1
        else: return 2

    df["salary_bracket"] = df["salary_avg"].apply(salary_bracket)

    FEATURES = ["Job Title", "Experience", "location",
                "Work Type", "Company Size", "Qualifications"]

    encoders = {}
    col_vals = {}
    model_df = df[FEATURES + ["salary_bracket"]].copy()

    for col in FEATURES:
        enc = LabelEncoder()
        model_df[col] = enc.fit_transform(model_df[col].astype(str))
        encoders[col] = enc
        col_vals[col] = sorted(df[col].astype(str).unique().tolist())

    X = model_df[FEATURES]
    y = model_df["salary_bracket"]

    clf = GradientBoostingClassifier(n_estimators=30, random_state=42)
    clf.fit(X, y)

    return clf, encoders, col_vals


@st.cache_data
def load_data():
    df = pd.read_csv("jobs_deploy.csv")
    skill_df = pd.read_csv("skill_demand.csv")
    return df, skill_df

@st.cache_resource
def load_model():
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv("jobs_deploy.csv")

    FEATURES = ["Job Title", "Experience", "location",
                "Work Type", "Company Size", "Qualifications"]

    def salary_bracket(sal):
        if sal < 75000: return 0
        elif sal <= 90000: return 1
        else: return 2

    df["salary_bracket"] = df["salary_avg"].apply(salary_bracket)
    model_df = df[FEATURES + ["salary_bracket"]].dropna()

    encoders = {}
    col_vals = {}
    encoded = model_df.copy()

    for col in FEATURES:
        enc = LabelEncoder()
        encoded[col] = enc.fit_transform(encoded[col].astype(str))
        encoders[col] = enc
        col_vals[col] = sorted(model_df[col].astype(str).unique().tolist())

    X = encoded[FEATURES]
    y = encoded["salary_bracket"]

    clf = GradientBoostingClassifier(n_estimators=30, random_state=42)
    clf.fit(X, y)

    return clf, encoders, col_vals


# ─── Load data and model ───────────────────────────────────
model, encoders, col_vals = load_model()
df, skill_df = load_data()

# ─── Header ────────────────────────────────────────────────
st.title("Smart Job Market Analyzer")
st.markdown("**1,610,462 jobs analyzed — powered by Random Forest ML**")
st.markdown("---")

# ─── Metric cards ──────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Jobs",     "1,610,462")
col2.metric("Median Salary",  "$82,500")
col3.metric("Model Accuracy", "60.9%")
col4.metric("Top Skill",      "Python")
st.markdown("---")

# ─── Sidebar ───────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Salary Predictor",
    "Top Skills",
    "Job Titles",
    "Salary Distribution",
    "Experience Analysis"
])

# ══════════════════════════════════════════════════════════
# PAGE 1 — Salary Predictor
# ══════════════════════════════════════════════════════════
if page == "Salary Predictor":
    st.header("Salary Bracket Predictor")
    st.write("Fill in the details below to predict your salary bracket.")
    st.warning(
        "Model note: We tested 3 algorithms — Gradient Boosting (60.9%), "
        "Logistic Regression (34.9%), and Random Forest (31.4%). "
        "Gradient Boosting is deployed as it handles class imbalance best."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        job_title  = st.selectbox("Job Title",   sorted(col_vals["Job Title"]))
        experience = st.selectbox("Experience",  sorted(col_vals["Experience"]))
    with col2:
        location   = st.selectbox("Location",    sorted(col_vals["location"]))
        work_type  = st.selectbox("Work Type",   sorted(col_vals["Work Type"]))
    with col3:
        company_size   = st.selectbox("Company Size",   sorted(col_vals["Company Size"]))
        qualifications = st.selectbox("Qualifications", sorted(col_vals["Qualifications"]))

    if st.button("Predict Salary Bracket", type="primary"):
        try:
            input_data = {
                "Job Title":      job_title,
                "Experience":     experience,
                "location":       location,
                "Work Type":      work_type,
                "Company Size":   company_size,
                "Qualifications": qualifications
            }
            input_df = pd.DataFrame([input_data])
            for col in input_df.columns:
                enc = encoders[col]
                val = input_df[col].astype(str).values[0]
                if val in enc.classes_:
                    input_df[col] = enc.transform([val])
                else:
                    input_df[col] = enc.transform([enc.classes_[0]])

            prediction = model.predict(input_df)[0]
            proba      = model.predict_proba(input_df)[0]

            bracket_map = {
                0: ("Low Salary",  "below $75,000",      "🔵"),
                1: ("Mid Salary",  "$75,000 to $90,000", "🟡"),
                2: ("High Salary", "above $90,000",      "🟢")
            }
            label, range_str, emoji = bracket_map[prediction]
            st.success(f"{emoji} Predicted: **{label}** ({range_str})")

            conf_df = pd.DataFrame({
                "Bracket":     ["Low (<$75K)", "Mid ($75K-$90K)", "High (>$90K)"],
                "Probability": [f"{p*100:.1f}%" for p in proba]
            })
            st.write("**Prediction confidence:**")
            st.dataframe(conf_df, hide_index=True)

        except Exception as e:
            st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════
# PAGE 2 — Top Skills
# ══════════════════════════════════════════════════════════
elif page == "Top Skills":
    st.header("Top 15 Most In-Demand Skills")
    st.info(
        "Python is the most in-demand technical skill appearing in 65,789 "
        "job postings. SQL follows at 62,074. "
        "Python + SQL is the most powerful combination for hirability."
    )
    top_skills = skill_df.head(15)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=top_skills, x="job_count", y="skill",
                hue="skill", palette="viridis", legend=False, ax=ax)
    ax.set_title("Top 15 Skills by Job Demand", fontsize=16, fontweight="bold")
    ax.set_xlabel("Number of Job Postings")
    ax.set_ylabel("Skill")
    for i, v in enumerate(top_skills["job_count"]):
        ax.text(v + 100, i, str(v), va="center", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    st.subheader("Raw Data")
    st.dataframe(skill_df.head(20), hide_index=True)

# ══════════════════════════════════════════════════════════
# PAGE 3 — Job Titles
# ══════════════════════════════════════════════════════════
elif page == "Job Titles":
    st.header("Top Job Titles by Posting Volume")
    st.info(
        "UX/UI Designer is the most posted role — nearly double "
        "Software Engineer. Data Analyst appears in the top 15, "
        "confirming strong demand for data professionals."
    )
    top_titles = df["Job Title"].value_counts().head(10).reset_index()
    top_titles.columns = ["Job Title", "count"]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=top_titles, x="count", y="Job Title",
                hue="Job Title", palette="mako", legend=False, ax=ax)
    ax.set_title("Top Job Titles", fontsize=16, fontweight="bold")
    ax.set_xlabel("Number of Postings")
    ax.set_ylabel("Job Title")
    plt.tight_layout()
    st.pyplot(fig)

# ══════════════════════════════════════════════════════════
# PAGE 4 — Salary Distribution
# ══════════════════════════════════════════════════════════
elif page == "Salary Distribution":
    st.header("Salary Distribution Across All Jobs")
    st.info(
        "Salaries range from $67,500 to $97,500 with a median of $82,500. "
        "The uniform distribution confirms this is a synthetic dataset — "
        "real data would show a right-skewed distribution."
    )
    salary_data = df["salary_avg"].dropna()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(salary_data, bins=50, color="#4C72B0", edgecolor="white")
    median_sal = salary_data.median()
    ax.axvline(median_sal, color="red", linestyle="--",
               linewidth=2, label=f"Median: ${median_sal:,.0f}")
    ax.set_title("Salary Distribution", fontsize=16, fontweight="bold")
    ax.set_xlabel("Average Salary (USD)")
    ax.set_ylabel("Number of Jobs")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    c1, c2, c3 = st.columns(3)
    c1.metric("Minimum", f"${salary_data.min():,.0f}")
    c2.metric("Median",  f"${salary_data.median():,.0f}")
    c3.metric("Maximum", f"${salary_data.max():,.0f}")

# ══════════════════════════════════════════════════════════
# PAGE 5 — Experience Analysis
# ══════════════════════════════════════════════════════════
elif page == "Experience Analysis":
    st.header("Experience Level Analysis")
    st.info(
        "Mid-level roles (3-5 years) dominate the market. "
        "This means more opportunities exist for candidates with "
        "some experience than for complete freshers."
    )
    exp_counts = df["exp_category"].value_counts().reset_index()
    exp_counts.columns = ["Experience Level", "count"]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=exp_counts, x="Experience Level", y="count",
                hue="Experience Level", palette="flare",
                legend=False, ax=ax)
    ax.set_title("Jobs by Experience Level", fontsize=16, fontweight="bold")
    ax.set_xlabel("Experience Level")
    ax.set_ylabel("Number of Jobs")
    for i, v in enumerate(exp_counts["count"]):
        ax.text(i, v + 10, str(v), ha="center", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)