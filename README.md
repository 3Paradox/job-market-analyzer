# Smart Job Market Analyzer

> End-to-end data science project analyzing 1,610,462 job postings
> to predict salary brackets and uncover skill demand trends.

## Live Demo
[Click here to view the app](https://job-market-analyzer-ehhurd69zcrfjsveynwuxh.streamlit.app)
*(Update this link after deployment)*

---

## Project Overview

| Phase | What was built |
|---|---|
| Phase 1 — Data Pipeline | Cleaned 1.6M job rows using Pandas and SQLite |
| Phase 2 — SQL Database | ETL pipeline with automated skill extraction |
| Phase 3 — EDA | 9 visualizations uncovering skill and salary trends |
| Phase 4 — ML Model | Compared 3 algorithms, deployed Gradient Boosting |
| Phase 5 — Dashboard | Streamlit app with live salary predictor |

---

## Key Findings
- Python is the #1 technical skill — appears in 65,789 job postings
- SQL follows at 62,074 — Python + SQL is the most hireable combination
- Company Size is the strongest salary predictor (30.6% feature importance)
- Mid-level roles (3-5 yrs) dominate the market with 806,169 postings
- Median salary across all 1.6M roles: $82,500

---

## Model Comparison

| Model | Accuracy | Notes |
|---|---|---|
| Gradient Boosting | 60.9% | Best — handles class imbalance internally |
| Logistic Regression | 34.9% | Balanced with class_weight |
| Random Forest v1 | 60.9% | Misleading — only predicted Mid bracket |
| Random Forest v2 | 31.4% | Honest after class_weight=balanced fix |

**Key learning:** High accuracy can be misleading with imbalanced classes.
Random Forest v1 achieved 60.9% by always predicting Mid salary — zero
precision on Low and High. Gradient Boosting achieves the same accuracy
while genuinely predicting all three brackets correctly.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python + Pandas | Data cleaning and manipulation |
| SQLite | Database storage and SQL queries |
| Scikit-learn | ML model training and evaluation |
| Matplotlib + Seaborn | Data visualization |
| Streamlit | Web app deployment |

---

## Project Structure
```
job-market-analyzer/
├── app.py                    # Streamlit dashboard
├── jobs_clean.csv            # Cleaned dataset (1.6M rows)
├── salary_model.pkl          # Trained Gradient Boosting model
├── label_encoders.pkl        # Text encoders for ML input
├── column_values.pkl         # Dropdown values for dashboard
├── skill_demand.csv          # Extracted skill counts
├── requirements.txt          # Python dependencies
└── Untitled6.ipynb           # Full analysis notebook
```

---

## How to Run Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## Resume Bullet Points

- Built ETL pipeline processing 1.6M job postings into SQLite using
  Python and Pandas with automated regex-based skill extraction
- Compared 3 ML algorithms (Gradient Boosting 60.9%, Logistic Regression
  34.9%, Random Forest 31.4%); identified and fixed class imbalance issue
- Generated 9 EDA visualizations revealing Python and SQL as top skills
  across 1.6M job postings; Company Size identified as top salary predictor
- Deployed interactive Streamlit dashboard with live salary bracket
  predictor, confidence scores, and skill demand analytics

---

## Author
Tushar Gupta
[GitHub](https://github.com/3Paradox)
[LinkedIn](https://linkedin.com/in/yourprofile)