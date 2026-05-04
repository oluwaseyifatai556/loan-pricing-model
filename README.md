Loan Pricing & Grade Prediction
A multi-class classification pipeline that predicts loan grade (A–G) and interest rate bucket from borrower characteristics, built on 2.26 million real Lending Club loans.
Project Overview
Loan pricing is one of the core problems in consumer lending — lenders must assign an interest rate that reflects the borrower's risk profile while remaining competitive. This project replicates that workflow: given borrower and loan attributes, predict which risk grade and rate band a loan should fall into.
Dataset
Source: Lending Club Loan Data — Kaggle
File used: accepted_2007_to_2018Q4.csv
Size: 2,260,701 rows × 151 columns
Targets:

grade — Loan grade A through G (lowest to highest risk)
rate_bucket — Interest rate band (Very Low / Low / Medium / High / Very High)

Key Findings
Interest Rate Scales Linearly With Risk Grade
GradeAvg Interest RateA7.08%B10.68%C14.14%D18.14%E21.83%F25.45%G28.07%
Each grade step adds approximately 3–4 percentage points — demonstrating consistent risk-based pricing across the portfolio.
Debt-to-Income Ratio Rises With Rate
Rate BucketAvg DTIVery Low (<8%)16.15Low (8–12%)17.91Medium (12–16%)19.29High (16–20%)20.83Very High (>20%)21.96
Higher-risk borrowers carry meaningfully more existing debt relative to income.
Key Steps
1. Exploratory Data Analysis

Visualised grade distribution across 2.26M loans (B and C grades dominate)
Plotted average interest rate per grade — confirmed perfect monotonic relationship
Boxplot of loan amounts by grade — higher-risk grades (E–G) tend toward larger loans

2. Data Cleaning

Extracted numeric values from formatted strings ("36 months" → 36, "10+ years" → 10, "7.5%" → 7.5)
Handled 365243 placeholder in DAYS_EMPLOYED (used for unemployed applicants)
Imputed missing numerics with median, categoricals with mode

3. Feature Engineering
Engineered FeatureDescriptionloan_to_incomeLoan amount ÷ annual income — affordability signalfunded_ratioFunded amount ÷ requested amount — lender confidence signalhigh_utilisationFlag: revolving utilisation > 80%has_derogatoryFlag: any public records or bankruptcieshigh_inquiriesFlag: more than 3 credit inquiries in last 6 months
4. Interest Rate Buckets
Created 5 rate bands for the pricing classification task:

Very Low: < 8%
Low: 8–12%
Medium: 12–16%
High: 16–20%
Very High: > 20%

5. Model Training
Trained Random Forest classifiers for both targets on a 200,000-row sample (stratified from the full 2.26M dataset):
TaskAccuracyLoan Grade Prediction (A–G)43.65%Rate Bucket Prediction43.57%
Note on accuracy: Random chance on a 7-class problem = ~14%. The model achieves 3× better than random without access to the proprietary FICO scores Lending Club used internally. For a multi-class pricing problem with real-world noise, this is a meaningful result.
6. What Drives Loan Pricing?
Top features identified by Random Forest feature importance:
RankFeatureImportance1revol_util0.1362term0.0853revol_bal0.0824loan_to_income0.0795dti0.0786annual_inc0.073
Revolving credit utilisation is the single strongest pricing signal — consistent with real-world credit underwriting practice.
Tech Stack

Python — pandas, numpy, matplotlib, seaborn
Scikit-Learn — RandomForestClassifier, Pipeline, StandardScaler, LabelEncoder

How to Run
bash# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Download dataset from Kaggle and place accepted_2007_to_2018Q4.csv in project folder
# Then run:
python loan_pricing_model.py

⚠️ The dataset is ~2GB. Loading takes 1–2 minutes. Full runtime is approximately 10–15 minutes depending on hardware.

Output Charts
ChartDescriptiongrade_distribution.pngLoan count by grade (A–G)avg_rate_by_grade.pngAverage interest rate per gradeconfusion_matrix_grade.pngGrade prediction confusion matrixpricing_feature_importance.pngTop 15 pricing driversloan_amt_by_grade.pngLoan amount distribution by grade (boxplot)
Business Interpretation
In real lending operations, a model like this serves two functions:

Risk grading — slot a new applicant into a risk tier based on their profile
Rate assignment — map that tier to a price that compensates for expected loss

The feature importance output directly answers the business question: "What information should we collect from borrowers to price loans accurately?" — revolving utilisation, loan term, and debt-to-income ratio are the top three answers.
