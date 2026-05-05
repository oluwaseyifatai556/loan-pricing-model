# PROJECT: Loan Pricing & Grade Prediction
# Dataset: Lending Club (accepted_2007_to_2018Q4.csv)
# Goal: Predict loan grade (A-G) and interest rate bucket
# based on borrower and loan characteristics

# --- STEP 1: IMPORT LIBRARIES ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# --- STEP 2: LOAD THE DATA ---
useful_columns = [
    "loan_amnt",          # amount of loan applied for
    "funded_amnt",        # amount actually funded
    "term",               # 36 or 60 months
    "int_rate",           # interest rate — this is our pricing target
    "grade",              # loan grade A-G — this is our classification target
    "sub_grade",          # more granular grade (A1-G5)
    "emp_length",         # employment length
    "home_ownership",     # rent, own, mortgage
    "annual_inc",         # self-reported annual income
    "verification_status",# was income verified?
    "purpose",            # reason for loan
    "dti",                # debt-to-income ratio
    "delinq_2yrs",        # delinquencies in last 2 years
    "inq_last_6mths",     # credit inquiries in last 6 months
    "open_acc",           # number of open credit lines
    "pub_rec",            # public derogatory records
    "revol_bal",          # revolving balance
    "revol_util",         # revolving utilisation rate
    "total_acc",          # total credit lines
    "mort_acc",           # number of mortgage accounts
    "pub_rec_bankruptcies" # number of bankruptcies
]

df = pd.read_csv(
    "accepted_2007_to_2018Q4.csv",
    usecols=useful_columns,
    low_memory=False
)

print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
print("\nGrade distribution:")
print(df["grade"].value_counts().sort_index())


# --- STEP 3: EXPLORATORY DATA ANALYSIS ---

# Plot loan grade distribution
plt.figure(figsize=(8, 5))
grade_counts = df["grade"].value_counts().sort_index()
sns.barplot(x=grade_counts.index, y=grade_counts.values, palette="viridis")
plt.title("Loan Grade Distribution (A = Lowest Risk, G = Highest Risk)")
plt.xlabel("Grade")
plt.ylabel("Number of Loans")
plt.tight_layout()
plt.show()

# Plot average interest rate per grade
plt.figure(figsize=(8, 5))
df["int_rate_clean"] = df["int_rate"].astype(str).str.replace("%", "").astype(float)
avg_rate = df.groupby("grade")["int_rate_clean"].mean().sort_index()
sns.barplot(x=avg_rate.index, y=avg_rate.values, palette="rocket")
plt.title("Average Interest Rate by Loan Grade")
plt.xlabel("Grade")
plt.ylabel("Average Interest Rate (%)")
plt.tight_layout()
plt.savefig("avg_rate_by_grade.png")
plt.show()

print("\nAverage interest rate per grade:")
print(avg_rate.round(2))


# --- STEP 4: CLEAN THE DATA ---
df = df.drop(columns=["int_rate"])
df = df.rename(columns={"int_rate_clean": "int_rate"})

# Clean term: extract just the number (36 or 60)
# "36 months" -> 36
df["term"] = df["term"].astype(str).str.strip().str.extract(r"(\d+)").astype(float)

# Clean emp_length: extract years as a number
# "10+ years" -> 10, "< 1 year" -> 0
def clean_emp_length(val):
    if pd.isnull(val):
        return np.nan
    val = str(val)
    if "10+" in val:
        return 10
    if "< 1" in val:
        return 0
    nums = ''.join(filter(str.isdigit, val))
    return int(nums) if nums else np.nan

df["emp_length"] = df["emp_length"].apply(clean_emp_length)

# Clean revol_util: remove % and convert
df["revol_util"] = df["revol_util"].astype(str).str.replace("%", "").str.strip()
df["revol_util"] = pd.to_numeric(df["revol_util"], errors="coerce")

# Drop sub_grade — too granular and leaks info about grade
df = df.drop(columns=["sub_grade"])

# --- STEP 5: HANDLE MISSING VALUES ---
print("\nMissing values before cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Numeric columns: fill with median
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical columns: fill with mode
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
categorical_cols = [col for col in categorical_cols if col != "grade"]
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Drop any remaining rows with null grade (our target)
df = df.dropna(subset=["grade"])

print(f"\nFinal dataset shape: {df.shape}")

# --- STEP 6: CREATE INTEREST RATE BUCKETS (Second Target) ---
# Alongside predicting grade, we'll also predict which interest rate
# "band" a loan falls into — this is what loan pricing models do in fintech

def rate_bucket(rate):
    if rate < 8:
        return "Very Low (<8%)"
    elif rate < 12:
        return "Low (8-12%)"
    elif rate < 16:
        return "Medium (12-16%)"
    elif rate < 20:
        return "High (16-20%)"
    else:
        return "Very High (>20%)"

df["rate_bucket"] = df["int_rate"].apply(rate_bucket)
print("\nInterest rate bucket distribution:")
print(df["rate_bucket"].value_counts())


# --- STEP 7: FEATURE ENGINEERING ---
# Creating new meaningful features from existing ones

# Loan to income ratio — how large is the loan relative to income?
# Higher = more financial burden on borrower
df["loan_to_income"] = df["loan_amnt"] / (df["annual_inc"] + 1)

# Funded ratio — was the full requested amount funded?
# If funded_amnt < loan_amnt, lender was cautious
df["funded_ratio"] = df["funded_amnt"] / (df["loan_amnt"] + 1)

# Credit utilisation flag — over 80% is a red flag in credit scoring
df["high_utilisation"] = (df["revol_util"] > 80).astype(int)

# Derogatory flag — has any negative public record or bankruptcy?
df["has_derogatory"] = ((df["pub_rec"] > 0) | (df["pub_rec_bankruptcies"] > 0)).astype(int)

# Inquiry flag — multiple recent inquiries signal credit-seeking behaviour
df["high_inquiries"] = (df["inq_last_6mths"] > 3).astype(int)

print("\nFeature engineering complete. Shape:", df.shape)


# --- STEP 8: ENCODE CATEGORICAL VARIABLES ---
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# Encode the rate_bucket target too
df["rate_bucket_encoded"] = le.fit_transform(df["rate_bucket"])

# Drop target columns and non-feature columns from X
drop_cols = ["grade", "int_rate", "rate_bucket", "rate_bucket_encoded", "funded_amnt"]
feature_cols = [col for col in df.columns if col not in drop_cols]

X = df[feature_cols]
y_grade = df["grade"]           # Target 1: predict loan grade (A-G)
y_rate = df["rate_bucket_encoded"]  # Target 2: predict interest rate bucket

print(f"\nFeatures: {X.shape[1]} columns")
print(f"Sample features: {list(X.columns[:8])}")


# --- STEP 10: SAMPLE THE DATA FOR SPEED ---
# With 2M+ rows, full training takes very long
# We'll sample 200,000 rows — still a very large, statistically solid dataset
# In a real job, I'd use the full data or distributed computing

print("\nSampling 200,000 rows for training efficiency...")
sample_idx = np.random.choice(len(df), size=200000, replace=False)
X_sample = X.iloc[sample_idx].reset_index(drop=True)
y_grade_sample = y_grade.iloc[sample_idx].reset_index(drop=True)
y_rate_sample = y_rate.iloc[sample_idx].reset_index(drop=True)



# PART A: LOAN GRADE PREDICTION (A, B, C, D, E, F, G)

print("PART A: LOAN GRADE PREDICTION")

X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(
    X_sample, y_grade_sample, test_size=0.2, random_state=42, stratify=y_grade_sample
)

# MODEL 1: Logistic Regression for grade
# Good interpretable baseline — useful for explaining to stakeholders
lr_grade_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1))
])
lr_grade_pipeline.fit(X_train_g, y_train_g)

y_pred_grade_lr = lr_grade_pipeline.predict(X_test_g)
lr_grade_acc = accuracy_score(y_test_g, y_pred_grade_lr)
print(f"Logistic Regression Grade Accuracy: {lr_grade_acc:.4f}")
print(classification_report(y_test_g, y_pred_grade_lr))

# MODEL 2: Random Forest for grade
print("\nTraining Random Forest for grade prediction...")
rf_grade_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1, max_depth=15
    ))
])
rf_grade_pipeline.fit(X_train_g, y_train_g)

y_pred_grade_rf = rf_grade_pipeline.predict(X_test_g)
rf_grade_acc = accuracy_score(y_test_g, y_pred_grade_rf)
print(f"Random Forest Grade Accuracy: {rf_grade_acc:.4f}")
print(classification_report(y_test_g, y_pred_grade_rf))

# Confusion matrix — Random Forest only (better model)
plt.figure(figsize=(10, 7))
cm_grade = confusion_matrix(y_test_g, y_pred_grade_rf, labels=sorted(y_grade_sample.unique()))
sns.heatmap(cm_grade, annot=True, fmt="d", cmap="Blues",
            xticklabels=sorted(y_grade_sample.unique()),
            yticklabels=sorted(y_grade_sample.unique()))
plt.title("Confusion Matrix — Loan Grade Prediction (Random Forest)")
plt.xlabel("Predicted Grade")
plt.ylabel("Actual Grade")
plt.tight_layout()
plt.show()

# Bar chart comparing accuracies
plt.figure(figsize=(6, 4))
models = ["Logistic Regression", "Random Forest"]
accs = [lr_grade_acc, rf_grade_acc]
sns.barplot(x=models, y=accs, palette=["steelblue", "tomato"])
plt.title("Grade Prediction — Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
for i, v in enumerate(accs):
    plt.text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=11)
plt.tight_layout()
plt.show()


# PART B: PREDICT INTEREST RATE BUCKET (Loan Pricing)
print("PART B: INTEREST RATE BUCKET PREDICTION (PRICING MODEL)")

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_sample, y_rate_sample, test_size=0.2, random_state=42, stratify=y_rate_sample
)

# Logistic Regression for rate bucket
print("\nTraining Logistic Regression for rate bucket prediction...")
lr_rate_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1))
])
lr_rate_pipeline.fit(X_train_r, y_train_r)

y_pred_rate_lr = lr_rate_pipeline.predict(X_test_r)
lr_rate_acc = accuracy_score(y_test_r, y_pred_rate_lr)
print(f"Logistic Regression Rate Bucket Accuracy: {lr_rate_acc:.4f}")
print(classification_report(y_test_r, y_pred_rate_lr))

# Random Forest for rate bucket
print("\nTraining Random Forest for rate bucket prediction...")
rf_rate_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1, max_depth=15
    ))
])
rf_rate_pipeline.fit(X_train_r, y_train_r)

y_pred_rate_rf = rf_rate_pipeline.predict(X_test_r)
rf_rate_acc = accuracy_score(y_test_r, y_pred_rate_rf)
print(f"Random Forest Rate Bucket Accuracy: {rf_rate_acc:.4f}")
print(classification_report(y_test_r, y_pred_rate_rf))

# Bar chart comparing rate bucket accuracies
plt.figure(figsize=(6, 4))
models = ["Logistic Regression", "Random Forest"]
accs = [lr_rate_acc, rf_rate_acc]
sns.barplot(x=models, y=accs, palette=["steelblue", "tomato"])
plt.title("Rate Bucket Prediction — Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)

for i, v in enumerate(accs):
    plt.text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=11)
plt.tight_layout()
plt.show()

# PART C: WHAT DRIVES LOAN PRICING? (Feature Importance)

print("PART C: WHAT DRIVES LOAN PRICING?")
importances = rate_pipeline.named_steps["model"].feature_importances_
feat_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": importances
}).sort_values("Importance", ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_df, x="Importance", y="Feature", palette="magma")
plt.title("Top 15 Features Driving Loan Pricing (Interest Rate Bucket)")
plt.tight_layout()
plt.show()

print("\nTop 10 pricing drivers:")
print(feat_df.head(10).to_string(index=False))


# PART D: BUSINESS INSIGHT — WHAT DOES THIS MEAN FOR LENDING?

print("PART D: BUSINESS INSIGHTS")

# Average interest rate by home ownership
print("\nAverage interest rate by home ownership:")
home_rates = df.groupby("home_ownership")["int_rate"].mean().sort_values()
# Note: home_ownership is encoded, so we show the encoded values
# In a real project you'd reverse-encode for presentation

# Average interest rate by loan purpose
print("\nAverage DTI by rate bucket:")
print(df.groupby("rate_bucket")["dti"].mean().sort_values().round(2))

# Loan amount distribution by grade
plt.figure(figsize=(10, 5))
# Re-load grade as original for plotting
df_plot = pd.read_csv(
    "accepted_2007_to_2018Q4.csv",
    usecols=["grade", "loan_amnt", "int_rate"],
    low_memory=False,
    nrows=100000   # just first 100k rows for the plot
)
df_plot["int_rate"] = df_plot["int_rate"].astype(str).str.replace("%", "").astype(float)

sns.boxplot(data=df_plot, x="grade", y="loan_amnt",
            order=["A", "B", "C", "D", "E", "F", "G"],
            palette="viridis")
plt.title("Loan Amount Distribution by Grade")
plt.xlabel("Loan Grade")
plt.ylabel("Loan Amount ($)")
plt.tight_layout()
plt.show()

print(f"  Grade Prediction     — Logistic Regression: {lr_grade_acc:.4f}  |  Random Forest: {rf_grade_acc:.4f}")
print(f"  Rate Bucket Prediction — Logistic Regression: {lr_rate_acc:.4f}  |  Random Forest: {rf_rate_acc:.4f}")
