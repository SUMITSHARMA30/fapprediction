

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

sns.set(style="whitegrid")



CSV_FILE = r"C:\Users\Sumit\Downloads\project.csv"  # your file path
df = pd.read_csv(CSV_FILE)
df.columns = df.columns.str.strip()

print("Original shape:", df.shape)

# ============================================
# 2. DROP EMPTY & USELESS DATA
# ============================================

TARGET = "FAP Success Per"

# Keep only rows where target exists
df = df.dropna(subset=[TARGET])

# Drop fully empty columns
df = df.dropna(axis=1, how="all")

# Drop RCA if exists
df = df.drop(columns=["RCA"], errors="ignore")

print("Shape after dropping empty rows:", df.shape)

# ============================================
# 3. CONVERT NUMERIC COLUMNS
# ============================================

possible_numeric = [
    "Total Pickup Assigned", "Pickup Done", "FAP",
    "FAP Success", "Pickup Canceled", "QC Failure",
    "Pickup Failure", "Total Pickup Per", "FAP Success Per"
]

for col in possible_numeric:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ============================================
# 4. DATE FEATURES (NO ROW DROP)
# ============================================

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["day"] = df["Date"].dt.day
    df["month"] = df["Date"].dt.month
    df = df.drop(columns=["Date"])

# ============================================
# 5. SAFE MISSING VALUE HANDLING
# ============================================

# Numeric → median
for col in df.select_dtypes(include=["int64", "float64"]).columns:
    df[col] = df[col].fillna(df[col].median())

# Categorical → mode or "Unknown"
for col in df.select_dtypes(include="object").columns:
    if df[col].notna().sum() > 0:
        df[col] = df[col].fillna(df[col].mode().iloc[0])
    else:
        df[col] = df[col].fillna("Unknown")

# Drop all-NaN columns if any
all_nan_cols = [c for c in df.columns if df[c].isna().all()]
if all_nan_cols:
    print("Dropping all-NaN columns:", all_nan_cols)
    df = df.drop(columns=all_nan_cols)

print("\nMissing values after cleaning:")
print(df.isna().sum())

# ============================================
# 6. RICH EDA SECTION 😎
# ============================================

print("\n===== BASIC STATS =====")
print(df[possible_numeric].describe())

# 6.1 Distribution of FAP Success Per
plt.figure(figsize=(7, 4))
sns.histplot(df[TARGET], kde=True)
plt.title("Distribution of FAP Success Per")
plt.xlabel("FAP Success Per")
plt.tight_layout()
plt.show()

# 6.2 FAP Success Per by AM (if AM exists)
if "AM" in df.columns:
    plt.figure(figsize=(8, 4))
    am_group = df.groupby("AM")[TARGET].agg(["mean", "count"]).sort_values("mean", ascending=False)
    print("\n===== AM-wise Performance =====")
    print(am_group)

    sns.barplot(x=am_group.index, y=am_group["mean"])
    plt.title("Average FAP Success Per by AM")
    plt.ylabel("Mean FAP Success Per")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 6.3 Station-wise FAP boxplot (performance spread)
if "station_code" in df.columns:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="station_code", y=TARGET)
    plt.title("FAP Success Per Distribution by Station Code")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

# 6.4 Relationship: Pickup Done vs FAP Success Per
if "Pickup Done" in df.columns:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df, x="Pickup Done", y=TARGET, hue=df.get("AM", None))
    plt.title("Pickup Done vs FAP Success Per")
    plt.tight_layout()
    plt.show()

# 6.5 Heatmap of correlations
num_for_corr = [c for c in possible_numeric if c in df.columns and df[c].nunique() > 1]
if len(num_for_corr) >= 2:
    plt.figure(figsize=(7, 5))
    sns.heatmap(df[num_for_corr].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    plt.show()
else:
    print("\n⚠️ Not enough numeric variation for correlation heatmap.")

# 6.6 Pairplot (small numeric subset)
num_subset = [c for c in ["Total Pickup Assigned", "Pickup Done", "FAP Success Per"] if c in df.columns]
if len(num_subset) >= 2:
    sns.pairplot(df[num_subset])
    plt.suptitle("Pairplot of Key Numeric Features", y=1.02)
    plt.show()

# ============================================
# 7. REGRESSION – LINEAR REGRESSION
# ============================================

print("\n===== REGRESSION: Predicting FAP Success Per =====")

X_reg = df.drop(columns=[TARGET])
X_reg = pd.get_dummies(X_reg, drop_first=True)
X_reg = X_reg.fillna(0)
y_reg = df[TARGET]

print("Shape of X_reg:", X_reg.shape)

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

scaler_reg = StandardScaler()
Xr_train_s = scaler_reg.fit_transform(Xr_train)
Xr_test_s = scaler_reg.transform(Xr_test)

reg = LinearRegression()
reg.fit(Xr_train_s, yr_train)
yr_pred = reg.predict(Xr_test_s)

mse = mean_squared_error(yr_test, yr_pred)
rmse = mse ** 0.5

print("MAE :", mean_absolute_error(yr_test, yr_pred))
print("RMSE:", rmse)
print("R2  :", r2_score(yr_test, yr_pred))

# ============================================
# 8. CLASSIFICATION – DECISION TREE
# ============================================

print("\n===== CLASSIFICATION: Good vs Poor FAP =====")

threshold = df[TARGET].median()
df["FAP_Class"] = (df[TARGET] >= threshold).astype(int)

X_clf = df.drop(columns=[TARGET, "FAP_Class"])
X_clf = pd.get_dummies(X_clf, drop_first=True)
X_clf = X_clf.fillna(0)
y_clf = df["FAP_Class"]

print("Shape of X_clf:", X_clf.shape)

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

scaler_clf = StandardScaler()
Xc_train_s = scaler_clf.fit_transform(Xc_train)
Xc_test_s = scaler_clf.transform(Xc_test)

clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(Xc_train_s, yc_train)
yc_pred = clf.predict(Xc_test_s)

print("Accuracy :", accuracy_score(yc_test, yc_pred))
print("Precision:", precision_score(yc_test, yc_pred))
print("Recall   :", recall_score(yc_test, yc_pred))
print("F1 Score :", f1_score(yc_test, yc_pred))
print("Confusion Matrix:\n", confusion_matrix(yc_test, yc_pred))

print("\n✅ SCRIPT RAN SUCCESSFULLY ✅")
