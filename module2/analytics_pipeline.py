# ============================================================
# MODULE 2 — ANALYTICS PIPELINE
# ALL PART-A TASKS IN ONE CELL
# ============================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid")

print("=" * 70)
print("MODULE 2 — ANALYTICS PIPELINE")
print("=" * 70)


# ============================================================
# 1. LOAD DATASET ONCE + SAVE OFFLINE FALLBACK
# ============================================================

print("\n" + "=" * 70)
print("1. LOADING TITANIC DATASET")
print("=" * 70)

# IMPORTANT: This is the ONLY sns.load_dataset() call
df = sns.load_dataset("titanic")

print("Dataset loaded successfully.")
print("Original shape:", df.shape)

# Immediately save raw dataset as required
df.to_csv("titanic.csv", index=False)

print("Raw dataset saved as: titanic.csv")


# ============================================================
# 2. PROFILE DATASET
# ============================================================

print("\n" + "=" * 70)
print("2. DATASET INFORMATION")
print("=" * 70)

print("\n--- df.info() ---")
df.info()

print("\n--- df.describe() ---")
print(df.describe())

print("\n--- df.shape ---")
print(df.shape)


# ============================================================
# 3. MISSING VALUE PERCENTAGES
# ============================================================

print("\n" + "=" * 70)
print("3. MISSING VALUE ANALYSIS")
print("=" * 70)

missing_percentage = (df.isnull().sum() / len(df)) * 100

missing_percentage = missing_percentage[
    missing_percentage > 0
].sort_values(ascending=False)

print("\nMissing-value percentage for affected columns:")

for column, percentage in missing_percentage.items():
    print(f"{column:15s}: {percentage:.2f}%")


# ============================================================
# 4. MISSING VALUE HANDLING
# ============================================================

print("\n" + "=" * 70)
print("4. MISSING VALUE HANDLING STRATEGY")
print("=" * 70)

clean_df = df.copy()

for column, percentage in missing_percentage.items():

    if percentage < 5:
        print(
            f"{column}: {percentage:.2f}% missing "
            "-> DROP affected rows"
        )

    elif percentage <= 30:
        print(
            f"{column}: {percentage:.2f}% missing "
            "-> IMPUTE"
        )

    else:
        print(
            f"{column}: {percentage:.2f}% missing "
            "-> DROP COLUMN"
        )


# High missingness: deck
# Imputation would be unreliable because most values are missing.
clean_df = clean_df.drop(columns=["deck"])

# Under 5% missing -> drop affected rows
clean_df = clean_df.dropna(
    subset=["embarked", "embark_town"]
)

# 5%-30% missing -> median imputation
clean_df["age"] = clean_df["age"].fillna(
    clean_df["age"].median()
)

print("\nCleaning decisions:")
print("- deck: dropped because missingness is very high")
print("- embarked: rows with missing values dropped")
print("- embark_town: rows with missing values dropped")
print("- age: median imputation")

print("\nCleaned shape:", clean_df.shape)

print("\nRemaining missing values:")
print(clean_df.isnull().sum())


# ============================================================
# 5. SAVE CLEANED DATASET
# ============================================================

# The same committed CSV is used by the modeling notebook.
clean_df.to_csv("titanic.csv", index=False)

print("\nCleaned dataset saved to titanic.csv")


# ============================================================
# 6. UNIVARIATE ANALYSIS — AGE HISTOGRAM
# ============================================================

print("\n" + "=" * 70)
print("5. UNIVARIATE ANALYSIS — AGE")
print("=" * 70)

plt.figure(figsize=(8, 5))

sns.histplot(
    clean_df["age"],
    bins=30,
    kde=True
)

plt.title("Distribution of Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

print(
    "\nInterpretation: The histogram shows the distribution of "
    "passenger ages. Most passengers are concentrated in the "
    "younger and middle-age ranges, with fewer very old passengers."
)


# ============================================================
# 7. AGE BOXPLOT
# ============================================================

plt.figure(figsize=(8, 4))

sns.boxplot(
    x=clean_df["age"]
)

plt.title("Box Plot of Age")
plt.xlabel("Age")
plt.tight_layout()
plt.show()


# ============================================================
# 8. AGE IQR OUTLIERS
# ============================================================

age_q1 = clean_df["age"].quantile(0.25)
age_q3 = clean_df["age"].quantile(0.75)
age_iqr = age_q3 - age_q1

age_lower = age_q1 - 1.5 * age_iqr
age_upper = age_q3 + 1.5 * age_iqr

age_outliers = clean_df[
    (clean_df["age"] < age_lower) |
    (clean_df["age"] > age_upper)
]

print("\nAGE IQR ANALYSIS")
print(f"Q1              : {age_q1:.2f}")
print(f"Q3              : {age_q3:.2f}")
print(f"IQR             : {age_iqr:.2f}")
print(f"Lower Bound     : {age_lower:.2f}")
print(f"Upper Bound     : {age_upper:.2f}")
print(f"Number Outliers : {len(age_outliers)}")


# ============================================================
# 9. FARE HISTOGRAM
# ============================================================

print("\n" + "=" * 70)
print("6. UNIVARIATE ANALYSIS — FARE")
print("=" * 70)

plt.figure(figsize=(8, 5))

sns.histplot(
    clean_df["fare"],
    bins=40,
    kde=True
)

plt.title("Distribution of Fare")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# ============================================================
# 10. FARE BOXPLOT
# ============================================================

plt.figure(figsize=(8, 4))

sns.boxplot(
    x=clean_df["fare"]
)

plt.title("Box Plot of Fare")
plt.xlabel("Fare")
plt.tight_layout()
plt.show()


# ============================================================
# 11. FARE IQR OUTLIERS
# ============================================================

fare_q1 = clean_df["fare"].quantile(0.25)
fare_q3 = clean_df["fare"].quantile(0.75)
fare_iqr = fare_q3 - fare_q1

fare_lower = fare_q1 - 1.5 * fare_iqr
fare_upper = fare_q3 + 1.5 * fare_iqr

fare_outliers = clean_df[
    (clean_df["fare"] < fare_lower) |
    (clean_df["fare"] > fare_upper)
]

print("\nFARE IQR ANALYSIS")
print(f"Q1              : {fare_q1:.2f}")
print(f"Q3              : {fare_q3:.2f}")
print(f"IQR             : {fare_iqr:.2f}")
print(f"Lower Bound     : {fare_lower:.2f}")
print(f"Upper Bound     : {fare_upper:.2f}")
print(f"Number Outliers : {len(fare_outliers)}")


# ============================================================
# 12. FARE MEAN MEDIAN MODE
# ============================================================

fare_mean = clean_df["fare"].mean()
fare_median = clean_df["fare"].median()
fare_mode = clean_df["fare"].mode()[0]

print("\nFARE STATISTICS")
print(f"Mean   : {fare_mean:.2f}")
print(f"Median : {fare_median:.2f}")
print(f"Mode   : {fare_mode:.2f}")

if fare_mean > fare_median > fare_mode:
    skew_result = "RIGHT-SKEWED"
elif fare_mean < fare_median < fare_mode:
    skew_result = "LEFT-SKEWED"
else:
    skew_result = "Not determined by simple mean/median/mode ordering"

print("Distribution:", skew_result)

print(
    "\nInterpretation: Fare is right-skewed because the mean is "
    "greater than the median, reflecting a smaller number of "
    "passengers who paid substantially higher fares."
)


# ============================================================
# 13. SURVIVAL RATE BY SEX
# ============================================================

print("\n" + "=" * 70)
print("7. SURVIVAL RATE BY SEX")
print("=" * 70)

female = clean_df[
    clean_df["sex"] == "female"
]

male = clean_df[
    clean_df["sex"] == "male"
]

female_survival = female["survived"].mean() * 100
male_survival = male["survived"].mean() * 100

print(f"Female survival rate: {female_survival:.2f}%")
print(f"Male survival rate  : {male_survival:.2f}%")


# ============================================================
# 14. SURVIVAL RATE BY PCLASS
# ============================================================

print("\n" + "=" * 70)
print("8. SURVIVAL RATE BY PASSENGER CLASS")
print("=" * 70)

for pclass in sorted(clean_df["pclass"].unique()):

    subset = clean_df[
        clean_df["pclass"] == pclass
    ]

    rate = subset["survived"].mean() * 100

    print(
        f"Class {pclass}: {rate:.2f}%"
    )


# ============================================================
# 15. SURVIVAL RATE BY SEX + PCLASS
# ============================================================

print("\n" + "=" * 70)
print("9. SURVIVAL RATE BY SEX AND PCLASS")
print("=" * 70)

for sex in ["female", "male"]:

    for pclass in sorted(
        clean_df["pclass"].unique()
    ):

        subset = clean_df[
            (clean_df["sex"] == sex) &
            (clean_df["pclass"] == pclass)
        ]

        rate = subset["survived"].mean() * 100

        print(
            f"{sex.capitalize():6s}, "
            f"Class {pclass}: "
            f"{rate:.2f}%"
        )


# ============================================================
# 16. CHART 1 — SURVIVAL BY SEX
# ============================================================

plt.figure(figsize=(8, 5))

sns.barplot(
    data=clean_df,
    x="sex",
    y="survived"
)

plt.title("Survival Rate by Sex")
plt.xlabel("Sex")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.show()

print(
    "\nChart 1 Interpretation: Survival rates differ substantially "
    "between female and male passengers. This indicates that sex "
    "is strongly associated with the observed survival outcome."
)


# ============================================================
# 17. CHART 2 — SURVIVAL BY PCLASS
# ============================================================

plt.figure(figsize=(8, 5))

sns.barplot(
    data=clean_df,
    x="pclass",
    y="survived"
)

plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.show()

print(
    "\nChart 2 Interpretation: Survival rates vary across passenger "
    "classes. Passenger class therefore provides useful information "
    "about differences in observed survival outcomes."
)


# ============================================================
# 18. CHART 3 — SEX + PCLASS
# ============================================================

plt.figure(figsize=(9, 5))

sns.barplot(
    data=clean_df,
    x="pclass",
    y="survived",
    hue="sex"
)

plt.title("Survival Rate by Sex and Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.show()

print(
    "\nChart 3 Interpretation: Combining sex and passenger class "
    "reveals a more detailed survival pattern than either variable "
    "alone. Survival rates differ across the sex-class combinations, "
    "showing that the variables jointly contain useful information."
)


# ============================================================
# 19. CHART 4 — AGE VS SURVIVAL
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=clean_df,
    x="survived",
    y="age"
)

plt.title("Age Distribution by Survival")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Age")
plt.tight_layout()
plt.show()

print(
    "\nChart 4 Interpretation: The age distributions of survivors "
    "and non-survivors overlap considerably. Age alone does not "
    "completely separate the two groups, but it can provide useful "
    "information when combined with other passenger characteristics."
)


# ============================================================
# 20. CORRELATION MATRIX — EXACTLY SIX COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("10. CORRELATION MATRIX")
print("=" * 70)

corr_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

corr_matrix = clean_df[
    corr_columns
].corr()

print("\nCorrelation Matrix:")
print(corr_matrix)


# ============================================================
# 21. CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(9, 7))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True
)

plt.title("Correlation Matrix — Titanic Numeric Features")
plt.tight_layout()
plt.show()


# ============================================================
# 22. FIND TWO STRONGEST CORRELATIONS
# ============================================================

corr_pairs = []

columns = corr_matrix.columns

for i in range(len(columns)):

    for j in range(i + 1, len(columns)):

        col1 = columns[i]
        col2 = columns[j]

        value = corr_matrix.loc[
            col1,
            col2
        ]

        corr_pairs.append(
            (
                col1,
                col2,
                value,
                abs(value)
            )
        )


strongest_pairs = sorted(
    corr_pairs,
    key=lambda x: x[3],
    reverse=True
)[:2]

print("\nTWO STRONGEST CORRELATIONS:")

for col1, col2, value, absolute_value in strongest_pairs:

    direction = (
        "positive"
        if value > 0
        else "negative"
    )

    print(
        f"{col1} <-> {col2}: "
        f"{value:.4f} "
        f"({direction}, "
        f"|r| = {absolute_value:.4f})"
    )


print(
    "\nCorrelation Interpretation: The two relationships printed "
    "above are the strongest correlations because they have the "
    "largest absolute off-diagonal correlation coefficients."
)

print(
    "The correlation values describe linear association and do not "
    "by themselves establish causation."
)


# ============================================================
# 23. STANDARDIZATION — EDA SANITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("11. STANDARDIZATION CHECK")
print("=" * 70)

print("\nBEFORE STANDARDIZATION")

print(
    f"Age  Mean: {clean_df['age'].mean():.4f}"
)
print(
    f"Age  Std : {clean_df['age'].std():.4f}"
)

print(
    f"Fare Mean: {clean_df['fare'].mean():.4f}"
)
print(
    f"Fare Std : {clean_df['fare'].std():.4f}"
)


# Create a separate copy ONLY for EDA standardization
standardized_df = clean_df.copy()

scaler = StandardScaler()

standardized_df[
    ["age", "fare"]
] = scaler.fit_transform(
    standardized_df[
        ["age", "fare"]
    ]
)


# ============================================================
# 24. AFTER STANDARDIZATION
# ============================================================

print("\nAFTER STANDARDIZATION")

print(
    f"Age  Mean: {standardized_df['age'].mean():.4f}"
)
print(
    f"Age  Std : {standardized_df['age'].std():.4f}"
)

print(
    f"Fare Mean: {standardized_df['fare'].mean():.4f}"
)
print(
    f"Fare Std : {standardized_df['fare'].std():.4f}"
)


# ============================================================
# 25. BEFORE/AFTER COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame({

    "Variable": [
        "age",
        "fare"
    ],

    "Before Mean": [
        clean_df["age"].mean(),
        clean_df["fare"].mean()
    ],

    "Before Std": [
        clean_df["age"].std(),
        clean_df["fare"].std()
    ],

    "After Mean": [
        standardized_df["age"].mean(),
        standardized_df["fare"].mean()
    ],

    "After Std": [
        standardized_df["age"].std(),
        standardized_df["fare"].std()
    ]
})

print("\nBEFORE / AFTER STANDARDIZATION")
print(
    comparison.to_string(index=False)
)


# ============================================================
# 26. STANDARDIZATION VISUAL COMPARISON
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 8)
)

sns.histplot(
    clean_df["age"],
    kde=True,
    ax=axes[0, 0]
)

axes[0, 0].set_title(
    "Age — Before Standardization"
)

sns.histplot(
    standardized_df["age"],
    kde=True,
    ax=axes[0, 1]
)

axes[0, 1].set_title(
    "Age — After Standardization"
)

sns.histplot(
    clean_df["fare"],
    kde=True,
    ax=axes[1, 0]
)

axes[1, 0].set_title(
    "Fare — Before Standardization"
)

sns.histplot(
    standardized_df["fare"],
    kde=True,
    ax=axes[1, 1]
)

axes[1, 1].set_title(
    "Fare — After Standardization"
)

plt.tight_layout()
plt.show()


print(
    "\nStandardization Interpretation: After applying z-score "
    "standardization, age and fare have approximately zero mean "
    "and unit standard deviation."
)

print(
    "This standardization is only an EDA sanity check. It must NOT "
    "be reused directly in the modeling pipeline because the modeling "
    "pipeline should fit its preprocessing only on the training data."
)


# ============================================================
# 27. FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

print("\nCleaned dataset shape:", clean_df.shape)

print("\nColumns in cleaned dataset:")
print(list(clean_df.columns))

print("\nRemaining missing values:")
print(
    clean_df.isnull().sum()
)

print("\nCorrelation columns used:")
print(corr_columns)

print("\nNumber of charts generated:")
print("1. Age histogram")
print("2. Age box plot")
print("3. Fare histogram")
print("4. Fare box plot")
print("5. Survival by sex")
print("6. Survival by passenger class")
print("7. Survival by sex + class")
print("8. Age vs survival")
print("9. Correlation heatmap")
print("10. Standardization before/after distributions")

print("\n" + "=" * 70)
print("MODULE 2 PART A COMPLETED")
print("=" * 70)
# ============================================================
# ZEpto / ANALYTICS MODULE 2 - PART B
# Predictive Modeling
# ============================================================

# If needed, install once:
# !pip install imbalanced-learn joblib

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import joblib


# ============================================================
# 1. SETUP
# ============================================================

os.makedirs("charts", exist_ok=True)

print("=" * 70)
print("PART B - PREDICTIVE MODELING")
print("=" * 70)


# ============================================================
# 2. LOAD THE OFFLINE FALLBACK DATASET
# IMPORTANT:
# Do NOT load sns.load_dataset() again.
# titanic.csv was created in Part A.
# ============================================================

DATA_PATH = "titanic.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "titanic.csv not found. Run Part A first and create it using:\n"
        'df.to_csv("titanic.csv", index=False)'
    )

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded from:", DATA_PATH)
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 3. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "survived",
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

missing_required = [
    col for col in required_columns
    if col not in df.columns
]

if missing_required:
    raise ValueError(
        f"Missing required columns: {missing_required}"
    )


# ============================================================
# 4. CLASS BALANCE
# ============================================================

print("\n" + "=" * 70)
print("CLASS BALANCE")
print("=" * 70)

class_counts = df["survived"].value_counts().sort_index()
class_percentages = (
    df["survived"].value_counts(normalize=True)
    .sort_index()
    * 100
)

balance_table = pd.DataFrame({
    "Class": ["Not Survived (0)", "Survived (1)"],
    "Count": [
        class_counts.get(0, 0),
        class_counts.get(1, 0)
    ],
    "Percentage": [
        class_percentages.get(0, 0),
        class_percentages.get(1, 0)
    ]
})

print(balance_table.to_string(index=False))

print(
    "\nStratification justification:"
    "\nThe survived classes are not perfectly balanced. "
    "A stratified train/test split preserves approximately the same "
    "class proportions in both training and testing data. "
    "This makes model evaluation more representative and prevents "
    "the minority class proportion from changing substantially between splits."
)


# ============================================================
# 5. PREPARE MODELING DATA
# ============================================================

# Classification features
classification_features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

X = df[classification_features].copy()
y = df["survived"].copy()


# ============================================================
# 6. STRATIFIED TRAIN/TEST SPLIT
# IMPORTANT:
# This happens BEFORE preprocessing.
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print("Training shape:", X_train.shape)
print("Testing shape :", X_test.shape)

print("\nTraining class distribution:")
print(y_train.value_counts(normalize=True).sort_index())

print("\nTesting class distribution:")
print(y_test.value_counts(normalize=True).sort_index())


# ============================================================
# 7. PREPROCESSING
# ============================================================
# Numeric:
# - Median imputation
# - StandardScaler
#
# Categorical:
# - Most-frequent imputation
# - OneHotEncoder
#
# IMPORTANT:
# ColumnTransformer is fitted ONLY on X_train.
# Therefore all statistics are learned from training data only.
# ============================================================

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked"
]

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])

preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_features
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )
])


print("\n" + "=" * 70)
print("PREPROCESSING STRATEGY")
print("=" * 70)

print("""
Numeric columns:
- Missing values: median imputation
- Scaling: StandardScaler

Categorical columns:
- Missing values: most-frequent imputation
- Encoding: OneHotEncoder(handle_unknown='ignore')

The preprocessing is inside a Pipeline/ColumnTransformer.
It is fitted only on X_train and then used to transform X_test.
Therefore test-set information is not used to calculate imputation,
encoding, or scaling statistics.

Percentage-based missing-value rule:
- 0% to <5%   : simple imputation is generally sufficient.
- 5% to <30%  : imputation with a defensible statistic is appropriate.
- >=30%       : investigate whether the feature should be retained.
For this modeling pipeline, the required modeling columns are retained
and missing numeric values are median-imputed while categorical values
are most-frequent-imputed.
""")


# ============================================================
# 8. MISSING VALUES IN MODELING COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing_info = pd.DataFrame({
    "Column": classification_features,
    "Missing_Count": [
        df[col].isna().sum()
        for col in classification_features
    ],
    "Missing_Percentage": [
        df[col].isna().mean() * 100
        for col in classification_features
    ]
})

print(
    missing_info.to_string(
        index=False,
        formatters={
            "Missing_Percentage": "{:.2f}%".format
        }
    )
)


# ============================================================
# 9. FUNCTION FOR MODEL EVALUATION
# ============================================================

def evaluate_classifier(
    model_name,
    model,
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Fit model and calculate:
    confusion matrix, accuracy, precision, recall,
    F1, ROC curve and AUC.
    """

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)

    cm = confusion_matrix(y_test, y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )
    auc = roc_auc_score(
        y_test,
        y_prob
    )

    fpr, tpr, _ = roc_curve(
        y_test,
        y_prob
    )

    print("\n" + "-" * 60)
    print(model_name)
    print("-" * 60)

    print("\nConfusion Matrix:")
    print(cm)

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}")

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC": auc,
        "Confusion Matrix": cm,
        "FPR": fpr,
        "TPR": tpr,
        "model_object": model
    }


# ============================================================
# 10. THREE CLASSIFIERS
# ============================================================

logistic_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])


decision_tree_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        )
    )
])


random_forest_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    )
])


# ============================================================
# 11. TRAIN AND EVALUATE ALL THREE
# ============================================================

print("\n" + "=" * 70)
print("THREE CLASSIFIERS")
print("=" * 70)

results = []

results.append(
    evaluate_classifier(
        "Logistic Regression",
        logistic_pipeline,
        X_train,
        X_test,
        y_train,
        y_test
    )
)

results.append(
    evaluate_classifier(
        "Decision Tree",
        decision_tree_pipeline,
        X_train,
        X_test,
        y_train,
        y_test
    )
)

results.append(
    evaluate_classifier(
        "Random Forest",
        random_forest_pipeline,
        X_train,
        X_test,
        y_train,
        y_test
    )
)


# ============================================================
# 12. CLASSIFICATION COMPARISON TABLE
# ============================================================

classification_results = pd.DataFrame([
    {
        "Model": r["Model"],
        "Accuracy": r["Accuracy"],
        "Precision": r["Precision"],
        "Recall": r["Recall"],
        "F1": r["F1"],
        "AUC": r["AUC"]
    }
    for r in results
])

print("\n" + "=" * 70)
print("CLASSIFIER COMPARISON")
print("=" * 70)

print(
    classification_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

classification_results.to_csv(
    "classifier_comparison.csv",
    index=False
)


# ============================================================
# 13. CONFUSION MATRICES SIDE BY SIDE
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 4)
)

for ax, result in zip(axes, results):

    sns.heatmap(
        result["Confusion Matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax
    )

    ax.set_title(result["Model"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "charts/confusion_matrices.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 14. ROC CURVES FOR ALL THREE MODELS
# ============================================================

plt.figure(figsize=(8, 6))

for result in results:

    plt.plot(
        result["FPR"],
        result["TPR"],
        label=f'{result["Model"]} (AUC={result["AUC"]:.3f})'
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Three Classifiers")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig(
    "charts/roc_curves.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 15. DECISION TREE VISUALIZATION
# ============================================================

print("\n" + "=" * 70)
print("DECISION TREE VISUALIZATION")
print("=" * 70)

dt_fitted = decision_tree_pipeline

feature_names = (
    dt_fitted
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

tree_model = (
    dt_fitted
    .named_steps["classifier"]
)

plt.figure(figsize=(24, 12))

plot_tree(
    tree_model,
    feature_names=feature_names,
    class_names=["Not Survived", "Survived"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree Classifier")

plt.savefig(
    "charts/decision_tree.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 16. IMBALANCE HANDLING COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("IMBALANCE HANDLING COMPARISON")
print("=" * 70)

# We use Logistic Regression for this comparison.

# ------------------------------------------------------------
# A. Baseline
# ------------------------------------------------------------

baseline_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

baseline_pipeline.fit(
    X_train,
    y_train
)

baseline_pred = baseline_pipeline.predict(X_test)

baseline_metrics = {
    "Strategy": "Baseline",
    "Precision": precision_score(
        y_test,
        baseline_pred,
        zero_division=0
    ),
    "Recall": recall_score(
        y_test,
        baseline_pred,
        zero_division=0
    ),
    "F1": f1_score(
        y_test,
        baseline_pred,
        zero_division=0
    )
}


# ------------------------------------------------------------
# B. class_weight='balanced'
# ------------------------------------------------------------

balanced_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        )
    )
])

balanced_pipeline.fit(
    X_train,
    y_train
)

balanced_pred = balanced_pipeline.predict(X_test)

balanced_metrics = {
    "Strategy": "class_weight='balanced'",
    "Precision": precision_score(
        y_test,
        balanced_pred,
        zero_division=0
    ),
    "Recall": recall_score(
        y_test,
        balanced_pred,
        zero_division=0
    ),
    "F1": f1_score(
        y_test,
        balanced_pred,
        zero_division=0
    )
}


# ------------------------------------------------------------
# C. SMOTE
# IMPORTANT:
# SMOTE is applied only after preprocessing and only on X_train.
# Test data is NEVER oversampled.
# ------------------------------------------------------------

smote_pipeline = ImbPipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "smote",
        SMOTE(
            random_state=42
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

smote_pipeline.fit(
    X_train,
    y_train
)

smote_pred = smote_pipeline.predict(X_test)

smote_metrics = {
    "Strategy": "SMOTE",
    "Precision": precision_score(
        y_test,
        smote_pred,
        zero_division=0
    ),
    "Recall": recall_score(
        y_test,
        smote_pred,
        zero_division=0
    ),
    "F1": f1_score(
        y_test,
        smote_pred,
        zero_division=0
    )
}


imbalance_results = pd.DataFrame([
    baseline_metrics,
    balanced_metrics,
    smote_metrics
])

print(
    imbalance_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

imbalance_results.to_csv(
    "imbalance_comparison.csv",
    index=False
)


# ============================================================
# 17. AUTOMATIC IMBALANCE CONCLUSION
# ============================================================

best_imbalance_row = imbalance_results.loc[
    imbalance_results["F1"].idxmax()
]

print("\nImbalance conclusion:")

print(
    f"The {best_imbalance_row['Strategy']} strategy produced "
    f"the highest F1 score of {best_imbalance_row['F1']:.4f} "
    f"among the three tested strategies. "
    f"Its precision was {best_imbalance_row['Precision']:.4f} "
    f"and recall was {best_imbalance_row['Recall']:.4f}. "
    f"The choice is based on F1 because it balances precision and recall. "
    f"SMOTE was applied only to the training data to prevent test-set leakage."
)


# ============================================================
# 18. RANDOM FOREST GRID SEARCH
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST GRID SEARCH")
print("=" * 70)

rf_for_grid = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        RandomForestClassifier(
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )
    )
])

param_grid = {
    "classifier__n_estimators": [
        100,
        200
    ],
    "classifier__max_depth": [
        None,
        5,
        10
    ],
    "classifier__max_features": [
        "sqrt",
        "log2"
    ]
}

grid_search = GridSearchCV(
    estimator=rf_for_grid,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    refit=True
)

grid_search.fit(
    X_train,
    y_train
)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest CV F1 Score:")
print(f"{grid_search.best_score_:.4f}")

best_rf_pipeline = grid_search.best_estimator_

best_rf_estimator = (
    best_rf_pipeline
    .named_steps["classifier"]
)

print("\nOOB Score:")
print(f"{best_rf_estimator.oob_score_:.4f}")


# ============================================================
# 19. EVALUATE TUNED RANDOM FOREST
# ============================================================

tuned_rf_pred = best_rf_pipeline.predict(X_test)

tuned_rf_prob = best_rf_pipeline.predict_proba(
    X_test
)[:, 1]

tuned_rf_accuracy = accuracy_score(
    y_test,
    tuned_rf_pred
)

tuned_rf_precision = precision_score(
    y_test,
    tuned_rf_pred,
    zero_division=0
)

tuned_rf_recall = recall_score(
    y_test,
    tuned_rf_pred,
    zero_division=0
)

tuned_rf_f1 = f1_score(
    y_test,
    tuned_rf_pred,
    zero_division=0
)

tuned_rf_auc = roc_auc_score(
    y_test,
    tuned_rf_prob
)

print("\nTuned Random Forest Test Metrics:")
print(f"Accuracy : {tuned_rf_accuracy:.4f}")
print(f"Precision: {tuned_rf_precision:.4f}")
print(f"Recall   : {tuned_rf_recall:.4f}")
print(f"F1       : {tuned_rf_f1:.4f}")
print(f"AUC      : {tuned_rf_auc:.4f}")


# ============================================================
# 20. REGRESSION SIDE TASK - PREDICT FARE
# ============================================================

print("\n" + "=" * 70)
print("REGRESSION: PREDICT FARE")
print("=" * 70)

regression_features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "embarked"
]

reg_df = df[
    regression_features + ["fare"]
].copy()

# Remove rows with missing target fare.
reg_df = reg_df.dropna(
    subset=["fare"]
)

X_reg = reg_df[regression_features]
y_reg = reg_df["fare"]


# ------------------------------------------------------------
# Regression train/test split
# ------------------------------------------------------------

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)


# ------------------------------------------------------------
# Regression preprocessing
# ------------------------------------------------------------

reg_numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch"
]

reg_categorical_features = [
    "sex",
    "embarked"
]

reg_numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])

reg_categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])

reg_preprocessor = ColumnTransformer([
    (
        "numeric",
        reg_numeric_pipeline,
        reg_numeric_features
    ),
    (
        "categorical",
        reg_categorical_pipeline,
        reg_categorical_features
    )
])


regression_pipeline = Pipeline([
    (
        "preprocessor",
        reg_preprocessor
    ),
    (
        "regressor",
        LinearRegression()
    )
])


regression_pipeline.fit(
    X_reg_train,
    y_reg_train
)

y_reg_pred = regression_pipeline.predict(
    X_reg_test
)


# ============================================================
# 21. REGRESSION METRICS
# ============================================================

mae = mean_absolute_error(
    y_reg_test,
    y_reg_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        y_reg_pred
    )
)

r2 = r2_score(
    y_reg_test,
    y_reg_pred
)

# Number of observations
n = len(y_reg_test)

# Number of predictors after encoding
reg_feature_count = (
    regression_pipeline
    .named_steps["preprocessor"]
    .transform(X_reg_test)
    .shape[1]
)

p = reg_feature_count

if n - p - 1 > 0:
    adjusted_r2 = (
        1
        - (
            (1 - r2) * (n - 1)
            / (n - p - 1)
        )
    )
else:
    adjusted_r2 = np.nan

print(f"MAE        : {mae:.4f}")
print(f"RMSE       : {rmse:.4f}")
print(f"R²         : {r2:.4f}")
print(f"Adjusted R²: {adjusted_r2:.4f}")


# ============================================================
# 22. RESIDUAL PLOT
# ============================================================

residuals = y_reg_test - y_reg_pred

plt.figure(figsize=(8, 6))

plt.scatter(
    y_reg_pred,
    residuals,
    alpha=0.6
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Predicted Fare")
plt.ylabel("Residual")
plt.title("Fare Regression Residual Plot")

plt.grid(alpha=0.3)

plt.savefig(
    "charts/fare_residual_plot.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 23. HETEROSCEDASTICITY CONCLUSION
# ============================================================

# Simple visual diagnostic:
# compare residual spread in lower and upper prediction ranges.

prediction_median = np.median(y_reg_pred)

low_residuals = residuals[
    y_reg_pred <= prediction_median
]

high_residuals = residuals[
    y_reg_pred > prediction_median
]

low_std = np.std(low_residuals)
high_std = np.std(high_residuals)

spread_ratio = (
    max(low_std, high_std)
    / max(min(low_std, high_std), 1e-9)
)

print("\nHeteroscedasticity assessment:")

if spread_ratio > 1.5:
    hetero_conclusion = (
        "The residual plot suggests heteroscedasticity because "
        "the residual spread changes noticeably across the range "
        "of predicted fares."
    )
else:
    hetero_conclusion = (
        "The residual plot does not show strong evidence of "
        "heteroscedasticity based on the visual spread of residuals."
    )

print(hetero_conclusion)


# ============================================================
# 24. FINAL MODEL COMPARISON TABLE
# CLASSIFICATION AND REGRESSION ARE SEPARATE METRIC GROUPS
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

final_comparison = classification_results.copy()

# Add separate regression metric columns.
final_comparison["Regression MAE"] = np.nan
final_comparison["Regression RMSE"] = np.nan
final_comparison["Regression R2"] = np.nan
final_comparison["Regression Adjusted R2"] = np.nan

# Put regression values in a separate row.
regression_row = pd.DataFrame([{
    "Model": "Multivariate Linear Regression",
    "Accuracy": np.nan,
    "Precision": np.nan,
    "Recall": np.nan,
    "F1": np.nan,
    "AUC": np.nan,
    "Regression MAE": mae,
    "Regression RMSE": rmse,
    "Regression R2": r2,
    "Regression Adjusted R2": adjusted_r2
}])

final_comparison = pd.concat(
    [
        final_comparison,
        regression_row
    ],
    ignore_index=True
)

print(
    final_comparison.to_string(
        index=False,
        float_format=lambda x: (
            f"{x:.4f}" if pd.notna(x) else ""
        )
    )
)

final_comparison.to_csv(
    "final_model_comparison.csv",
    index=False
)


# ============================================================
# 25. FINAL CLASSIFIER SELECTION
# ============================================================
# Selection criterion:
# highest F1 among the three classifiers.
# We report the values rather than using a subjective rating.

best_classifier_row = classification_results.loc[
    classification_results["F1"].idxmax()
]

best_classifier_name = best_classifier_row["Model"]

print("\n" + "=" * 70)
print("FINAL WRITTEN RECOMMENDATION")
print("=" * 70)

recommendation = (
    f"Based on the held-out test set, {best_classifier_name} "
    f"has the highest F1 score among the three tested classifiers, "
    f"with an F1 of {best_classifier_row['F1']:.4f}. "
    f"Its accuracy is {best_classifier_row['Accuracy']:.4f}, "
    f"precision is {best_classifier_row['Precision']:.4f}, "
    f"recall is {best_classifier_row['Recall']:.4f}, "
    f"and AUC is {best_classifier_row['AUC']:.4f}. "
    f"These metrics indicate how the model balances correct "
    f"classification, positive-class identification, and ranking "
    f"performance on the held-out data. "
    f"The regression model is evaluated separately using MAE, RMSE, "
    f"R², and Adjusted R² because regression and classification metrics "
    f"are on different scales and are not directly comparable."
)

print(recommendation)


# ============================================================
# 26. SAVE THE BEST COMPLETE PIPELINE
# ============================================================
# IMPORTANT:
# Save the COMPLETE pipeline, not just the estimator.
#
# The pipeline contains:
# preprocessing
#     -> imputation
#     -> encoding
#     -> scaling
#     -> final classifier
# ============================================================

# We choose the best classifier according to test-set F1.
# To avoid using an incompatible fitted object, construct a fresh
# complete pipeline of the selected model and fit it on training data.

if best_classifier_name == "Logistic Regression":

    final_pipeline = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])

elif best_classifier_name == "Decision Tree":

    final_pipeline = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            DecisionTreeClassifier(
                max_depth=5,
                random_state=42
            )
        )
    ])

else:

    final_pipeline = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )
        )
    ])


final_pipeline.fit(
    X_train,
    y_train
)


# Save complete pipeline.
PIPELINE_PATH = "best_titanic_pipeline.joblib"

joblib.dump(
    final_pipeline,
    PIPELINE_PATH
)

print("\nSaved complete pipeline:")
print(PIPELINE_PATH)


# ============================================================
# 27. RELOAD SAVED PIPELINE
# ============================================================

loaded_pipeline = joblib.load(
    PIPELINE_PATH
)

print("\nPipeline successfully reloaded:")
print(loaded_pipeline)


# ============================================================
# 28. TEST RELOADED PIPELINE ON RAW INPUT
# ============================================================

raw_new_data = pd.DataFrame({
    "pclass": [3],
    "sex": ["male"],
    "age": [25],
    "sibsp": [0],
    "parch": [0],
    "fare": [10.0],
    "embarked": ["S"]
})

raw_prediction = loaded_pipeline.predict(
    raw_new_data
)

raw_probability = loaded_pipeline.predict_proba(
    raw_new_data
)[:, 1]

print("\n" + "=" * 70)
print("RELOADED PIPELINE RAW INPUT TEST")
print("=" * 70)

print("\nRaw input:")
print(raw_new_data)

print("\nPrediction:", raw_prediction[0])
print(
    "Predicted class:",
    "Survived" if raw_prediction[0] == 1
    else "Not Survived"
)

print(
    f"Survival probability: {raw_probability[0]:.4f}"
)

print(
    "\nSUCCESS: The saved artifact accepts raw, "
    "unpreprocessed input and performs preprocessing + prediction end-to-end."
)


# ============================================================
# 29. SAVE MODELING README
# ============================================================

readme_content = f"""
# Analytics Module - Part B: Predictive Modeling

## Dataset

The modeling stage loads `titanic.csv`, which was created during Part A.
The raw Seaborn Titanic dataset is not independently reloaded here.

## Class Balance

The classification target is `survived`.

The classes are not perfectly balanced, so a stratified train/test split
was used. Stratification preserves approximately the same class proportions
in both training and testing sets.

## Preprocessing

Numeric features:
- pclass
- age
- sibsp
- parch
- fare

Numeric preprocessing:
- Median imputation
- StandardScaler

Categorical features:
- sex
- embarked

Categorical preprocessing:
- Most-frequent imputation
- OneHotEncoder(handle_unknown='ignore')

All preprocessing is contained inside a ColumnTransformer and Pipeline.
The preprocessing is fitted only on the training split and then used to
transform the test split. No preprocessing step is fitted on the test data.

Missing-value threshold rule:
- 0% to <5%: simple imputation is generally sufficient.
- 5% to <30%: defensible statistical imputation is appropriate.
- >=30%: investigate whether the feature should be retained.

## Classifier Results

{classification_results.to_string(index=False)}

## Imbalance Handling

{imbalance_results.to_string(index=False)}

Conclusion:
The {best_imbalance_row['Strategy']} strategy produced the highest F1 score
among the tested imbalance strategies, with F1 =
{best_imbalance_row['F1']:.4f}. SMOTE was applied only to the training fold.

## Random Forest Grid Search

Best parameters:

{grid_search.best_params_}

Best cross-validation F1:
{grid_search.best_score_:.4f}

OOB score:
{best_rf_estimator.oob_score_:.4f}

The Random Forest was constructed with `oob_score=True`, so the OOB score
is available from the fitted estimator.

## Regression

The regression task predicts `fare` from the remaining available features.

MAE: {mae:.4f}

RMSE: {rmse:.4f}

R2: {r2:.4f}

Adjusted R2: {adjusted_r2:.4f}

Heteroscedasticity conclusion:
{hetero_conclusion}

## Final Model Comparison

{final_comparison.to_string(index=False)}

Classification metrics and regression metrics are presented as separate
metric groups because they measure different tasks and are not directly
comparable.

## Final Recommendation

{recommendation}

## Saved Artifact

The complete fitted preprocessing + classifier pipeline is saved as:

`best_titanic_pipeline.joblib`

The artifact was reloaded with `joblib.load()` and successfully tested on
raw, unpreprocessed input.
"""

with open(
    "README.md",
    "w",
    encoding="utf-8"
) as f:
    f.write(readme_content)


# ============================================================
# 30. FINAL FILE CHECK
# ============================================================

print("\n" + "=" * 70)
print("PART B COMPLETE")
print("=" * 70)

required_outputs = [
    "titanic.csv",
    "classifier_comparison.csv",
    "imbalance_comparison.csv",
    "final_model_comparison.csv",
    "best_titanic_pipeline.joblib",
    "README.md",
    "charts/confusion_matrices.png",
    "charts/roc_curves.png",
    "charts/decision_tree.png",
    "charts/fare_residual_plot.png"
]

print("\nRequired output files:")

for file in required_outputs:
    print(
        f"[{'OK' if os.path.exists(file) else 'MISSING'}] {file}"
    )

print("\nPart B execution finished successfully.")