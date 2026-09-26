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