
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

              Model  Accuracy  Precision   Recall       F1      AUC
Logistic Regression  0.808989   0.783333 0.691176 0.734375 0.860963
      Decision Tree  0.764045   0.760000 0.558824 0.644068 0.837366
      Random Forest  0.808989   0.765625 0.720588 0.742424 0.819586

## Imbalance Handling

               Strategy  Precision   Recall       F1
               Baseline   0.783333 0.691176 0.734375
class_weight='balanced'   0.718310 0.750000 0.733813
                  SMOTE   0.735294 0.735294 0.735294

Conclusion:
The SMOTE strategy produced the highest F1 score
among the tested imbalance strategies, with F1 =
0.7353. SMOTE was applied only to the training fold.

## Random Forest Grid Search

Best parameters:

{'classifier__max_depth': 5, 'classifier__max_features': 'sqrt', 'classifier__n_estimators': 200}

Best cross-validation F1:
0.7408

OOB score:
0.8214

The Random Forest was constructed with `oob_score=True`, so the OOB score
is available from the fitted estimator.

## Regression

The regression task predicts `fare` from the remaining available features.

MAE: 21.1386

RMSE: 41.7465

R2: 0.3468

Adjusted R2: 0.3118

Heteroscedasticity conclusion:
The residual plot suggests heteroscedasticity because the residual spread changes noticeably across the range of predicted fares.

## Final Model Comparison

                         Model  Accuracy  Precision   Recall       F1      AUC  Regression MAE  Regression RMSE  Regression R2  Regression Adjusted R2
           Logistic Regression  0.808989   0.783333 0.691176 0.734375 0.860963             NaN              NaN            NaN                     NaN
                 Decision Tree  0.764045   0.760000 0.558824 0.644068 0.837366             NaN              NaN            NaN                     NaN
                 Random Forest  0.808989   0.765625 0.720588 0.742424 0.819586             NaN              NaN            NaN                     NaN
Multivariate Linear Regression       NaN        NaN      NaN      NaN      NaN       21.138552        41.746502       0.346774                 0.31178

Classification metrics and regression metrics are presented as separate
metric groups because they measure different tasks and are not directly
comparable.

## Final Recommendation

Based on the held-out test set, Random Forest has the highest F1 score among the three tested classifiers, with an F1 of 0.7424. Its accuracy is 0.8090, precision is 0.7656, recall is 0.7206, and AUC is 0.8196. These metrics indicate how the model balances correct classification, positive-class identification, and ranking performance on the held-out data. The regression model is evaluated separately using MAE, RMSE, R², and Adjusted R² because regression and classification metrics are on different scales and are not directly comparable.

## Saved Artifact

The complete fitted preprocessing + classifier pipeline is saved as:

`best_titanic_pipeline.joblib`

The artifact was reloaded with `joblib.load()` and successfully tested on
raw, unpreprocessed input.
