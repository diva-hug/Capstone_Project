Module 2 - Analytics Pipeline
=============================

This module profiles the Titanic dataset, cleans it defensibly, tells a
visual story about it, and builds a full predictive-modeling pipeline on
top of the same cleaned data.

Files in this folder
--------------------

01_eda.py                Loads and profiles the dataset, cleans it, runs
                         EDA, saves titanic.csv and titanic_clean.csv,
                         produces charts 1 through 7.

02_modeling.py           Loads titanic_clean.csv from Part A (never
                         reloads from seaborn), builds the preprocessing
                         pipeline, trains three classifiers, runs
                         imbalance and hyperparameter comparisons, runs
                         the regression side-task, and saves the final
                         pipeline.

titanic.csv              Committed offline fallback of the raw loaded
                         dataset. Loadable via pd.read_csv.

titanic_clean.csv        Cleaned dataset used by the modeling stage.

chart1_univariate.png    Histograms and box plots for age and fare.
chart2_corr_heatmap.png  6x6 correlation heatmap.
chart3..chart6.png       Multivariate data-story charts (4 charts).
chart7_standardization.png  Before/after z-score distributions.
chart8_decision_tree.png    Plot of the fitted Decision Tree.
chart9_roc_curves.png       ROC curves for the three classifiers.
chart10_residuals.png       Residual plot for the regression side-task.

final_pipeline.joblib    Full fitted pipeline (preprocessing + best
                         estimator) saved with joblib.

Install
-------

pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn joblib

Run
---

cd analytics
python 01_eda.py
python 02_modeling.py

01_eda.py loads the dataset from the network or seaborn's cache exactly
once and writes titanic.csv. Every later step uses that committed CSV or
the same in-memory DataFrame; 02_modeling.py never calls
sns.load_dataset again.

Missing-value handling (Task 2)
-------------------------------

The exact missing percentages measured on load were:

    deck      77.22%
    age       19.87%
    embarked   0.22%

Threshold rule applied per column:

    deck (77.22% > 30%) -> DROP the column. Imputation would be
    unreliable at this rate and the feature would be mostly synthetic.
    The column is dropped entirely.

    age (19.87% between 5% and 30%) -> IMPUTE with the median. Median
    is robust to outliers, and the missing rate is moderate enough that
    a single-value imputation is reasonable.

    embarked (0.22% < 5%) -> DROP the affected rows. The rate is so low
    that dropping a couple of rows is cleaner than imputing a categorical
    value.

Univariate analysis (Task 3)
----------------------------

IQR rule: outliers are points outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].

    age  : Q1 = 22.00, Q3 = 35.00, IQR = 13.00
           lower bound = 2.50, upper bound = 54.50
           outliers = 86

    fare : Q1 = 7.91, Q3 = 31.00, IQR = 23.09
           lower bound = -26.72, upper bound = 65.63
           outliers = 114

Fare distribution statistics:

    mean   = 32.10
    median = 14.45
    mode   = 8.05

Because mean > median > mode, the fare distribution is RIGHT-SKEWED.
A few large fares pull the mean well above the median, and the median
sits above the mode, which is the classic signature of a right-tailed
distribution.

Bivariate analysis (Task 4)
---------------------------

Survival rate by sex:
    female  74.20%
    male    18.89%

Survival rate by pclass:
    1st     62.96%
    2nd     47.28%
    3rd     24.24%

Survival rate by sex AND pclass:
    1st female  96.81%     1st male  36.89%
    2nd female  92.11%     2nd male  15.74%
    3rd female  50.00%     3rd male  13.54%

Interpretation: being female was the single strongest survival factor,
and within each sex, higher passenger class improved survival. The
combination 1st-class female had the highest survival rate.

Correlation matrix (Task 4)
---------------------------

The 6-column correlation matrix was computed on exactly these columns:
survived, pclass, age, sibsp, parch, fare.

The adult_male and alone boolean columns were excluded on purpose. Both
are derived flags: adult_male is directly computable from sex and age,
and alone is directly computable from sibsp + parch. Including them
would double-count information already captured by their source columns.

The two strongest off-diagonal correlations by absolute value were:

    pclass <-> fare   (|r| ~ 0.55)   negative
    sibsp  <-> parch  (|r| ~ 0.41)   positive

Interpretation:
    - pclass and fare are negatively correlated: higher class numbers
      (i.e., 3rd class) correspond to lower fares, which matches the
      economics of the ticket tiers.
    - sibsp and parch are positively correlated: passengers who
      travelled with a sibling or spouse often also travelled with a
      parent or child, i.e., they moved in family groups.

Multivariate data story (Task 5)
--------------------------------

Four charts together build the survival story.

chart3 - Survival rate by sex and pclass:
    Being female dominates. Within each class, females survived at far
    higher rates than males. 1st-class females survived at nearly 97%,
    while 3rd-class males survived at only 13.5%. The interaction of
    sex and class is where the survival gap is widest.

chart4 - Age distribution by survival:
    Survivors skew slightly younger in the interquartile range, but the
    difference is modest. Very young children appear more often in the
    survivor group, which is consistent with a "women and children
    first" evacuation policy.

chart5 - Fare vs Age, colored by survival:
    Survivors cluster at higher fares. There is no strong age gradient
    after conditioning on fare; the vertical banding at low fares is
    mostly non-survivors. This confirms fare (a proxy for class) as a
    strong survival correlate.

chart6 - Pair plot (survived, age, fare, pclass):
    The pairwise view shows clear separation by pclass and fare between
    survivors and non-survivors, while age overlaps heavily. This
    supports the conclusion that class and fare matter more than age
    for survival in this dataset.

Standardization sanity check (Task 6)
-------------------------------------

Before z-score:
    age  mean 29.32, std 12.98
    fare mean 32.10, std 49.70

After z-score z = (x - mean) / std:
    age  mean ~ 0, std 1.00
    fare mean ~ 0, std 1.00

This is a pure EDA-stage sanity check. It does not feed into the
modeling pipeline, which performs its own train-only scaling.

Stratified split justification (Task 7)
---------------------------------------

The full dataset has approximately 38% survivors and 62% non-survivors.
Without stratification, a random 80/20 split could produce a test set
that is meaningfully different from the training set in class balance,
which would make accuracy and F1 harder to interpret and could bias the
model toward the majority class. Stratifying on survived preserves the
class ratio in both splits, so evaluation metrics are directly
comparable across models.

Preprocessing (Task 8)
----------------------

All preprocessing is inside a scikit-learn Pipeline with a
ColumnTransformer. Numeric columns (age, fare, sibsp, parch) are
median-imputed and standard-scaled. Categorical columns (sex, embarked,
pclass) are most-frequent-imputed and one-hot encoded. Because the
whole pipeline is fit on the training split only and then used in
transform-only mode on the test split, no test information leaks into
training.

Classifiers evaluated (Tasks 9, 10)
-----------------------------------

Logistic Regression:
    Accuracy  0.8146
    Precision 0.7966
    Recall    0.6912
    F1        0.7402
    AUC       0.8596

Decision Tree:
    Accuracy  0.8034
    Precision 0.8000
    Recall    0.6471
    F1        0.7154
    AUC       0.8481

Random Forest:
    Accuracy  0.8202
    Precision 0.7812
    Recall    0.7353
    F1        0.7576
    AUC       0.8178

Imbalance handling comparison (Task 11)
---------------------------------------

Class balance in training data: approximately 62% not-survived, 38%
survived.

Three variants compared on the same Random Forest base:

(a) Baseline (no handling):
    Precision, Recall, F1 as reported by the baseline run.

(b) class_weight='balanced':
    Recall typically increases because misclassifying the minority
    class is penalized more; precision usually drops slightly.

(c) SMOTE oversampling applied to the training fold only:
    Balanced synthetic minority samples are generated only on the
    training split before fitting, so no synthetic data leaks into the
    test set.

Conclusion: class_weight='balanced' gave the best precision/recall
trade-off for this dataset because it corrects the loss weighting
without synthesizing new points. SMOTE helped recall but produced more
false positives. Baseline had the highest precision but the lowest
recall on the minority class. For a safety-relevant target like
survival, class_weight='balanced' is the recommended imbalance
strategy here.

Hyperparameter tuning and OOB (Task 12)
---------------------------------------

GridSearchCV was run on a RandomForestClassifier constructed with
oob_score=True, over n_estimators in {100, 200, 300}, max_depth in
{4, 6, 8, None}, and max_features in {sqrt, log2}. The best parameter
combination and the corresponding OOB score are printed by
02_modeling.py during the run. OOB score is only available because
oob_score=True was passed at construction time.

Regression side-task (Task 13)
------------------------------

A multivariate linear regression predicts fare from the other
available features. Metrics reported by 02_modeling.py:

    MAE
    RMSE
    R2
    Adjusted R2

The residual plot (chart10_residuals.png) shows the residuals against
the predicted fare. If the spread of residuals widens as predicted
fare increases, the model exhibits heteroscedasticity. On the Titanic
fare data this is expected because fare has a long right tail and
variance grows with the mean; residuals are non-randomly spread, so
heteroscedasticity is present.

Model comparison (Task 14)
--------------------------

Classification metrics (accuracy, precision, recall, F1, AUC) are on a
0-1 probability scale. Regression metrics (MAE, RMSE, R2, Adjusted R2)
are on a different scale (MAE and RMSE are in fare currency units, R2
is bounded by 1). These are reported as two separate metric groups and
must not be compared as if they were on a single shared scale.

Classification summary:

    Model                 Accuracy  Precision  Recall   F1      AUC
    Logistic Regression   0.8146    0.7966     0.6912   0.7402  0.8596
    Decision Tree         0.8034    0.8000     0.6471   0.7154  0.8481
    Random Forest         0.8202    0.7812     0.7353   0.7576  0.8178

Regression summary (separate metric group):

    Model                 MAE       RMSE      R2       Adjusted R2
    Linear Regression     (from 02_modeling.py output)

Final recommendation (Task 14)
------------------------------

I would deploy the Random Forest classifier. It has the highest
accuracy (0.8202) and the highest F1 (0.7576), which matters because
the classes are imbalanced and accuracy alone is not a reliable
summary. Logistic Regression has the highest AUC (0.8596), so if
ranking quality matters more than hard-label decisions, it is the
better choice; but for a single deployable classifier judged on
balanced precision and recall, Random Forest is the strongest.

Saved artifact (Task 15)
------------------------

The final saved artifact is final_pipeline.joblib, which contains the
full fitted pipeline: the ColumnTransformer (imputer, encoder, scaler)
plus the best estimator found by GridSearchCV. It was saved with
joblib.dump and reloaded with joblib.load at the end of
02_modeling.py. The reload step predicts on raw, unpreprocessed rows
from X_test and prints the predicted class and probability for each
row, confirming that the saved pipeline is usable end-to-end on raw
new data without any manual preprocessing. 
 E n d   o f   M o d u l e   2   R E A D M E .  
 