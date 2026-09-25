Analytics Pipeline
==================

This module loads the Titanic dataset once, cleans it, explores it
with charts, and builds a full modeling pipeline on the same cleaned
data.

The dataset is loaded once with sns.load_dataset and saved immediately
as titanic.csv. Every later step reads from that same CSV. The
modeling script never calls the seaborn loader again.

Install
-------

    pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn joblib

Run
---

    cd analytics
    python 01_eda.py
    python 02_modeling.py

Files
-----

01_eda.py        loads, cleans, runs EDA, saves titanic.csv and charts
02_modeling.py   reads titanic_clean.csv, trains and evaluates models
titanic.csv      committed offline fallback
titanic_clean.csv  cleaned data used by modeling

Missing values
--------------

I measured the missing percentage in each affected column and applied
this rule:

    deck      77.22 percent  - dropped the column, too many missing
    age       19.87 percent  - imputed with the median
    embarked   0.22 percent  - dropped those rows

Univariate
----------

Fare stats:

    mean   32.10
    median 14.45
    mode    8.05

Mean is greater than median and median is greater than mode, so the
fare distribution is right-skewed.

IQR outliers: age had 86 and fare had 114.

Bivariate
---------

Survival rate by sex:

    female 74.20 percent
    male   18.89 percent

Survival rate by pclass:

    1st 62.96 percent
    2nd 47.28 percent
    3rd 24.24 percent

Correlation matrix
------------------

I used exactly six columns: survived, pclass, age, sibsp, parch,
fare. I left out adult_male and alone because they are derived
flags, not independent features.

The two strongest off-diagonal correlations were pclass and fare
(about -0.55, negative) and sibsp and parch (about 0.41, positive).
Higher class numbers mean cheaper tickets, and passengers travelling
with a spouse often also travelled with a child.

Charts
------

Four charts together tell the survival story:

    chart3  survival rate by sex and pclass
    chart4  age distribution by survival
    chart5  fare versus age colored by survival
    chart6  pair plot of survived, age, fare, pclass

Modeling
--------

Stratified split first because survived is about 38/62. Preprocessing
imputes, encodes, and scales inside a ColumnTransformer wrapped in a
Pipeline so it is fit on train only.

Classifier results on the test split:

    Model                 Accuracy  Precision  Recall  F1      AUC
    Logistic Regression   0.8146    0.7966     0.6912  0.7402  0.8596
    Decision Tree         0.8034    0.8000     0.6471  0.7154  0.8481
    Random Forest         0.8202    0.7812     0.7353  0.7576  0.8178

Imbalance
---------

Compared baseline, class_weight equal to balanced, and SMOTE applied
only to the training fold. class_weight equal to balanced gave the
best precision and recall trade-off. SMOTE helped recall but added
false positives.

Regression side task
--------------------

Linear regression predicts fare. Metrics are printed by
02_modeling.py. The residual plot in chart10_residuals.png shows the
residual spread widening with predicted fare, so there is
heteroscedasticity.

Recommendation
--------------

Deploy the Random Forest. It has the highest accuracy and F1.
Logistic Regression has the highest AUC, so it is the better choice
if ranking matters more than hard labels.

Saved artifact
--------------

final_pipeline.joblib contains the full pipeline: the
ColumnTransformer plus the best estimator. It is saved with
joblib.dump and reloaded with joblib.load at the end of the script.