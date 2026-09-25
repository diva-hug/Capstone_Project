# Analytic pipline Part B 

import pandas as pd
import numpy as np
import seaborn as sns
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split ,GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import(confusion_matrix,accuracy_score,precision_score,recall_score,
                            f1_score,roc_auc_score,roc_curve,mean_absolute_error,mean_squared_error,r2_score,)
from imblearn.over_sampling import SMOTE

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"]=100

print("="*70)
print("Load Cleaned data set from Part A")
print("="*70)

df=pd.read_csv("titanic_clean.csv")
print(f"Shape:{df.shape}")
print(f"Columns:{df.columns.tolist()}")

print("\n" + "=" * 70)
print("TASK 7:  Split the data into train/test")
print("="*70)

print("\n class balance in full dataset:")
print(df["survived"].value_counts(normalize=True))

X=df.drop(columns=["survived"])
y=df["survived"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,stratify=y,random_state=42)

print(f"\n Train size : {len(X_train)}, Test size : {len(X_test)}")
print(f"\n Train class balance :\n {y_train.value_counts(normalize=True)}")
print(f"\n Test class balance :\n {y_test.value_counts(normalize=True)}")

print("\n" + "=" * 70)
print("TASK 8 — Preprocessing Pipeline")
print("="*70)

numeric_features = ["age", "fare", "sibsp", "parch"]
categorical_features = ["sex","embarked", "pclass"]

numeric_transformer =Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler()),

])

categorical_transformer = Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("onehot",OneHotEncoder(handle_unknown="ignore")),
])

preprocessor =ColumnTransformer(transformers=[
    ("num",numeric_transformer,numeric_features),
    ("cat",categorical_transformer,categorical_features),
])

print("Preprocessor built. fit will happen only on training split")

print("\n" + "=" * 70)
print("TASK 9 and 10 — Train 3 Classifiers + Evaluate")
print("="*70)


def evaluate_classifier(name,pipeline,X_train,y_train,X_test,y_test):
    pipeline.fit(X_train,y_train)
    y_pred=pipeline.predict(X_test)
    y_prob =pipeline.predict_proba(X_test)[:,1]

    cm =confusion_matrix(y_test,y_pred)
    acc=accuracy_score(y_test,y_pred)
    prec=precision_score(y_test,y_pred)
    rec=recall_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    auc=roc_auc_score(y_test,y_prob)

    print(f"\n--- {name} ---")
    print(f"\n confusion_matrix:\n{cm}")
    print(f"Accuracy:{acc:.4f}")
    print(f"Precision_score:{prec:.4f}")
    print(f"recall_Score:{rec:.4f}")
    print(f"f1_Score:{f1:.4f}")
    print(f"roc_auc: {auc:.4f}")

    return {
        "name": name,
        "pipeline": pipeline,
        "confusion_matrix": cm,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }

lr_pipeline=Pipeline(steps=[
    ("Preprocessor",preprocessor),
    ("clf",LogisticRegression(max_iter=1000,random_state=42)),
])

lr_result=evaluate_classifier("Logistic Regression",lr_pipeline,
                              X_train,y_train,X_test,y_test)

dt_pipeline=Pipeline(steps=[
    ("preprocess",preprocessor),
    ("clf",DecisionTreeClassifier(max_depth=4,random_state=42)),
])

dt_result =evaluate_classifier("Decision Tree",dt_pipeline,X_train,y_train,X_test,y_test)

rf_pipeline=Pipeline(steps=[
    ("preprocess",preprocessor),
    ("clf",RandomForestClassifier(n_estimators=200,random_state=42)),
])

rf_result = evaluate_classifier("Random Forest",rf_pipeline,X_train,y_train,X_test,y_test)

# Decision Tree  Visualization
print("\nRendering decision tree...")
dt_pipeline.fit(X_train, y_train)
feature_names = (
    numeric_features
    + list(dt_pipeline.named_steps["preprocess"]
           .named_transformers_["cat"]
           .named_steps["onehot"]
           .get_feature_names_out(categorical_features))
)

plt.figure(figsize=(20,10))
plot_tree(
    dt_pipeline.named_steps["clf"],
    feature_names=feature_names,
    class_names=["Not Survived", "Survived"],
    filled=True,rounded=True,fontsize=8,
)


plt.tight_layout()
plt.savefig("chart8_decision_tree.png")
plt.close()
print(" saved chart8_decision_tree.png")


# Roc Curves 

plt.figure(figsize=(8, 6))
for res in [lr_result, dt_result, rf_result]:
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    plt.plot(fpr, tpr, label=f"{res['name']} (AUC = {res['auc']:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.xlabel("False postive rate")
plt.ylabel("True postive rate")
plt.title("ROC Curves - 3 Classifiers")
plt.legend()
plt.tight_layout()
plt.savefig("chart9_roc_curves.png")
plt.close()
print("Saved chart9_roc_curves.png")


print("\n" + "=" * 70)
print("TASK 11: Imbalance handling comparison")
print("=" * 70)


print("\n class balance in training set :")
print(y_train.value_counts(normalize=True))

#  A:---baseline
base_pipeline =Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
])

base_pipeline.fit(X_train,y_train)
y_pred_base=base_pipeline.predict(X_test)
print("\n (A) baseline :")
print(f"Precision: {precision_score(y_test, y_pred_base):.4f}")
print(f" Recall : {recall_score(y_test, y_pred_base):.4f}")
print(f"F1 score: {f1_score(y_test, y_pred_base):.4f}")

# class weight =Balanced

balanced_pipeline =Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)),

])

balanced_pipeline.fit(X_train,y_train)
y_pred_bal =balanced_pipeline.predict(X_test)
print("\n(B) class_weight = balanced")
print(f" Precision:{precision_score(y_test, y_pred_bal):.4f}")
print(f" Recall:{recall_score(y_test, y_pred_bal):.4f}")
print(f"F1 : {f1_score(y_test, y_pred_bal):.4f}")

#(c) SMOTE oversampling applied only to the training fold 
x_train_enc= preprocessor.fit_transform(X_train)
x_test_enc=preprocessor.transform(X_test)

smote =SMOTE(random_state=42)
x_train_smote,y_train_smote =smote.fit_resample(x_train_enc,y_train)

rf_smote =RandomForestClassifier(n_estimators=200,random_state=42)
rf_smote.fit(x_train_smote,y_train_smote)
y_pred_smote=rf_smote.predict(x_test_enc)

print("\n (c) SMOTE (training fold only):")

print(f"Precision : {precision_score(y_test, y_pred_smote):.4f}")
print(f" Recall  : {recall_score(y_test, y_pred_smote):.4f}")
print(f" F1  : {f1_score(y_test, y_pred_smote):.4f}")


print("\n" + "=" * 70)
print("\nTASK 12 — GridSearchCV + OOB Score")
print("=" * 70) 

param_grid ={
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth": [4, 6, 8, None],
    "clf__max_features": ["sqrt", "log2"],
}

rf_for_grid =Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("clf", RandomForestClassifier(oob_score=True, random_state=42)),
])

grid = GridSearchCV(
    rf_for_grid,param_grid,
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42),
    scoring="f1",n_jobs=-1,
)
grid.fit(X_train,y_train)

print(f"\n Best parameter : {grid.best_params_}")
print(f"\n Best CV f1 : {grid.best_score_:.4f}")

best_rf =grid.best_estimator_.named_steps["clf"]
print(f"OOB score : {best_rf.oob_score_}")


print("\n" + "=" * 70)
print("TASK 13: Regression side-task - predict fare")
print("=" * 70)

reg_df = df.drop(columns=["fare"])
reg_target =df["fare"]


reg_numeric =["age","sibsp","parch"]
reg_categorical =["sex", "embarked", "pclass"]

reg_preprocessor =ColumnTransformer(transformers=[
    ("num",Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]),reg_numeric),
    ("cat",Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]),reg_categorical)
])

Xr_train ,Xr_test,yr_train,yr_test =train_test_split(
    reg_df,reg_target,test_size=0.20,random_state=42,
)

reg_pipeline =Pipeline(steps=[
    ("preprocessor", reg_preprocessor),
    ("reg", LinearRegression()),
])

reg_pipeline.fit(Xr_train,yr_train)
yr_pred = reg_pipeline.predict(Xr_test)

mae =mean_absolute_error(yr_test, yr_pred)
rmse=np.sqrt(mean_squared_error(yr_test, yr_pred))
r2=r2_score(yr_test, yr_pred)
n=len(yr_test)
p=Xr_test.shape[1]
adj_r2 =1-(1-r2)*(n-1)/(n-p-1)

print(f"MSE :      {mae:.4f}")
print(f"RMSE :     {rmse:.4f}")
print(f"R2  :      {r2:.4f}")
print(f"Adjust r2 : {adj_r2:.4f}")

residuals =yr_test-yr_pred

plt.figure(figsize=(8,6))
plt.scatter(yr_pred,residuals,alpha=0.6)
plt.axhline(0,color="red",linestyle="--")
plt.xlabel("Predicted fare")
plt.ylabel("Residuals")
plt.title("Residual plot - Linear Regression")
plt.tight_layout()
plt.savefig("chart10_residuals.png")
plt.close()
print("saved chart10_residuals.png")

print("\n" + "=" * 70)
print("TASK 14 — Model Comparison Table")
print("=" * 70)

classif_df = pd.DataFrame([
    {"model":r["name"],"Accuracy":r["accuracy"],
     "Precision":r["precision"],"Recall":r["recall"],
     "F1": r["f1"], "AUC": r["auc"]}
    for r in [lr_result, dt_result, rf_result]
])

print("\nClassification metrics:")
print(classif_df.round(4).to_string(index=False))

reg_df_metrics =pd.DataFrame([
    {"Model":"Linear Regression","MAE": mae, "RMSE": rmse,
     "R2":r2,"Adjust_R2":adj_r2}
])

print("\nRegression metrics separate scale:")
print(reg_df_metrics.round(4).to_string(index=False))

print("\n" + "=" * 70)
print("TASK 15: Save full pipeline + reload verification")
print("=" * 70)

best_pipeline =grid.best_estimator_
joblib.dump(best_pipeline,"final_pipeline.joblib")
print("Saved final_pipeline.joblib")

loaded=joblib.load("final_pipeline.joblib")
sample_raw =X_test.head(5)
preds =loaded.predict(sample_raw)
probs =loaded.predict_proba(sample_raw)[:,1]

print("\nReload verification on 5 raw rows:")

for i ,(pr,prob) in enumerate(zip(preds,probs)):
    print(f"row{i}: prediction={pr},probability_survied={prob:.4f}")


print("\n" + "=" * 70)
print("PART B COMPLETE")
print("=" * 70)