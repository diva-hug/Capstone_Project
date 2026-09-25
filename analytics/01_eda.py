""" Module 2 - Analytics pipeline

Load, profile, clean ,EDA , save Titanic.csv"""

import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt


sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100


# TASK 1 : Load and Profile the dataset
print("TASK 1 :- Load and Profile the dataset")

df = sns.load_dataset("titanic")

# immediately  save  offline fallback 
df.to_csv("titanic.csv", index=False)
print("Saved titanic.csv  as offline fallback ")

print("\n --------df.shape-----------")
print(df.shape)
print("\n --------df.info()-----------")
print(df.info())
print("\n --------df.describe()-----------")
print(df.describe())

print("\n -- Missing values for column")
miss_pct =(df.isnull().sum()/len(df))*100
miss_pct =miss_pct[miss_pct>0].sort_values(ascending=False)
print(miss_pct)


print("\n Class Balncees (survied)-------")
print(df["survived"].value_counts(normalize=True))

#Apply missing-value handling per column, following this threshold rule 
print("\n" + "=" * 70)
print("TASK 2: Missing-value handling per threshold rule")
print("=" * 70)

missing =(df.isnull().sum()/len(df))*100
missing =missing[missing>0]
print("\n  Missing values before cleaning ")
print(missing)

# decision per columns  based on threshold  rule

deck_pct = missing.get("deck",0)
age_pct =missing.get("age",0)
emb_pct =missing.get("embarked",0)

print(f"\n deck: {deck_pct:.2f}% missing -> ABOVE 30%  -> DROP column (imputation  unrealiable)")
df=df.drop(columns=["deck"])

print(f"\n age: {age_pct:.2f}%  missing ->  5-30%--> impute with  median")
df["age"]=df["age"].fillna(df["age"].median())

print(f"\n embarked: {emb_pct:.2f}% missing ---> under 5% drop those  rows ")
df = df.dropna(subset=["embarked"])

print(f"\n shape after cleaning : {df.shape}")
print("reamaing missing :",df.isnull().sum().sum())


#Univariate analysis: plot a histogram and a box plot for both
print("\n" + "=" * 70)
print("TASK 3: Univariate analysis for age and fare")
print("=" * 70)

def iqr_outliers(series):
    q1 = series.quantile(0.25)
    q3 =series.quantile(0.75)
    iqr = q3 -q1 
    lower =q1 - 1.5 * iqr
    upper =q3 + 1.5 * iqr 
    count =((series<lower)|(series>upper)).sum()
    return q1, q3, iqr,lower, upper,count 

for col in ["age","fare"]:
    q1,q3, iqr, lower,upper,count = iqr_outliers(df[col])
    print(f"\n--{col}---")
    print(f"Q1={q1:.2f},Q3={q3:.2f},IQR={iqr:.2f}")
    print(f"lower bound ={lower:.2f},upper bound={upper:.2f}")
    print(f"Outliers (IQR rule):{count}")


fig, axes = plt.subplots(2,2,figsize=(12,8))
axes [0,0].hist(df["age"],bins=30,edgecolor="black")
axes [0,0].set_title("Age-Histogram")
axes [0,0].set_xlabel("Age")

axes [0,1].boxplot(df["age"].dropna())
axes [0,1].set_title("Age-Box plot ")

axes [1,0].hist(df["fare"],bins=30,edgecolor="black")
axes [1,0].set_title("Fare -Histogram")
axes [1,0].set_xlabel("Fare")

axes[1,1].boxplot(df["fare"].dropna())
axes[1,1].set_title("Fare -Box plot")

plt.tight_layout()
plt.savefig("chart1_univariate.png")
plt.close()
print("\nchart1_univariate.png")

fare_mean =df["fare"].mean()
fare_median =df["fare"].median()
fare_mode = df["fare"].mode()[0]


print(f"\nFare -mean ={fare_mean:.2f},median={fare_median:.2f},mode={fare_mode:.2f}")
if fare_mean>fare_median>fare_mode:
    print("Distribution: Right-skewed (mean > median >mode).")

elif fare_mean < fare_median < fare_mode:
    print("Distribution: left-skewed (mean < median < mode).")
else:
    print("Distribution: approximately symmetric.")


# Task -4 Bivariate analysis
print("\n" + "=" * 70)
print("TASK 4: Bivariate analysis")
print("=" * 70)

print(f"\n Survival rate  by sex:")
print(df.groupby("sex")["survived"].mean())

print(f"\n Survival rate  by pclass:")
print(df.groupby("pclass")["survived"].mean())

print(f"\n Survival rate  by  sex and pclass:")
print(df.groupby(["sex","pclass"])["survived"].mean())

print("\n boolean masking ")  

print(f"\n Female survivors: {len(df[(df['sex'] == 'female') & (df['survived'] == 1)])}")
print(f" First class OR fare>100: {len(df[(df['pclass'] == 1) | (df['fare'] > 100)])}")

cols =["survived","pclass","age","sibsp","parch","fare"]
corr = df[cols].corr()
print("\nCorrelation matrix (6 columns):")
print(corr) 


plt.figure(figsize=(8,6))
sns.heatmap(corr,annot=True,cmap="coolwarm",center=0,fmt=".2f")
plt.title("correlation Matrix (6 features)")
plt.tight_layout()
plt.savefig("chart2_corr_heatmap.png")
plt.close()

# srong diagonal correlations 

off_diag =corr.where(~np.eye(len(corr),dtype=bool)).abs().unstack().sort_values(ascending=False)
seen =set()
top_pairs =[]
for (a,b), val in off_diag.items():
    key =tuple(sorted([a,b]))
    if key in seen:
        continue
    seen.add(key)
    top_pairs.append((a,b,val))
    if len(top_pairs)==2:
        break 
print("\n Top 2  strong correclations ")

for a,b,v in top_pairs:
    print(f" {a}->{b}:  {v:.3f}")


#Multivariate "data story": produce at least 4 distinct charts (any combination of  bar/box/scatter/heatmap/pair-plot)
print("\n" + "=" * 70)
print("TASK 5: Multivariate  data story ")
print("=" * 70)

plt.figure(figsize=(8,5))
sns.barplot(data=df,x="pclass",y="survived",hue="sex",errorbar=None)
plt.title("Survival rate  by sex and plcass")
plt.ylabel("Survival rate")
plt.tight_layout()
plt.savefig("chart3_survival_by_sex_pclass.png")
plt.close()

plt.figure(figsize=(8,5))
sns.barplot(data=df,x="survived",y="age",hue="sex")
plt.title(" Age distrubtion  by Survival ")
plt.xlabel("Survival (0=no ,1 =yes)")
plt.tight_layout()
plt.savefig("chart4_age_by_survival.png")
plt.close()


plt.figure(figsize=(8,5))
sns.scatterplot(data=df,x="age",y="fare",hue="survived",alpha=0.6)
plt.title(" Fare vs Age ,colored by Survival ")
plt.tight_layout()
plt.savefig("chart5_fare_age_survival.png")
plt.close()


pair_colors =["survived", "age", "fare", "pclass"]
sns.pairplot(df[pair_colors],hue="survived",diag_kind="hist")
plt.savefig("chart6_pairplot.png")
plt.close()


print("saved chart 3 - chart 6 (4 multivariate charts)")

#z-score formula  z = (x − mean) / std on the full cleaned DataFrame

print("\n" + "=" * 70)
print("TASK 5: Z score standardization")
print("=" * 70)

print("\n before  standardization:")
print(df[["age","fare"]].agg(["mean","std"]))

# Compute z-scores: z = (x - mean) / std
df_std = df.copy()
for col in ["age", "fare"]:
    col_mean = df_std[col].mean()
    col_std = df_std[col].std()
    df_std[col] = (df_std[col] - col_mean) / col_std

print("\n After  standardization:")
print(df_std[["age","fare"]].agg(["mean","std"]))

fig, axes =plt.subplots(2,2,figsize=(12,8))
axes[0,0].hist(df["age"],bins=30)
axes[0,0].set_title("Age — before z-score")
axes[0,1].hist(df_std["age"],bins=30)
axes[0,1].set_title("Age — after z-score")
axes[1,0].hist(df_std["age"],bins=30)
axes[1,0].set_title("Fare — before")
axes[1,1].hist(df_std["fare"], bins=30)
axes[1,1].set_title("Fare — after z-score")
plt.tight_layout()
plt.savefig("chart7_standardization.png")
plt.close()


# save cleaned datafrom  for modeling stage

df.to_csv("titanic_clean.csv", index=False)
print("\n Saved titanic  for modeling  stage")


print("\n" + "=" * 70)
print("Complete")
print("=" * 70)

