import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

df = pd.read_csv("Telco_Customer_Churn_Dataset  (3).csv")

df_seg = df[['tenure', 'MonthlyCharges', 'Contract', 'Churn']].copy()
print(df_seg.head())
df_seg['TenureGroup'] = pd.cut(
    df_seg['tenure'],
    bins=[0, 12, 24, 48, 72],
    labels=['0-12', '13-24', '25-48', '49-72']
)
df_seg['ChargeGroup'] = pd.cut(
    df_seg['MonthlyCharges'],
    bins=[0, 35, 70, 120],
    labels=['Low', 'Medium', 'High']
)
tenure_churn = pd.crosstab(
    df_seg['TenureGroup'],
    df_seg['Churn'],
    normalize='index'
) * 100

print("\nChurn Rate by Tenure Group (%):")
print(tenure_churn)

charge_churn = pd.crosstab(
    df_seg['ChargeGroup'],
    df_seg['Churn'],
    normalize='index'
) * 100

print("\nChurn Rate by Monthly Charges Group (%):")
print(charge_churn)

contract_churn = pd.crosstab(
    df_seg['Contract'],
    df_seg['Churn'],
    normalize='index'
) * 100

print("\nChurn Rate by Contract Type (%):")
print(contract_churn)
sns.countplot(x='TenureGroup', hue='Churn', data=df_seg)
plt.title("Churn by Tenure Group")
plt.show()

sns.countplot(x='ChargeGroup', hue='Churn', data=df_seg)
plt.title("Churn by Monthly Charges Group")
plt.show()

sns.countplot(x='Contract', hue='Churn', data=df_seg)
plt.title("Churn by Contract Type")
plt.xticks(rotation=15)
plt.show()

high_value_risk = df_seg[
    (df_seg['ChargeGroup'] == 'High') &
    (df_seg['TenureGroup'].isin(['25-48', '49-72'])) &
    (df_seg['Churn'] == 'Yes')
]

print("\nHigh-Value Customers at Risk:")
print(high_value_risk.head())
print("Total High-Value Customers at Risk:", len(high_value_risk))

df_seg.to_csv("segmented_customers.csv", index=False)