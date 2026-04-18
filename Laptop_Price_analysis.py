import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind

df = pd.read_csv("laptop_prices.csv")
df['Ram'] = df['Ram'].astype(str).str.replace('GB', '').astype(int)
print("\n===== INFO =====")
print(df.info())

print("\n===== DESCRIBE =====")
print(df.describe())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

df = df[['Company', 'TypeName', 'Ram', 'Weight', 'Inches','CPU_freq', 'PrimaryStorage', 'Price_euros']].dropna()
features = ['Ram', 'Weight', 'Inches', 'CPU_freq', 'PrimaryStorage', 'Price_euros']
#Skewness
for col in features:
    plt.figure()
    sns.histplot(df[col], kde=True, color='skyblue', edgecolor='black')

    plt.title(f"{col} Distribution (Skewness: {round(df[col].skew(), 2)})")
    plt.xlabel(col)
    plt.ylabel("Frequency")

    plt.show()

# BOXPLOT
df_original = df.copy()
# IQR function
def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return data[(data[column] >= lower) & (data[column] <= upper)]

df_clean = df.copy()
for col in features:
    temp = remove_outliers(df, col)
    df_clean = df_clean[df_clean.index.isin(temp.index)]
for col in features:
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    sns.boxplot(y=df_original[col], color='skyblue')
    plt.title(f"Before Outlier Removal\n{col}")

    plt.subplot(1, 2, 2)
    sns.boxplot(y=df_clean[col], color='lightgreen')
    plt.title(f"After Outlier Removal\n{col}")

    plt.tight_layout()
    plt.show()
df = df_clean

# HEATMAP
numeric_df = df.select_dtypes(include=np.number)
plt.figure(figsize=(10, 6))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    linewidths=0.5,
    square=True,
    cbar_kws={"shrink": 0.8}
)

plt.title("Correlation Heatmap of Numerical Features", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()

# OBJECTIVE 1: RAM vs PRICE

sns.regplot(x='Ram', y='Price_euros', data=df)
plt.title("RAM vs Price")
plt.show()
print("Correlation:", df['Ram'].corr(df['Price_euros']))
X = df[['Ram']]
y = df['Price_euros']

slr_model = LinearRegression()
slr_model.fit(X, y)
slope = slr_model.coef_[0]
intercept = slr_model.intercept_

print(f"Equation: Price = {slope:.2f} * Ram + {intercept:.2f}")
from sklearn.metrics import r2_score

y_pred = slr_model.predict(X)
r2 = r2_score(y, y_pred)

print(f"R² Score: {r2:.4f}")
# -------------------------------
# Prediction (example: 8GB RAM)
# -------------------------------
new_ram = pd.DataFrame([[8]], columns=['Ram'])
predicted_price = slr_model.predict(new_ram)

print(f"Predicted Price for 8GB RAM: {predicted_price[0]:.2f} euros")
# Plot
sns.scatterplot(x=X['Ram'], y=y, alpha=0.6, label="Actual Data")
plt.plot(X, slr_model.predict(X), color='red', label="Regression Line")
plt.scatter(new_ram, predicted_price, color='green', s=100, label="Predicted Point")
plt.text(8, predicted_price[0], f"({8}, {predicted_price[0]:.0f})")
plt.xlabel("RAM (GB)")
plt.ylabel("Price (euros)")
plt.title("Simple Linear Regression with Prediction")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
# OBJECTIVE 2: Pie chart
type_counts = df['TypeName'].value_counts()
plt.figure(figsize=(10, 7))

plt.pie(
    type_counts,
    labels=type_counts.index,
    autopct='%1.1f%%',
    startangle=140,
    wedgeprops={'edgecolor': 'black'},
    pctdistance=0.8
)
plt.title("Distribution of Laptop Types", fontsize=15, pad=20)
plt.tight_layout()
plt.show()

# OBJECTIVE 3: ram count
df2 = df['Ram'].value_counts().head().reset_index()
df2.columns = ['Ram', 'Count']
plt.figure(figsize=(8,5))
sns.barplot(data=df2, x='Count', y='Ram', hue='Ram', palette='magma', legend=False)
plt.xlabel("Number of Laptops")
plt.ylabel("RAM (GB)")
plt.title("Top RAM Configurations in Laptops")
for i, v in enumerate(df2['Count']):
    plt.text(v + 2, i, str(v), va='center')
plt.subplots_adjust(left=0.2, right=0.9)
plt.show()

# OBJECTIVE 4: bar graph
company_counts = df['Company'].value_counts()
plt.figure(figsize=(12,6))

company_counts.plot(kind='bar',color=plt.cm.Set2.colors)
plt.xlabel("Company")
plt.ylabel("Count")
plt.title("Number of Laptops by Company")
plt.xticks(rotation=45, ha='right')
plt.subplots_adjust(left=0.2, right=0.9)
plt.show()

# OBJECTIVE 5: HYPOTHESIS TESTING (T-TEST)
low_ram = df[df['Ram'] <= 8]['Price_euros']
high_ram = df[df['Ram'] > 8]['Price_euros']
t_stat, p_value = ttest_ind(low_ram, high_ram)

# Display results
print("\n===== T-TEST RESULT =====")
print(f"T-Statistic: {t_stat:.4f}")
print(f"P-Value: {p_value:.4f}")
alpha = 0.05

if p_value < alpha:
    print("Conclusion: Reject Null Hypothesis (Significant Difference in Prices)")
else:
    print("Conclusion: Fail to Reject Null Hypothesis (No Significant Difference)")