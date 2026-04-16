import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind

df = pd.read_csv("laptop_prices.csv")

print("\n===== INFO =====")
print(df.info())

print("\n===== DESCRIBE =====")
print(df.describe())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

df = df[['Ram', 'Weight', 'Inches', 'CPU_freq', 'PrimaryStorage', 'Price_euros']].dropna()
features = ['Ram', 'Weight', 'Inches', 'CPU_freq', 'PrimaryStorage', 'Price_euros']



# =========================
# HISTOGRAM + KDE
# =========================
#Skewness
for col in features:
    plt.figure()
    sns.histplot(df[col], kde=True)

    plt.title(f"{col} Distribution (Skewness: {round(df[col].skew(), 2)})")
    plt.xlabel(col)
    plt.ylabel("Frequency")

    plt.show()



# =========================
# BOXPLOT
# =========================
for col in features:
    plt.figure()
    sns.boxplot(y=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return df[(df[column] >= lower) & (df[column] <= upper)]
for col in features:
    plt.figure()
    sns.boxplot(y=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()
# =========================
# HEATMAP
# =========================
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# =========================
# OBJECTIVE 1: RAM vs PRICE
# =========================
sns.regplot(x='Ram', y='Price_euros', data=df)
plt.title("RAM vs Price")
plt.show()

print("Correlation:", df['Ram'].corr(df['Price_euros']))
# Data cleaning (if needed)
df['Ram'] = df['Ram'].astype(str).str.replace('GB', '').astype(int)

# Features
X = df[['Ram']]
y = df['Price_euros']

# Train model
slr_model = LinearRegression()
slr_model.fit(X, y)

# Coefficients
slope = slr_model.coef_[0]
intercept = slr_model.intercept_

print(f"Equation: Price = {slope:.2f} * Ram + {intercept:.2f}")

# -------------------------------
# Prediction (example: 8GB RAM)
# -------------------------------
new_ram = [[8]]
predicted_price = slr_model.predict(new_ram)

print(f"Predicted Price for 8GB RAM: {predicted_price[0]:.2f} euros")

# -------------------------------
# Plot
# -------------------------------
plt.scatter(X, y, label="Actual Data")

# Regression line
plt.plot(X, slr_model.predict(X), color='red', label="Regression Line")

# Predicted point
plt.scatter(new_ram, predicted_price, color='green', s=100, label="Predicted Point")

plt.text(8, predicted_price[0], f"({8}, {predicted_price[0]:.0f})")
plt.xlabel("RAM (GB)")
plt.ylabel("Price (euros)")
plt.title("Simple Linear Regression with Prediction")
plt.legend()

plt.show()
# =========================
# OBJECTIVE 2: Pie chart
# =========================
import matplotlib.pyplot as plt

# Data
type_counts = df['TypeName'].value_counts()

# Plot
plt.figure(figsize=(10, 7))

plt.pie(type_counts,labels=type_counts.index,autopct='%1.1f%%',startangle=140,colors=plt.cm.Pastel1.colors, 
    explode=[0.03] * len(type_counts), 
    shadow=True,
    pctdistance=0.85 )
plt.title("Distribution of Laptop Types", fontsize=15, pad=20)
plt.tight_layout()
plt.show()
# =========================
# OBJECTIVE 3: ram count
# =========================
df2 = df['Ram'].value_counts().head().reset_index()
df2.columns = ['Ram', 'Count']
plt.figure(figsize=(8,5))

sns.barplot(data=df2,x='Count',y='Ram',palette='magma')
plt.xlabel("Number of Laptops")
plt.ylabel("RAM (GB)")
plt.title("Top RAM Configurations in Laptops")
for i, v in enumerate(df2['Count']):
    plt.text(v + 2, i, str(v), va='center')
plt.tight_layout()
plt.show()
# =========================
# OBJECTIVE 4: bar graph
# =========================

company_counts = df['Company'].value_counts()
plt.figure(figsize=(12,6))

company_counts.plot(kind='bar',color=plt.cm.Set2.colors)
plt.xlabel("Company")
plt.ylabel("Count")
plt.title("Number of Laptops by Company")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
# =========================
# OBJECTIVE 5: HYPOTHESIS TESTING (T-TEST)
# =========================

# Create groups
low_ram = df[df['Ram'] <= 8]['Price_euros']
high_ram = df[df['Ram'] > 8]['Price_euros']

# Perform t-test
t_stat, p_value = ttest_ind(low_ram, high_ram)

# Display results
print("\n===== T-TEST RESULT =====")
print(f"T-Statistic: {t_stat:.4f}")
print(f"P-Value: {p_value:.4f}")

# Decision
alpha = 0.05

if p_value < alpha:
    print("Conclusion: Reject Null Hypothesis (Significant Difference in Prices)")
else:
    print("Conclusion: Fail to Reject Null Hypothesis (No Significant Difference)")