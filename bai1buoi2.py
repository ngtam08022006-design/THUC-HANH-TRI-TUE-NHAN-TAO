from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Tải dữ liệu và tạo dataframe
diabetes = load_diabetes(as_frame=True)
X = diabetes.data.copy()
y = diabetes.target.copy()
df = X.copy()
df["target"] = y

# Khai báo 3 feature đã chọn
selected_features = ["bmi", "bp", "s5"]
for col in selected_features:
    plt.figure(figsize=(6, 4))
    plt.hist(df[col], bins=25, edgecolor="black", alpha=0.8)
    plt.title(f"Phân phối của {col}")
    plt.xlabel(col)
    plt.ylabel("Tần suất")
    plt.grid(alpha=0.2)
    plt.show()
outlier_summary = []
for col in selected_features:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_summary.append([col, n_outliers, lower, upper])

outlier_df = pd.DataFrame(
    outlier_summary,
    columns=["Biến", "Số ngoại lệ theo IQR", "Ngưỡng dưới", "Ngưỡng trên"]
)
print(outlier_df)
corr = df[selected_features + ["target"]].corr(numeric_only=True)
target_corr = corr["target"].drop("target").sort_values(key=np.abs, ascending=False)
print("Tương quan với target:")
print(target_corr.to_frame("corr_with_target"))
for col in selected_features:
    plt.figure(figsize=(6, 4))
    plt.scatter(df[col], df["target"], alpha=0.6)
    plt.xlabel(col)
    plt.ylabel("target")
    plt.title(f"{col} và target")
    plt.grid(alpha=0.2)
    plt.show()